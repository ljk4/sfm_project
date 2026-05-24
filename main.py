"""
主入口: 三维重建管线 (Structure from Motion)

流程:
  1. EXIF 自标定 → 精确内参 K
  2. SIFT + FLANN 特征匹配
  3. 增量式 SfM (位姿恢复 + 发散检测)
  4. 迭代 PnP 光束法平差 + 多视图重三角化
  5. 重投影 + 距离双重过滤
  6. 导出 (PLY点云 + 相机视锥)
  7. open3d 交互可视化
"""

import sys, time
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from feature_matching import run_feature_matching
from sfm_pipeline import run_sfm
from bundle_adjustment import run_bundle_adjustment
from export_results import run_export
from visualize import run_visualization
from self_calibration import run_self_calibration


def filter_outliers(points3D, point_colors, Rs, ts, reg_indices, max_dist_ratio=50.0):
    """距离鲁棒过滤 + 负深度检查"""
    n_before = len(points3D)
    dist = np.linalg.norm(points3D, axis=1)
    med = np.median(dist); mad = np.median(np.abs(dist - med)) + 1e-10
    keep = dist < (med + max_dist_ratio * mad)
    for orig_idx in reg_indices[:5]:
        if orig_idx in Rs:
            R, t = Rs[orig_idx], ts[orig_idx]
            keep &= ((R @ points3D.T + t.reshape(3, 1))[2] > 0)
    pts = points3D[keep]; clrs = point_colors[keep]
    print(f"  [距离过滤] {n_before} -> {len(pts)} 点 (移除{n_before-len(pts)}, {(n_before-len(pts))/n_before*100:.1f}%)")
    return pts, clrs


def filter_by_reprojection(points3D, point_colors, Rs, ts, K, observations, max_err=5.0):
    """重投影误差过滤: 移除在所有观测中中位误差 > max_err 的点"""
    if not observations: return points3D, point_colors
    n = len(points3D)
    point_errs = {i: [] for i in range(n)}
    for orig_idx, _, pt3d_idx, u_obs, v_obs in observations:
        if pt3d_idx >= n or orig_idx not in Rs: continue
        X = points3D[pt3d_idx]; R, t = Rs[orig_idx], ts[orig_idx]
        x = R @ X + t.ravel()
        if x[2] <= 0: continue
        u = K @ x; up, vp = u[0]/u[2], u[1]/u[2]
        point_errs[pt3d_idx].append(np.sqrt((up-u_obs)**2 + (vp-v_obs)**2))
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if point_errs[i] and np.median(point_errs[i]) > max_err:
            keep[i] = False
    pts = points3D[keep]; clrs = point_colors[keep]
    all_med = [np.median(point_errs[i]) for i in range(n) if point_errs[i]]
    if all_med:
        all_med = np.array(all_med)
        print(f"  [重投影过滤] {n} -> {len(pts)} 点 "
              f"(移除{n-len(pts)}, 误差: 均值={all_med.mean():.2f}px 中位数={np.median(all_med):.2f}px)")
    return pts, clrs


def main():
    t_start = time.time()
    IMAGE_DIR = Path(__file__).parent / "Cat_RGB"
    OUTPUT_DIR = Path(__file__).parent / "output"
    MAX_DIM = 1200
    DO_BA, DO_VISUALIZE = True, True

    print("=" * 60)
    print("三维重建管线 (Structure from Motion)")
    print("=" * 60)

    # ====== 阶段0: 自标定 ======
    K_full, img_w_full, img_h_full, distortion_k1 = run_self_calibration(
        str(IMAGE_DIR))
    print(f"  EXIF自标定: f={K_full[0,0]:.1f}px (基于{img_w_full}x{img_h_full})")

    # ====== 阶段1: 特征提取与匹配 ======
    images, all_kp, all_des, matches_list, scales = \
        run_feature_matching(str(IMAGE_DIR), max_dim=MAX_DIM)

    h_curr, w_curr = images[0].shape[:2]
    scale_k = w_curr / img_w_full
    K = K_full.copy()
    K[0] *= scale_k; K[1] *= scale_k
    print(f"  K缩放至{w_curr}x{h_curr}: fx={K[0,0]:.1f}")

    # ====== 阶段2: 增量式 SfM ======
    Rs, ts, K, points3D, point_colors, reg_indices, observations = run_sfm(
        images, all_kp, all_des, matches_list, K_input=K)

    if len(reg_indices) < 2:
        print("\n[错误] 注册相机不足")
        return

    # ====== 阶段3: 光束法平差 ======
    if DO_BA and observations:
        Rs, ts, K, points3D = run_bundle_adjustment(
            Rs, ts, K, points3D, point_colors,
            reg_indices, all_kp, all_des, observations, max_iterations=3)

    # ====== 阶段4: 点云过滤 ======
    print("\n" + "=" * 60)
    print("点云过滤")
    print("=" * 60)
    points_filtered, colors_filtered = filter_by_reprojection(
        points3D, point_colors, Rs, ts, K, observations, max_err=5.0)
    points_filtered, colors_filtered = filter_outliers(
        points_filtered, colors_filtered, Rs, ts, reg_indices)

    # ====== 阶段5: 导出 ======
    run_export(str(OUTPUT_DIR), points_filtered, colors_filtered,
               Rs, ts, K, reg_indices, images)

    # ====== 阶段6: 可视化 ======
    if DO_VISUALIZE:
        try:
            run_visualization(points_filtered, colors_filtered,
                              Rs, ts, K, reg_indices, images, str(OUTPUT_DIR))
        except Exception as e:
            print(f"  [可视化] {e}")

    t_total = time.time() - t_start
    print(f"\n重建完成! {len(reg_indices)}台相机 {len(points_filtered)}个稀疏点"
          f" {t_total:.0f}s")
    print(f"结果: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
