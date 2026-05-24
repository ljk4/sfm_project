"""
主入口: 三维重建管线 (增强版)
用法: python main.py

流程:
  1. SIFT特征提取 + 交叉检查匹配
  2. 自动选最佳初始对 → 增量式SfM (严格PnP)
  3. 光束法平差 (固定参考系, motion-only)
  4. 重投影误差 + 距离离群点过滤
  5. 稠密重建 (直接SGBM)
  6. 导出结果 (PLY + 视锥)
  7. open3d交互可视化
"""

import sys, time
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from feature_matching import run_feature_matching
from sfm_pipeline import run_sfm
from bundle_adjustment import run_bundle_adjustment
from dense_reconstruction import run_dense_reconstruction
from export_results import run_export
from visualize import run_visualization


def check_camera_orientation(Rs, ts, point_cloud_center):
    """
    检查相机朝向: 光轴应大致指向点云中心
    返回: 每个相机是否合格
    """
    results = {}
    for idx in Rs:
        R, t = Rs[idx], ts[idx]
        cam_center = (-R.T @ t).ravel()
        # 相机光轴 (Z轴方向)
        optical_axis = R[2, :]  # 旋转矩阵第三行 = 相机Z轴在世界坐标的方向
        # 相机到点云中心的向量
        to_center = point_cloud_center - cam_center
        to_center = to_center / (np.linalg.norm(to_center) + 1e-10)
        # 光轴应指向点云中心: 点积应 > 0
        dot_product = np.dot(optical_axis, to_center)
        results[idx] = dot_product > -0.3  # 允许一定的偏差
    return results


def filter_outliers(points3D, point_colors, Rs, ts, reg_indices,
                    max_dist_ratio=50.0):
    """距离鲁棒过滤 + 负深度过滤"""
    n_before = len(points3D)
    dist = np.linalg.norm(points3D, axis=1)
    med = np.median(dist)
    mad = np.median(np.abs(dist - med)) + 1e-10
    keep = dist < (med + max_dist_ratio * mad)
    for orig_idx in reg_indices[:5]:
        if orig_idx in Rs:
            R, t = Rs[orig_idx], ts[orig_idx]
            X_cam = R @ points3D.T + t.reshape(3, 1)
            keep &= (X_cam[2] > 0)
    pts = points3D[keep]
    clrs = point_colors[keep]
    print(f"  [距离过滤] {n_before} → {len(pts)} 点 "
          f"(移除{n_before-len(pts)}, {(n_before-len(pts))/n_before*100:.1f}%)")
    return pts, clrs


def filter_by_reprojection(points3D, point_colors, Rs, ts, K, observations,
                            max_reproj_error=5.0):
    """重投影误差过滤: 移除在所有观测中误差>阈值的点"""
    if not observations:
        return points3D, point_colors

    n_points = len(points3D)
    point_errors = {i: [] for i in range(n_points)}

    for orig_idx, kp_idx, pt3d_idx, u_obs, v_obs in observations:
        if pt3d_idx >= n_points or orig_idx not in Rs:
            continue
        X = points3D[pt3d_idx]
        R, t = Rs[orig_idx], ts[orig_idx]
        x_cam = R @ X + t.ravel()
        if x_cam[2] <= 0:
            continue
        u_pred = K @ x_cam
        u_p, v_p = u_pred[0] / u_pred[2], u_pred[1] / u_pred[2]
        err = np.sqrt((u_p - u_obs)**2 + (v_p - v_obs)**2)
        point_errors[pt3d_idx].append(err)

    keep = np.ones(n_points, dtype=bool)
    for i in range(n_points):
        if point_errors[i]:
            if np.median(point_errors[i]) > max_reproj_error:
                keep[i] = False

    pts = points3D[keep]
    clrs = point_colors[keep]

    all_med = [np.median(point_errors[i]) for i in range(n_points) if point_errors[i]]
    if all_med:
        all_med = np.array(all_med)
        print(f"  [重投影过滤] {n_points} → {len(pts)} 点 "
              f"(移除{n_points-len(pts)}个误差>{max_reproj_error}px的点)")
        print(f"    误差: 均值={all_med.mean():.2f}px 中位数={np.median(all_med):.2f}px "
              f"90分位={np.percentile(all_med,90):.2f}px")
    return pts, clrs


def main():
    t_start = time.time()

    IMAGE_DIR = Path(__file__).parent / "Cat_RGB"
    OUTPUT_DIR = Path(__file__).parent / "output"
    MAX_DIM, FOCAL_RATIO = 1200, 1.2
    DO_BA, DO_DENSE, DO_VISUALIZE = False, True, True  # BA暂时关闭(需进一步调优)

    print("=" * 60)
    print("三维重建管线 (增强版)")
    print("=" * 60)

    # ====== 阶段1: 特征提取与交叉检查匹配 ======
    images, all_kp, all_des, matches_list, scales = \
        run_feature_matching(str(IMAGE_DIR), max_dim=MAX_DIM)

    # ====== 阶段2: 增量式SfM (自动选初始对 + 严格PnP) ======
    Rs, ts, K, points3D, point_colors, reg_indices, observations = run_sfm(
        images, all_kp, all_des, matches_list, focal_ratio=FOCAL_RATIO)

    if len(reg_indices) < 2:
        print("\n[错误] 注册相机不足")
        return

    # ====== 阶段3: 光束法平差 (固定参考系) ======
    if DO_BA and observations:
        Rs, ts, K, points3D = run_bundle_adjustment(
            Rs, ts, K, points3D, point_colors,
            reg_indices, observations, max_obs=30000)

    # ====== 阶段4: 相机朝向检查 ======
    print("\n" + "=" * 60)
    print("阶段4: 相机质量检查")
    print("=" * 60)
    if len(points3D) > 0:
        cloud_center = np.median(points3D, axis=0)
        orientation_ok = check_camera_orientation(Rs, ts, cloud_center)
        bad_cams = [idx for idx, ok in orientation_ok.items() if not ok]
        if bad_cams:
            print(f"  [朝向] 移除{len(bad_cams)}台朝向异常的相机: {bad_cams}")
            # 注意: 移除相机会影响observations, 这里简单标记
            reg_indices = [idx for idx in reg_indices
                          if idx not in bad_cams or idx == 0]  # 保留相机0
        else:
            print(f"  [朝向] 所有{len(reg_indices)}台相机朝向正常")

    # ====== 阶段5: 点云过滤 ======
    print("\n" + "=" * 60)
    print("阶段5: 点云过滤")
    print("=" * 60)
    points_filtered, colors_filtered = filter_by_reprojection(
        points3D, point_colors, Rs, ts, K, observations, max_reproj_error=5.0)
    points_filtered, colors_filtered = filter_outliers(
        points_filtered, colors_filtered, Rs, ts, reg_indices)

    if len(points_filtered) < 10:
        print("[错误] 过滤后点数太少")
        return

    # ====== 阶段6: 稠密重建 ======
    dense_pts, dense_colors = None, None
    if DO_DENSE:
        dense_pts, dense_colors = run_dense_reconstruction(
            images, Rs, ts, K, reg_indices, max_pairs=8)

    # ====== 阶段7: 导出 ======
    run_export(str(OUTPUT_DIR), points_filtered, colors_filtered,
               Rs, ts, K, reg_indices, images,
               dense_pts=dense_pts, dense_colors=dense_colors)

    # ====== 阶段8: 可视化 ======
    if DO_VISUALIZE:
        try:
            run_visualization(points_filtered, colors_filtered,
                              Rs, ts, K, reg_indices, images, str(OUTPUT_DIR))
        except Exception as e:
            print(f"  [可视化] 出错: {e}")

    # 汇总
    t_total = time.time() - t_start
    print("\n" + "=" * 60)
    print("重建完成!")
    print(f"  相机: {len(reg_indices)}台  稀疏点: {len(points_filtered)}个"
          f"{f'  稠密点: {len(dense_pts)}个' if dense_pts is not None and len(dense_pts) > 0 else ''}")
    print(f"  K:\n{K}")
    print(f"  总耗时: {t_total:.1f}s  ({t_total/60:.1f}min)")
    print(f"  结果: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
