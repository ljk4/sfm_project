"""
模块4: 稠密三维重建 (直接SGBM, 无校正)
- 利用旋转很小的相邻帧直接进行SGBM匹配
- 使用SfM相机位姿进行反投影 (无需Q矩阵)
- 多个相邻对的结果合并
"""

import cv2
import numpy as np
import time


def dense_stereo_pair_direct(img1, img2, K, R1, t1, R2, t2, max_dim=600):
    """
    不经过立体校正，直接用SGBM计算视差

    原理:
      - 相邻帧旋转极小, 极线已接近水平
      - SGBM可近似在水平方向搜索匹配
      - 视差通过已知相机位姿三角化到三维

    返回: dense_points3D (Nx3), dense_colors (Nx3)
    """
    h, w = img1.shape[:2]
    scale = max_dim / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)

    img1_s = cv2.resize(img1, (new_w, new_h))
    img2_s = cv2.resize(img2, (new_w, new_h))
    gray1 = cv2.cvtColor(img1_s, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2_s, cv2.COLOR_BGR2GRAY)

    K_s = K.copy()
    K_s[0] *= scale
    K_s[1] *= scale

    # 直接SGBM (不校正, 因为相邻帧旋转极小)
    sgbm = cv2.StereoSGBM_create(
        minDisparity=-64,
        numDisparities=128,
        blockSize=3,
        P1=8 * 3 * 3 ** 2,
        P2=32 * 3 * 3 ** 2,
        disp12MaxDiff=2,
        uniquenessRatio=5,
        speckleWindowSize=100,
        speckleRange=32,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    disparity = sgbm.compute(gray1, gray2).astype(np.float32) / 16.0

    # 有效视差范围
    valid = (disparity > -63) & (disparity < 63)
    n_valid = valid.sum()

    if n_valid < 500:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)

    # 视差统计(用于自适应过滤)
    d_vals = disparity[valid]
    d_median, d_mad = np.median(d_vals), np.median(np.abs(d_vals - np.median(d_vals)))
    # 保留视差在合理范围内的点
    valid &= np.abs(disparity - d_median) < max(10, d_mad * 5)

    if valid.sum() < 200:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)

    # 反投影: 对每个有效像素, 用相机位姿三角化
    # 像素在img1的坐标 → 反投影到归一化平面 → 在img2中找到对应点 → 三角化
    fx, fy = K_s[0, 0], K_s[1, 1]
    cx, cy = K_s[0, 2], K_s[1, 2]

    # 采样像素 (避免点数过多)
    sample_step = max(1, int(np.sqrt(n_valid / 200000)))
    ys, xs = np.where(valid)
    if sample_step > 1:
        idx = np.arange(0, len(ys), sample_step)
        ys, xs = ys[idx], xs[idx]
    disparities = disparity[ys, xs]

    n_pts = len(xs)
    if n_pts > 300000:
        idx = np.random.choice(n_pts, 300000, replace=False)
        ys, xs, disparities = ys[idx], xs[idx], disparities[idx]
        n_pts = len(xs)

    # 用OpenCV向量化三角化 (快速)
    pts1_px = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])
    pts2_px = np.column_stack([(xs + disparities).astype(np.float64), ys.astype(np.float64)])

    P1 = (K_s @ np.hstack((R1, t1.reshape(3, 1)))).astype(np.float64)
    P2 = (K_s @ np.hstack((R2, t2.reshape(3, 1)))).astype(np.float64)

    pts4d = cv2.triangulatePoints(P1, P2, pts1_px.T, pts2_px.T)
    pts3d = (pts4d[:3] / pts4d[3]).T

    # 过滤: 在两个相机前方
    X1 = R1 @ pts3d.T + t1.reshape(3, 1)
    X2 = R2 @ pts3d.T + t2.reshape(3, 1)
    front = (X1[2] > 0) & (X2[2] > 0)

    # 过滤: 合理深度范围
    z_vals = X1[2]
    z_max = np.percentile(z_vals[front], 99)
    good = front & (z_vals > 0.01) & (z_vals < z_max)

    pts3d = pts3d[good]
    colors = img1_s[ys[good], xs[good]]

    if len(pts3d) < 100:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)

    print(f"      稠密: {len(pts3d)}点 ({n_valid/disparity.size*100:.1f}%有效)")

    return pts3d.astype(np.float32), colors


def run_dense_reconstruction(images, Rs, ts, K, reg_indices, max_pairs=8):
    """
    对均匀分布的相邻相机对运行稠密重建
    """
    print("\n" + "=" * 60)
    print("阶段5: 稠密重建 (直接SGBM)")
    print("=" * 60)

    n_images = len(reg_indices)
    if n_images < 2:
        print("  [稠密] 相机不足")
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)

    step = max(1, n_images // max_pairs)
    selected_pairs = []
    for i in range(0, n_images - 1, step):
        a_idx = reg_indices[i]
        b_idx = reg_indices[min(i + 1, n_images - 1)]
        if a_idx != b_idx and b_idx - a_idx <= 2:
            selected_pairs.append((a_idx, b_idx))

    selected_pairs = selected_pairs[:max_pairs]
    if not selected_pairs:
        selected_pairs = [(reg_indices[0], reg_indices[1])]

    print(f"  [稠密] {len(selected_pairs)}对: {selected_pairs}")
    t0 = time.time()

    all_pts, all_colors = [], []

    for a_idx, b_idx in selected_pairs:
        print(f"    处理 ({a_idx}, {b_idx})...", end='')
        pts, clrs = dense_stereo_pair_direct(
            images[a_idx], images[b_idx], K,
            Rs[a_idx], ts[a_idx], Rs[b_idx], ts[b_idx]
        )
        all_pts.append(pts)
        all_colors.append(clrs)

    all_pts = [p for p in all_pts if len(p) > 0]
    all_colors = [c for c in all_colors if len(c) > 0]

    if all_pts:
        dense_pts = np.vstack(all_pts)
        dense_colors = np.vstack(all_colors)
    else:
        dense_pts = np.zeros((0, 3), dtype=np.float32)
        dense_colors = np.zeros((0, 3), dtype=np.uint8)

    t1 = time.time()
    print(f"  [稠密] 完成: {len(dense_pts)}点, {t1-t0:.1f}s")

    return dense_pts, dense_colors
