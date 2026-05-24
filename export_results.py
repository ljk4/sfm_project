"""
模块5: 导出结果
- 保存稀疏/稠密点云为 PLY 文件 (含顶点颜色)
- 生成相机视锥棱线并嵌入 PLY (红色线条)
- 保存相机参数为 NPZ 文件
"""

import numpy as np
import os
from pathlib import Path


def save_ply(filepath, points, colors, normals=None):
    """
    保存点云为 PLY 格式 (ASCII)

    PLY 格式:
      头部声明顶点数和属性
      数据行: x y z r g b (nx ny nz 可选)

    参数:
      points: Nx3 浮点数
      colors: Nx3 uint8 (0-255)
      normals: Nx3 浮点数 (可选)
    """
    n = len(points)
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        if normals is not None:
            f.write("property float nx\n")
            f.write("property float ny\n")
            f.write("property float nz\n")
        f.write("end_header\n")

        for i in range(n):
            x, y, z = points[i]
            r, g, b = colors[i]
            if normals is not None:
                nx, ny, nz = normals[i]
                f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)} {nx:.6f} {ny:.6f} {nz:.6f}\n")
            else:
                f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")

    size_mb = filepath.stat().st_size / (1024 * 1024)
    print(f"  [保存] {filepath.name}: {n} 个顶点, {size_mb:.1f} MB")


def generate_frustum_points(K, R, t, w, h, depth=1.5, n_samples_per_edge=100):
    """
    生成相机视锥的棱线采样点集 (用于嵌入PLY可视化)

    视锥由 5 个顶点构成:
      - 相机光心 (世界坐标)
      - 像平面四角在 depth 深度的投影

    8 条边:
      - 4 条从光心到远平面四角
      - 4 条连接远平面四边

    参数:
      K: 内参 (3x3)
      R, t: 外参 (世界→相机, 即 X_cam = R*X_world + t)
      w, h: 图像宽高
      depth: 视锥深度 (以焦距倍数衡量)
      n_samples_per_edge: 每条边的采样点数

    返回: frustum_points (Nx3), frustum_colors (Nx3)
    """
    # 光心在世界坐标: X_world = R^T * (-t)
    # 因为 X_cam = R*X_world + t, 当 X_cam=0 → X_world = -R^T*t
    R_inv = R.T
    optical_center = (-R_inv @ t).ravel()

    # 像平面四角在归一化坐标 (z=1)
    fx = K[0, 0]
    cx = K[0, 2]
    cy = K[1, 2]

    # 图像四角像素坐标
    corners_px = np.array([
        [0, 0],
        [w - 1, 0],
        [w - 1, h - 1],
        [0, h - 1]
    ], dtype=np.float64)

    # 反投影到 z=depth 平面 (相机坐标系)
    corners_cam = []
    for px, py in corners_px:
        x = (px - cx) / fx * depth
        y = (py - cy) / fx * depth
        z = depth
        corners_cam.append([x, y, z])
    corners_cam = np.array(corners_cam)

    # 变换到世界坐标
    corners_world = (R_inv @ corners_cam.T).T + optical_center

    frustum_vertices = np.vstack([[optical_center], corners_world])

    # 棱线: 0→1, 0→2, 0→3, 0→4 (光心到四角)
    # 矩形边: 1→2, 2→3, 3→4, 4→1 (远平面四边)
    edges = [
        (0, 1), (0, 2), (0, 3), (0, 4),
        (1, 2), (2, 3), (3, 4), (4, 1)
    ]

    # 在每条边上采样
    frustum_points = []
    for v1_idx, v2_idx in edges:
        v1 = frustum_vertices[v1_idx]
        v2 = frustum_vertices[v2_idx]
        for k in range(n_samples_per_edge):
            alpha = k / (n_samples_per_edge - 1)
            pt = v1 + alpha * (v2 - v1)
            frustum_points.append(pt)

    frustum_points = np.array(frustum_points)
    # 视锥用红色标记
    frustum_colors = np.full((len(frustum_points), 3), [255, 0, 0], dtype=np.uint8)

    return frustum_points, frustum_colors


def save_point_cloud_with_frustums(filepath, points, colors, Rs, ts, K,
                                    reg_indices, images, frustum_depth=1.5):
    """
    将点云和所有相机视锥合并保存到单个 PLY 文件

    视锥以红色线条嵌入，可在 MeshLab/CloudCompare 中同时查看点云和相机位姿
    """
    h, w = images[0].shape[:2]

    all_frustum_pts = []
    all_frustum_colors = []

    print(f"  [视锥] 生成 {len(reg_indices)} 个相机视锥...")
    for orig_idx in reg_indices:
        if orig_idx not in Rs:
            continue
        fpts, fcls = generate_frustum_points(K, Rs[orig_idx], ts[orig_idx], w, h,
                                              depth=frustum_depth)
        all_frustum_pts.append(fpts)
        all_frustum_colors.append(fcls)

    if all_frustum_pts:
        frustum_pts = np.vstack(all_frustum_pts)
        frustum_colors = np.vstack(all_frustum_colors)
    else:
        frustum_pts = np.zeros((0, 3))
        frustum_colors = np.zeros((0, 3), dtype=np.uint8)

    # 合并点云和视锥
    combined_pts = np.vstack([points, frustum_pts])
    combined_colors = np.vstack([colors, frustum_colors])

    save_ply(filepath, combined_pts, combined_colors)
    print(f"    点云: {len(points)} 点, 视锥棱线: {len(frustum_pts)} 点")


def save_camera_params(filepath, Rs, ts, K, reg_indices):
    """保存相机参数为 NPZ 文件"""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    cam_data = {}
    for orig_idx in reg_indices:
        if orig_idx in Rs:
            cam_data[f'R_{orig_idx}'] = Rs[orig_idx]
            cam_data[f't_{orig_idx}'] = ts[orig_idx]

    cam_data['K'] = K
    cam_data['reg_indices'] = np.array(reg_indices)

    np.savez_compressed(filepath, **cam_data)
    print(f"  [保存] {filepath.name}: {len(reg_indices)} 台相机参数")


def run_export(output_dir, points3D, point_colors, Rs, ts, K, reg_indices,
               images, dense_pts=None, dense_colors=None):
    """
    导出全部结果:
      1. 稀疏点云 + 相机视锥 → sparse_cloud.ply
      2. 稠密点云 → dense_cloud.ply
      3. 相机参数 → cameras.npz
      4. 点云 + 视锥合并 → cloud_with_frustums.ply
    """
    print("\n" + "=" * 60)
    print("阶段5: 导出结果")
    print("=" * 60)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 稀疏点云 (无相机)
    save_ply(output_dir / 'sparse_cloud.ply', points3D, point_colors)

    # 点云 + 视锥合并 (可在MeshLab中查看)
    save_point_cloud_with_frustums(
        output_dir / 'cloud_with_frustums.ply',
        points3D, point_colors, Rs, ts, K, reg_indices, images
    )

    # 稠密点云
    if dense_pts is not None and len(dense_pts) > 0:
        save_ply(output_dir / 'dense_cloud.ply', dense_pts, dense_colors)

    # 相机参数
    save_camera_params(output_dir / 'cameras.npz', Rs, ts, K, reg_indices)

    print("  [导出] 完成")
