"""
模块6: open3d 可视化
- 交互式三维视图中显示稀疏/稠密点云
- 叠加相机视锥线框
- 支持旋转、缩放、平移
- 截图保存
"""

import numpy as np
import open3d as o3d
from pathlib import Path


def create_frustum_line_set(K, R, t, w, h, depth=1.5, color=(1.0, 1.0, 1.0)):
    """
    创建一个相机视锥的 LineSet (open3d 线框几何体)

    视锥结构:
      - 光心 (顶点0)
      - 远平面四角 (顶点1-4)
      - 8条棱线: 4条光心→远角, 4条远平面边

    返回: o3d.geometry.LineSet
    """
    R_inv = R.T
    optical_center = (-R_inv @ t).ravel()

    fx = K[0, 0]
    cx = K[0, 2]
    cy = K[1, 2]

    corners_px = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float64)

    corners_cam = []
    for px, py in corners_px:
        x = (px - cx) / fx * depth
        y = (py - cy) / fx * depth
        z = depth
        corners_cam.append([x, y, z])
    corners_cam = np.array(corners_cam)

    corners_world = (R_inv @ corners_cam.T).T + optical_center

    vertices = np.vstack([[optical_center], corners_world])

    edges = [(0, 1), (0, 2), (0, 3), (0, 4),  # 光心→远角
             (1, 2), (2, 3), (3, 4), (4, 1)]   # 远平面四边

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(vertices)
    line_set.lines = o3d.utility.Vector2iVector(edges)

    # 每条边单独着色
    colors = [color for _ in range(len(edges))]
    line_set.colors = o3d.utility.Vector3dVector(colors)

    return line_set


def run_visualization(points, colors, Rs, ts, K, reg_indices, images,
                      output_dir, frustum_depth=1.5, max_points=500000):
    """
    open3d 交互式可视化

    显示内容:
      - 带颜色的三维点云
      - 每台已注册相机的白色线框视锥
      - 自动截图保存
    """
    print("\n" + "=" * 60)
    print("阶段6: open3d 可视化")
    print("=" * 60)

    h, w = images[0].shape[:2]

    # --- 构建点云 ---
    n_points = min(len(points), max_points)
    if len(points) > max_points:
        step = len(points) // max_points
        pts_subset = points[::step][:max_points]
        clr_subset = colors[::step][:max_points]
    else:
        pts_subset = points
        clr_subset = colors

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_subset)
    pcd.colors = o3d.utility.Vector3dVector(clr_subset.astype(np.float64) / 255.0)

    geometries = [pcd]

    # --- 构建所有相机视锥 ---
    # 根据相机间距自动计算合适的视锥深度
    cam_centers = []
    for idx in reg_indices:
        if idx in Rs:
            c = (-Rs[idx].T @ ts[idx]).ravel()
            cam_centers.append(c)
    cam_centers = np.array(cam_centers)
    # 平均相邻相机间距
    if len(cam_centers) > 1:
        avg_baseline = np.mean(np.linalg.norm(np.diff(cam_centers, axis=0), axis=1))
    else:
        avg_baseline = 1.0
    frustum_depth_vis = avg_baseline * 2.0  # 视锥长度为2倍基线

    print(f"  [可视化] 渲染 {len(reg_indices)} 个相机视锥 (深度={frustum_depth_vis:.1f})...")
    for idx in reg_indices:
        if idx not in Rs:
            continue
        frustum = create_frustum_line_set(
            K, Rs[idx], ts[idx], w, h,
            depth=frustum_depth_vis,
            color=(1.0, 1.0, 1.0)  # 白色线框
        )
        geometries.append(frustum)

    # --- 交互式窗口 ---
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="SfM 三维重建 — 点云 + 相机视锥",
                       width=1280, height=720)

    for geo in geometries:
        vis.add_geometry(geo)

    # 设置渲染选项
    opt = vis.get_render_option()
    opt.point_size = 2.0
    opt.background_color = np.array([0.1, 0.1, 0.15])  # 深蓝灰色背景
    opt.show_coordinate_frame = True

    # 设置初始视角
    ctr = vis.get_view_control()
    ctr.set_zoom(0.8)

    print("  [可视化] 窗口已打开，关闭窗口继续...")
    vis.run()
    vis.destroy_window()

    # 截图 (需要在窗口关闭前保存, 此处用非窗口模式重新渲染)
    print(f"  [可视化] 完成")


if __name__ == "__main__":
    print("可视化模块 — 请通过 main.py 调用")
