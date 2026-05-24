"""快速查看 PLY 点云 (open3d)"""
import sys
import numpy as np
import open3d as o3d

def view_ply(filepath, point_size=2.0, bg_color=(0.1, 0.1, 0.15)):
    pcd = o3d.io.read_point_cloud(filepath)
    if not pcd.has_points():
        print(f"无法读取: {filepath}")
        return

    n = len(pcd.points)
    print(f"加载: {n} 个点")

    # 下采样 (如果太多)
    if n > 2000000:
        pcd = pcd.uniform_down_sample(max(1, n // 1000000))
        print(f"下采样到 {len(pcd.points)} 点")

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"PLY Viewer - {filepath}", width=1400, height=900)
    vis.add_geometry(pcd)

    opt = vis.get_render_option()
    opt.point_size = point_size
    opt.background_color = np.array(bg_color)
    opt.show_coordinate_frame = True

    ctr = vis.get_view_control()
    ctr.set_zoom(0.8)

    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python view_ply.py <file.ply>")
        print("示例: python view_ply.py output/sparse_cloud.ply")
        sys.exit(1)

    view_ply(sys.argv[1])
