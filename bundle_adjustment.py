"""
模块3: 光束法平差 (Bundle Adjustment)
- 迭代式 motion-only BA: 每台相机PnP精化 → 三角化 → 重复
- 使用cv2.solvePnP内置LM优化, 比scipy更稳定
- 固定相机0为世界原点, 保持尺度一致
"""

import cv2
import numpy as np
import time


def rodrigues_vec(R):
    rvec, _ = cv2.Rodrigues(R)
    return rvec.ravel()


def rodrigues_mat(rvec):
    R, _ = cv2.Rodrigues(rvec.reshape(3, 1))
    return R


def triangulate_points_vec(P1, P2, pts1, pts2):
    pts4d = cv2.triangulatePoints(
        P1.astype(np.float64), P2.astype(np.float64),
        pts1.T.astype(np.float64).copy(), pts2.T.astype(np.float64).copy())
    return (pts4d[:3] / pts4d[3]).T.copy()


def run_bundle_adjustment(Rs, ts, K, points3D, point_colors, reg_indices,
                          all_keypoints, all_descriptors, observations,
                          max_iterations=3):
    """
    迭代式BA: 每台相机PnP精化 → 重三角化 → 循环

    1. 固定相机0 (世界原点)
    2. 对每台相机: 收集其2D-3D观测, solvePnP LM精化位姿
    3. 重三角化所有三维点 (用精化后的相机)
    4. 重复2-3

    效率高 (每相机仅几十个参数), 收敛稳定
    """
    print("\n" + "=" * 60)
    print("阶段3: 光束法平差 (迭代PnP)")
    print("=" * 60)

    if not observations or len(reg_indices) < 3:
        print("  [BA] 数据不足, 跳过")
        return Rs, ts, K, points3D

    n_cameras = len(reg_indices)
    n_points = len(points3D)
    h, w = 0, 0  # 图像尺寸 (用于后续三角化)

    # 按相机组织观测: {orig_idx: [(kp_idx, pt3d_idx, u_obs, v_obs)]}
    cam_observations = {idx: [] for idx in reg_indices}
    point_observations = {i: [] for i in range(n_points)}  # {pt_idx: [(cam_idx, u, v)]}
    for orig_idx, kp_idx, pt3d_idx, u_obs, v_obs in observations:
        if pt3d_idx < n_points and orig_idx in cam_observations:
            cam_observations[orig_idx].append((kp_idx, pt3d_idx, u_obs, v_obs))
            point_observations[pt3d_idx].append((orig_idx, u_obs, v_obs))

    # 统计初始误差
    init_errors = []
    for orig_idx in reg_indices:
        if orig_idx not in Rs: continue
        R, t = Rs[orig_idx], ts[orig_idx]
        for _, pt3d_idx, u_obs, v_obs in cam_observations[orig_idx]:
            if pt3d_idx >= n_points: continue
            X = points3D[pt3d_idx]
            x = R @ X + t.ravel()
            if x[2] <= 0: continue
            u = K @ x
            u_p, v_p = u[0]/u[2], u[1]/u[2]
            init_errors.append(np.sqrt((u_p-u_obs)**2 + (v_p-v_obs)**2))
    init_errors = np.array(init_errors)
    print(f"  [BA] {n_cameras}台相机, {n_points}个点")
    print(f"       初始误差: 均值={init_errors.mean():.2f}px "
          f"中位数={np.median(init_errors):.2f}px")

    # 迭代优化
    for iteration in range(max_iterations):
        t0 = time.time()

        # Step 1: 精化每台相机位姿 (相机0固定)
        improved_count = 0
        for orig_idx in reg_indices:
            if orig_idx == 0:
                continue  # 相机0固定
            if not cam_observations[orig_idx]:
                continue

            # 收集该相机的2D-3D对应
            pts3d_list, pts2d_list = [], []
            for _, pt3d_idx, u_obs, v_obs in cam_observations[orig_idx]:
                if pt3d_idx < n_points:
                    pts3d_list.append(points3D[pt3d_idx])
                    pts2d_list.append([u_obs, v_obs])

            if len(pts3d_list) < 6:
                continue

            pts3d_arr = np.float64(pts3d_list)
            pts2d_arr = np.float64(pts2d_list)

            # 用当前位姿作为初值, LM迭代精化
            rvec0 = rodrigues_vec(Rs[orig_idx])
            tvec0 = ts[orig_idx].ravel().astype(np.float64)

            # 先用RANSAC粗筛
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                pts3d_arr, pts2d_arr, K, None,
                rvec=rvec0, tvec=tvec0,
                useExtrinsicGuess=True,
                iterationsCount=100,
                reprojectionError=4.0,
                flags=cv2.SOLVEPNP_ITERATIVE)

            if success and inliers is not None and len(inliers) > 8:
                # 在内点上做精化
                inlier_pts3d = pts3d_arr[inliers.ravel()]
                inlier_pts2d = pts2d_arr[inliers.ravel()]
                rvec_refined, tvec_refined = cv2.solvePnPRefineLM(
                    inlier_pts3d, inlier_pts2d, K, None,
                    rvec, tvec,
                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 20, 1e-7))
                Rs[orig_idx] = rodrigues_mat(rvec_refined)
                ts[orig_idx] = tvec_refined.reshape(3, 1)
                improved_count += 1

        # Step 2: 多视图重三角化 - 用所有观测相机, 选基线最大的两帧
        if iteration < max_iterations - 1:
            n_retriangulated = 0
            for pt_idx in range(n_points):
                obs_list = point_observations[pt_idx]
                if len(obs_list) < 2:
                    continue
                # 选基线最大的两个相机 (而非前两个)
                best_baseline = -1
                best_pair = None
                for a in range(len(obs_list)):
                    for b in range(a+1, len(obs_list)):
                        ca, _, _ = obs_list[a]
                        cb, _, _ = obs_list[b]
                        if ca not in Rs or cb not in Rs: continue
                        pos_a = (-Rs[ca].T @ ts[ca]).ravel()
                        pos_b = (-Rs[cb].T @ ts[cb]).ravel()
                        bl = np.linalg.norm(pos_a - pos_b)
                        # 同时检查三角化角度 (避免退化的窄角)
                        X_est = points3D[pt_idx]
                        v1 = X_est - pos_a; v2 = X_est - pos_b
                        cos_angle = abs(np.dot(v1,v2))/(np.linalg.norm(v1)*np.linalg.norm(v2)+1e-10)
                        angle = np.arccos(np.clip(cos_angle,-1,1))
                        if bl > best_baseline and angle > 0.05:  # >3度
                            best_baseline = bl
                            best_pair = (obs_list[a], obs_list[b])
                if best_pair is None: continue
                (cam1, u1, v1), (cam2, u2, v2) = best_pair
                P1 = K @ np.hstack((Rs[cam1], ts[cam1]))
                P2 = K @ np.hstack((Rs[cam2], ts[cam2]))
                new_pt = triangulate_points_vec(P1, P2, np.float64([[u1,v1]]), np.float64([[u2,v2]]))
                if new_pt.shape[0] > 0:
                    X1 = Rs[cam1] @ new_pt[0] + ts[cam1].ravel()
                    X2 = Rs[cam2] @ new_pt[0] + ts[cam2].ravel()
                    if X1[2] > 0 and X2[2] > 0:
                        points3D[pt_idx] = new_pt[0]
                        n_retriangulated += 1
            if n_retriangulated > 0:
                print(f"    重三角化 {n_retriangulated}/{n_points} 个点")

        t1 = time.time()
        # 计算当前误差
        curr_errors = []
        for orig_idx in reg_indices:
            if orig_idx not in Rs: continue
            R, t = Rs[orig_idx], ts[orig_idx]
            for _, pt3d_idx, u_obs, v_obs in cam_observations[orig_idx]:
                if pt3d_idx >= n_points: continue
                X = points3D[pt3d_idx]
                x = R @ X + t.ravel()
                if x[2] <= 0: continue
                u = K @ x
                u_p, v_p = u[0]/u[2], u[1]/u[2]
                curr_errors.append(np.sqrt((u_p-u_obs)**2 + (v_p-v_obs)**2))
        curr_errors = np.array(curr_errors)

        print(f"  迭代{iteration+1}: {improved_count}台相机精化, "
              f"误差: 均值={curr_errors.mean():.2f}px "
              f"中位数={np.median(curr_errors):.2f}px "
              f"({t1-t0:.1f}s)")

    print(f"  [BA完成] 最终误差: 均值={curr_errors.mean():.2f}px "
          f"中位数={np.median(curr_errors):.2f}px")

    return Rs, ts, K, points3D
