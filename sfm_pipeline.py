"""
模块2: 增量式结构从运动 (Incremental Structure from Motion)
- 固定初始对(0,1), 同一相机、恒定内参K
- FLANN匹配 + Lowe's ratio test
- 增量式注册(PnP) + 位姿发散检测
"""

import cv2
import numpy as np
import time


def estimate_intrinsics(img_width, img_height, focal_ratio=1.2):
    """自标定: 估算内参K"""
    cx, cy = img_width / 2.0, img_height / 2.0
    f = max(img_width, img_height) * focal_ratio
    return np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)


def get_matched_points(kp1, kp2, matches):
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    return pts1, pts2


def decompose_essential_matrix(E):
    U, _, Vt = np.linalg.svd(E)
    if np.linalg.det(U @ Vt) < 0: Vt = -Vt
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    return U @ W @ Vt, U @ W.T @ Vt, U[:, 2]


def select_best_pose(R1, R2, t_dir, pts1, pts2, K):
    t_pos = t_dir.reshape(3, 1)
    candidates = [(R1, t_pos), (R1, -t_pos), (R2, t_pos), (R2, -t_pos)]
    P0 = (K @ np.hstack((np.eye(3), np.zeros((3, 1))))).astype(np.float64)
    pts1_64 = pts1.T.astype(np.float64).copy()
    pts2_64 = pts2.T.astype(np.float64).copy()
    best_count, best_idx = -1, -1
    for idx, (R, t) in enumerate(candidates):
        P1 = (K @ np.hstack((R, t))).astype(np.float64)
        pts4d = cv2.triangulatePoints(P0, P1, pts1_64, pts2_64)
        pts3d = pts4d[:3] / pts4d[3]
        d1, d2 = pts3d[2], (R @ pts3d + t)[2]
        count = np.sum((d1 > 0) & (d2 > 0))
        if count > best_count: best_count, best_idx = count, idx
    best_R, best_t = candidates[best_idx]
    P1_best = (K @ np.hstack((best_R, best_t))).astype(np.float64)
    pts4d = cv2.triangulatePoints(P0, P1_best, pts1_64, pts2_64)
    pts3d_all = pts4d[:3] / pts4d[3]
    mask = (pts3d_all[2] > 0) & ((best_R @ pts3d_all + best_t)[2] > 0)
    return best_R, best_t, pts1[mask], pts2[mask]


def initialize_from_pair(kp0, kp1, matches_01, K):
    pts1, pts2 = get_matched_points(kp0, kp1, matches_01)
    F, mask_F = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 0.5, 0.999)
    if F is None: return None
    inlier = mask_F.ravel() == 1
    pts1_in, pts2_in = pts1[inlier], pts2[inlier]
    E_raw = K.T @ F @ K
    U, S, Vt = np.linalg.svd(E_raw)
    E = U @ np.diag([(S[0]+S[1])/2, (S[0]+S[1])/2, 0]) @ Vt
    R1d, R2d, td = decompose_essential_matrix(E)
    R, t, pts1_f, pts2_f = select_best_pose(R1d, R2d, td, pts1_in, pts2_in, K)
    P1 = (K @ np.hstack((np.eye(3), np.zeros((3, 1))))).astype(np.float64)
    P2 = (K @ np.hstack((R, t))).astype(np.float64)
    pts4d = cv2.triangulatePoints(P1, P2, pts1_f.T.astype(np.float64).copy(),
                                   pts2_f.T.astype(np.float64).copy())
    pts3d = (pts4d[:3] / pts4d[3]).T.copy()
    valid_indices = np.where(inlier)[0]
    # select_best_pose可能进一步筛选, 重新对齐
    pt_set = set((round(p[0], 3), round(p[1], 3)) for p in pts1_f)
    valid_indices = [valid_indices[j] for j in range(len(pts1_in))
                     if (round(pts1_in[j][0], 3), round(pts1_in[j][1], 3)) in pt_set]
    print(f"  [初始化] {len(pts3d)}个三维点")
    return {'R0': np.eye(3), 't0': np.zeros((3, 1)), 'R1': R, 't1': t,
            'points3D': pts3d, 'pts1': pts1_f, 'pts2': pts2_f,
            'valid_indices': np.array(valid_indices)}


def triangulate_points(P1, P2, pts1, pts2):
    pts4d = cv2.triangulatePoints(P1.astype(np.float64), P2.astype(np.float64),
                                   pts1.T.astype(np.float64).copy(),
                                   pts2.T.astype(np.float64).copy())
    return (pts4d[:3] / pts4d[3]).T.copy()


def check_positive_depth(R, t, pts3d):
    X = R @ pts3d.T + t.reshape(3, 1)
    return X[2] > 0


def run_sfm(images, all_keypoints, all_descriptors, matches_list,
            focal_ratio=1.2, min_matches=30, K_input=None):
    print("\n" + "=" * 60)
    print("阶段2: 增量式 SfM")
    print("=" * 60)
    n_images = len(images)
    h, w = images[0].shape[:2]

    # 使用外部K (来自EXIF自标定) 或启发式估计
    if K_input is not None:
        K = K_input.copy()
        print(f"  [内参] 使用EXIF标定: f={K[0,0]:.1f}px")
    else:
        K = estimate_intrinsics(w, h, focal_ratio)

    # 选择最佳初始对: 尝试不同间隔, 选E内点最多且基线适中的
    best_init = None
    best_score = -1
    init_gaps = [1, 2, 3, 4, 5]  # 候选间隔
    for gap in init_gaps:
        if gap >= n_images: continue
        # 交叉检查匹配初始对
        des_a, des_b = all_descriptors[0], all_descriptors[gap]
        if des_a is None or des_b is None: continue
        bf = cv2.BFMatcher(cv2.NORM_L2)
        raw_ab = bf.knnMatch(des_a, des_b, k=2)
        raw_ba = bf.knnMatch(des_b, des_a, k=2)
        good_ab = {}
        for mp in raw_ab:
            if len(mp)==2 and mp[0].distance < 0.7 * mp[1].distance:
                good_ab[mp[0].queryIdx] = mp[0].trainIdx
        good_ba = {}
        for mp in raw_ba:
            if len(mp)==2 and mp[0].distance < 0.7 * mp[1].distance:
                good_ba[mp[0].queryIdx] = mp[0].trainIdx
        cross = []
        for q,t in good_ab.items():
            if t in good_ba and good_ba[t]==q:
                cross.append(cv2.DMatch(q, t, 0))
        if len(cross) < 30: continue
        # 检查E矩阵内点
        pts1, pts2 = get_matched_points(all_keypoints[0], all_keypoints[gap], cross)
        E_try, mask_E = cv2.findEssentialMat(pts1, pts2, K, cv2.RANSAC, 0.999, 0.5)
        if E_try is None: continue
        n_E = mask_E.sum()
        # 检查基线
        n_pts, R_try, t_try, _ = cv2.recoverPose(E_try, pts1[mask_E.ravel()==1], pts2[mask_E.ravel()==1], K)
        baseline = np.linalg.norm(t_try)
        if n_pts < 20 or baseline < 0.3: continue
        # 得分: 内点×基线
        score = n_E * baseline
        if score > best_score:
            best_score = score
            best_init = (0, gap, cross)
        print(f"    候选(0,{gap}): E内点={n_E} 基线={baseline:.2f} 得分={score:.0f}")

    if best_init is None:
        # 回退到(0,1)
        best_init = (0, 1, matches_list[0])
    init_i, init_j, init_matches = best_init
    print(f"\n  --- 初始化 ({init_i}, {init_j}) 间隔={init_j-init_i}帧 ---")
    init = initialize_from_pair(all_keypoints[init_i], all_keypoints[init_j],
                                 init_matches, K)
    if init is None:
        return {}, {}, K, np.zeros((0,3)), np.zeros((0,3),dtype=np.uint8), [], []

    Rs = {init_i: init['R0'], init_j: init['R1']}
    ts = {init_i: init['t0'], init_j: init['t1']}
    reg_indices = [init_i, init_j]
    points3D = init['points3D']

    point_maps = [{}, {}]
    # init_matches: 在cross列表中, queryIdx→kp_init_i, trainIdx→kp_init_j
    for local_idx, match_idx in enumerate(init['valid_indices']):
        m = init_matches[match_idx]
        point_maps[0][m.queryIdx] = local_idx
        point_maps[1][m.trainIdx] = local_idx

    point_colors = []
    img_init_rgb = cv2.cvtColor(images[init_i], cv2.COLOR_BGR2RGB)
    for pt in init['pts1']:
        px = max(0, min(w-1, int(round(pt[0]))))
        py = max(0, min(h-1, int(round(pt[1]))))
        point_colors.append(img_init_rgb[py, px])
    point_colors = np.array(point_colors, dtype=np.uint8)

    print(f"  [初始化] 2台相机, {len(points3D)}个三维点")

    # 增量注册
    FLANN_INDEX_KDTREE = 1
    flann = cv2.FlannBasedMatcher(dict(algorithm=FLANN_INDEX_KDTREE, trees=5),
                                   dict(checks=50))

    for i in range(2, n_images):
        kp_i, des_i = all_keypoints[i], all_descriptors[i]
        if des_i is None or len(des_i) < 2: continue

        # 收集2D-3D对应
        pts2D_list, pts3D_list = [], []

        # 方法1: 预计算匹配 (相邻帧)
        for reg_order, orig_idx in enumerate(reg_indices):
            pmap = point_maps[reg_order]
            if orig_idx + 1 == i and orig_idx < len(matches_list):
                for m in matches_list[orig_idx]:
                    pt3d_idx = pmap.get(m.queryIdx, -1)
                    if pt3d_idx >= 0:
                        pts2D_list.append(kp_i[m.trainIdx].pt)
                        pts3D_list.append(points3D[pt3d_idx])

        # 方法2: FLANN补充
        if len(pts2D_list) < 30:
            for reg_order, orig_idx in enumerate(reg_indices):
                pmap = point_maps[reg_order]
                des_reg = all_descriptors[orig_idx]
                if des_reg is None or len(des_reg) < 2: continue
                raw = flann.knnMatch(des_i, des_reg, k=2)
                for mp in raw:
                    if len(mp) == 2 and mp[0].distance < 0.7 * mp[1].distance:
                        pt3d_idx = pmap.get(mp[0].trainIdx, -1)
                        if pt3d_idx >= 0:
                            pts2D_list.append(kp_i[mp[0].queryIdx].pt)
                            pts3D_list.append(points3D[pt3d_idx])

        if len(pts2D_list) < 8:
            print(f"  [图{i}] 2D-3D对应不足({len(pts2D_list)}), 跳过"); continue

        pts2D_arr = np.float32(pts2D_list)
        pts3D_arr = np.float32(pts3D_list)

        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            pts3D_arr, pts2D_arr, K, None,
            flags=cv2.SOLVEPNP_EPNP, iterationsCount=200,
            reprojectionError=6.0, confidence=0.999)

        if not success or inliers is None or len(inliers) < 10:
            print(f"  [图{i}] PnP失败(内点={len(inliers) if inliers is not None else 0}), 跳过"); continue

        R, _ = cv2.Rodrigues(rvec)
        t_new = tvec.reshape(3, 1)

        # 位姿发散检测 + 大步长检测
        cam_center = (-R.T @ t_new).ravel()
        if len(reg_indices) >= 3:
            prev_c = (-Rs[reg_indices[-1]].T @ ts[reg_indices[-1]]).ravel()
            prev2_c = (-Rs[reg_indices[-2]].T @ ts[reg_indices[-2]]).ravel()
            expected = prev_c + (prev_c - prev2_c)
            step_ref = np.linalg.norm(prev_c - prev2_c) + 1e-6
            jump = np.linalg.norm(cam_center - expected)
            step_current = np.linalg.norm(cam_center - prev_c)

            if jump > step_ref * 10:
                print(f"  [图{i}] 位姿发散(跳变={jump:.0f}), 跳过"); continue

            # 检测异常大步长: 超过2倍历史中位步长
            if len(reg_indices) >= 4:
                all_steps_hist = [np.linalg.norm(
                    (-Rs[reg_indices[k]].T @ ts[reg_indices[k]]).ravel() -
                    (-Rs[reg_indices[k-1]].T @ ts[reg_indices[k-1]]).ravel())
                    for k in range(1, len(reg_indices))]
                med_step = np.median(all_steps_hist) + 1e-6
                if step_current > med_step * 2.0:
                    print(f"  [图{i}] 大步长({step_current:.1f}>>{med_step:.1f}), 重试更多参考帧...")
                    # 用更多参考相机重新PnP
                    pts2D_retry, pts3D_retry = [], []
                    # 匹配所有前5帧已注册相机
                    for reg_order in range(max(0, len(reg_indices)-8), len(reg_indices)):
                        orig_k = reg_indices[reg_order]
                        if orig_k + 1 == i and orig_k < len(matches_list):
                            for m in matches_list[orig_k]:
                                pt3d_idx = point_maps[reg_order].get(m.queryIdx, -1)
                                if pt3d_idx >= 0:
                                    pts2D_retry.append(kp_i[m.trainIdx].pt)
                                    pts3D_retry.append(points3D[pt3d_idx])

                    if len(pts2D_retry) > len(pts2D_list):
                        # 用更多对应重新PnP
                        pts2D_arr_r = np.float32(pts2D_retry)
                        pts3D_arr_r = np.float32(pts3D_retry)
                        success_r, rvec_r, tvec_r, inliers_r = cv2.solvePnPRansac(
                            pts3D_arr_r, pts2D_arr_r, K, None,
                            flags=cv2.SOLVEPNP_EPNP, iterationsCount=200,
                            reprojectionError=6.0, confidence=0.999)
                        if success_r and inliers_r is not None and len(inliers_r) > 10:
                            R_r, _ = cv2.Rodrigues(rvec_r)
                            t_r = tvec_r.reshape(3, 1)
                            cam_center_r = (-R_r.T @ t_r).ravel()
                            step_r = np.linalg.norm(cam_center_r - prev_c)
                            if step_r < step_current * 0.8:  # 步长明显改善
                                R, t_new = R_r, t_r
                                cam_center = cam_center_r
                                step_current = step_r
                                print(f"    重注册改善: 步长{step_current:.1f}")

        # 三角化新点
        prev_idx = reg_indices[-1]
        triangulated = False
        start_idx = len(points3D)

        if prev_idx + 1 == i and prev_idx < len(matches_list):
            pair_matches = matches_list[prev_idx]
            pmap_prev = point_maps[reg_indices.index(prev_idx)]
            kp_prev = all_keypoints[prev_idx]

            unmapped = []
            pts_a_l, pts_b_l = [], []
            for m in pair_matches:
                if m.queryIdx not in pmap_prev:
                    unmapped.append(m)
                    pts_a_l.append(kp_prev[m.queryIdx].pt)
                    pts_b_l.append(kp_i[m.trainIdx].pt)

            if len(unmapped) > 10:
                pts_a = np.float32(pts_a_l)
                pts_b = np.float32(pts_b_l)
                P_a = K @ np.hstack((Rs[prev_idx], ts[prev_idx]))
                P_b = K @ np.hstack((R, t_new))
                new_pts = triangulate_points(P_a, P_b, pts_a, pts_b)
                mask_f = (check_positive_depth(Rs[prev_idx], ts[prev_idx], new_pts) &
                         check_positive_depth(R, t_new, new_pts))

                if np.sum(mask_f) > 5:
                    new_pts = new_pts[mask_f]
                    pts_b_f = pts_b[mask_f]
                    img_i_rgb = cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB)
                    new_clrs = np.array([img_i_rgb[max(0,min(h-1,int(round(pt[1])))),
                                                    max(0,min(w-1,int(round(pt[0]))))]
                                         for pt in pts_b_f], dtype=np.uint8)
                    points3D = np.vstack([points3D, new_pts])
                    point_colors = np.vstack([point_colors, new_clrs])
                    cnt = 0
                    for jj, m in enumerate(unmapped):
                        if mask_f[jj]:
                            pmap_prev[m.queryIdx] = start_idx + cnt
                            cnt += 1
                    triangulated = True
                    print(f"     三角化+{cnt}点")

        Rs[i] = R; ts[i] = t_new
        reg_indices.append(i)
        point_maps.append({})

        if triangulated:
            new_map_idx = len(point_maps) - 1
            pmap_prev_idx = reg_indices.index(prev_idx)
            cnt = 0
            for jj, m in enumerate(unmapped):
                if mask_f[jj]:
                    point_maps[new_map_idx][m.trainIdx] = start_idx + cnt
                    cnt += 1

        print(f"  [图{i}] OK | PnP内点={len(inliers)}/{len(pts2D_list)} | "
              f"总3D点={len(points3D)} | 已注册{len(reg_indices)}")

    print(f"\n  [SfM完成] {len(reg_indices)}/{n_images}台相机, {len(points3D)}个三维点")

    # 收集观测
    observations = []
    for reg_order, orig_idx in enumerate(reg_indices):
        pmap = point_maps[reg_order]
        kp_list = all_keypoints[orig_idx]
        for kp_idx, pt3d_idx in pmap.items():
            if pt3d_idx < len(points3D):
                pt = kp_list[kp_idx].pt
                observations.append((orig_idx, kp_idx, pt3d_idx, pt[0], pt[1]))

    return Rs, ts, K, points3D, point_colors, reg_indices, observations
