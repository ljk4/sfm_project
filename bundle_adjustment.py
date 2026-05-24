"""
模块3: 光束法平差 (Bundle Adjustment) — 带参考系约束
- 固定第一台相机为世界原点 (R=I, t=0)
- 仅优化相机参数 (motion-only BA), 三维点固定
- 最小化实际特征观测的重投影误差
- 使用稀疏雅可比加速
"""

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
import cv2
import time


def rot_to_rvec(R):
    rvec, _ = cv2.Rodrigues(R)
    return rvec.ravel()


def rvec_to_rot(rvec):
    R, _ = cv2.Rodrigues(rvec.reshape(3, 1))
    return R


def project(X, rvec, tvec, K):
    """投影三维点到像素坐标"""
    R = rvec_to_rot(rvec)
    x = R @ X + tvec
    u = K @ x
    return u[:2] / u[2]


def compute_ba_residuals(params, n_cam_opt, cam_to_opt_idx, observations, points3D):
    """
    计算重投影残差

    params: [fx, cx, cy] + n_cam_opt * [rvec(3), tvec(3)]
    注意: 相机0不包含在params中 (固定为R=I, t=0)
    """
    fx, cx, cy = params[0], params[1], params[2]
    K = np.array([[fx, 0, cx], [0, fx, cy], [0, 0, 1]])

    residuals = np.zeros(len(observations) * 2)

    # 相机0固定值
    R0 = np.eye(3)
    t0 = np.zeros(3)

    for i, (orig_idx, kp_idx, pt3d_idx, u_obs, v_obs) in enumerate(observations):
        if pt3d_idx >= len(points3D):
            continue
        X = points3D[pt3d_idx]

        if orig_idx == 0:
            # 相机0: 固定 R=I, t=0
            u_pred, v_pred = project(X, np.zeros(3), np.zeros(3), K)
        else:
            opt_idx = cam_to_opt_idx.get(orig_idx, -1)
            if opt_idx < 0:
                continue
            offset = 3 + opt_idx * 6
            rvec = params[offset:offset + 3]
            tvec = params[offset + 3:offset + 6]
            u_pred, v_pred = project(X, rvec, tvec, K)

        residuals[2 * i] = u_pred - u_obs
        residuals[2 * i + 1] = v_pred - v_obs

    return residuals


def run_bundle_adjustment(Rs, ts, K, points3D, point_colors, reg_indices,
                          observations, max_obs=30000):
    """
    运动结构BA: 优化相机参数 (相机0固定为原点)

    约束:
      - 相机0: R=I, t=0 (不优化, 固定世界参考系)
      - 全局内参K (fx, cx, cy) 参与优化
      - 其余相机: R, t 全优化
      - 三维点固定 (motion-only approach)
    """
    print("\n" + "=" * 60)
    print("阶段3: 光束法平差 (固定参考系)")
    print("=" * 60)

    if not observations or len(reg_indices) < 2:
        print("  [BA] 数据不足, 跳过")
        return Rs, ts, K, points3D

    n_cam_total = len(reg_indices)
    n_obs_total = len(observations)

    # 降采样观测
    if n_obs_total > max_obs:
        idx = np.random.choice(n_obs_total, max_obs, replace=False)
        obs_subset = [observations[i] for i in idx]
    else:
        obs_subset = observations

    # 相机0固定在原点, 其余相机参与优化
    cam_opt_list = [idx for idx in reg_indices if idx != 0]
    cam_to_opt_idx = {orig_idx: opt_i for opt_i, orig_idx in enumerate(cam_opt_list)}
    n_cam_opt = len(cam_opt_list)

    # 参数向量: [fx, cx, cy] + n_cam_opt * [rvec(3), tvec(3)]
    n_params = 3 + n_cam_opt * 6
    params0 = np.zeros(n_params)
    params0[0] = K[0, 0]
    params0[1] = K[0, 2]
    params0[2] = K[1, 2]

    for opt_idx, orig_idx in enumerate(cam_opt_list):
        offset = 3 + opt_idx * 6
        params0[offset:offset + 3] = rot_to_rvec(Rs[orig_idx])
        params0[offset + 3:offset + 6] = ts[orig_idx].ravel()

    # 稀疏雅可比
    n_res = len(obs_subset) * 2
    jac_sp = lil_matrix((n_res, n_params))
    for obs_i, (orig_idx, _, _, _, _) in enumerate(obs_subset):
        row = 2 * obs_i
        jac_sp[row:row + 2, 0:3] = 1  # 全局K影响所有残差
        if orig_idx != 0:
            opt_i = cam_to_opt_idx.get(orig_idx)
            if opt_i is not None:
                off = 3 + opt_i * 6
                jac_sp[row:row + 2, off:off + 6] = 1

    print(f"  [BA] {n_cam_total}台相机({n_cam_opt}优化+1固定), {len(obs_subset)}个观测")
    t0 = time.time()

    result = least_squares(
        compute_ba_residuals,
        params0,
        jac_sparsity=jac_sp,
        verbose=0,
        x_scale='jac',
        ftol=1e-8,
        xtol=1e-8,
        gtol=1e-8,
        max_nfev=200,
        method='trf',
        args=(n_cam_opt, cam_to_opt_idx, obs_subset, points3D)
    )

    t1 = time.time()
    params_opt = result.x

    # 提取优化后参数
    K_opt = np.array([[params_opt[0], 0, params_opt[1]],
                       [0, params_opt[0], params_opt[2]],
                       [0, 0, 1]])

    Rs_opt = {0: np.eye(3)}
    ts_opt = {0: np.zeros((3, 1))}
    for opt_idx, orig_idx in enumerate(cam_opt_list):
        offset = 3 + opt_idx * 6
        Rs_opt[orig_idx] = rvec_to_rot(params_opt[offset:offset + 3])
        ts_opt[orig_idx] = params_opt[offset + 3:offset + 6].reshape(3, 1)

    # 重投影误差统计
    errors = []
    for obs in obs_subset:
        orig_idx, _, pt3d_idx, u_obs, v_obs = obs
        if pt3d_idx >= len(points3D):
            continue
        X = points3D[pt3d_idx]
        if orig_idx == 0:
            rv, tv = np.zeros(3), np.zeros(3)
        elif orig_idx in Rs_opt:
            rv = rot_to_rvec(Rs_opt[orig_idx])
            tv = ts_opt[orig_idx].ravel()
        else:
            continue
        u_p, v_p = project(X, rv, tv, K_opt)
        errors.append(np.sqrt((u_p - u_obs)**2 + (v_p - v_obs)**2))

    errors = np.array(errors)
    print(f"  [BA] 完成({t1-t0:.1f}s): 损失={result.cost:.2e}")
    print(f"       误差: 均值={errors.mean():.3f}px 中位数={np.median(errors):.3f}px "
          f"90分位={np.percentile(errors, 90):.2f}px")
    print(f"       K变化: fx {K[0,0]:.0f}→{K_opt[0,0]:.0f} "
          f"cx {K[0,2]:.0f}→{K_opt[0,2]:.0f} cy {K[1,2]:.0f}→{K_opt[1,2]:.0f}")

    return Rs_opt, ts_opt, K_opt, points3D
