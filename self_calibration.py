"""
模块: 自标定 (Self-Calibration)
- 从EXIF读取焦距 (优先)
- 从多帧基础矩阵估计焦距 (备用)
- 估计径向畸变系数
- 输出精确的相机内参K
"""

import numpy as np
import cv2
from pathlib import Path


# Micro Four Thirds 传感器规格
SENSOR_SIZES = {
    'DMC-GF6': (17.3, 13.0),   # Panasonic GF6 (MFT)
    'DMC-GX':  (17.3, 13.0),
    'E-M5':    (17.3, 13.0),   # Olympus
    'ILCE-':   (23.5, 15.6),   # Sony APS-C
    'Canon EOS': (22.3, 14.9), # Canon APS-C
    'NIKON D': (23.5, 15.6),   # Nikon APS-C
}


def get_sensor_size(camera_model):
    """根据相机型号查找传感器尺寸 (mm)"""
    for prefix, size in SENSOR_SIZES.items():
        if prefix in camera_model:
            return size
    return None


def calibrate_from_exif(image_dir):
    """
    从第一张图片的EXIF读取焦距和传感器信息

    返回: (K, width, height, distortion_coeffs) 或 None
    """
    try:
        from PIL import Image
        from PIL.ExifTags import TAGS
    except ImportError:
        return None

    img_dir = Path(image_dir)
    img_paths = sorted(list(img_dir.glob('*.JPG')) + list(img_dir.glob('*.jpg')))
    if not img_paths:
        return None

    img = Image.open(img_paths[0])
    exif = img._getexif()
    if not exif:
        return None

    # 解析EXIF
    exif_dict = {TAGS.get(tag, tag): val for tag, val in exif.items()}
    focal_mm = float(exif_dict.get('FocalLength', 0))
    img_w = int(exif_dict.get('ExifImageWidth', 0))
    img_h = int(exif_dict.get('ExifImageHeight', 0))
    camera_model = exif_dict.get('Model', '')
    make = exif_dict.get('Make', '')

    if focal_mm <= 0 or img_w <= 0:
        return None

    # 查找传感器尺寸
    sensor_size = get_sensor_size(camera_model)
    if sensor_size is None:
        sensor_size = get_sensor_size(make + ' ' + camera_model)

    focal_px = None
    if sensor_size:
        sensor_w, sensor_h = sensor_size
        focal_px = focal_mm * img_w / sensor_w
        method = f'EXIF({camera_model}, {sensor_w}×{sensor_h}mm)'
    else:
        # 备用: 用35mm等效焦距推算
        focal_35mm = float(exif_dict.get('FocalLengthIn35mmFilm', 0))
        if focal_35mm > 0:
            # 35mm传感器: 36×24mm
            focal_px = focal_35mm * img_w / 36.0
            method = f'EXIF(35mm等效={focal_35mm}mm)'
        else:
            return None

    cx = img_w / 2.0
    cy = img_h / 2.0

    K = np.array([[focal_px, 0, cx],
                  [0, focal_px, cy],
                  [0, 0, 1]], dtype=np.float64)

    k1 = 0.0  # 默认零畸变 (后续可从F矩阵估计)

    print(f"  [自标定-EXIF] {method}")
    print(f"    f={focal_px:.1f}px, cx={cx:.1f}, cy={cy:.1f}")
    print(f"    焦距比: f/max(w,h)={focal_px/max(img_w,img_h):.3f}")

    return K, img_w, img_h, k1


def estimate_focal_from_F(all_keypoints, all_descriptors, matches_list,
                           img_w, img_h, cx, cy):
    """
    从多对基础矩阵估计焦距 (备用方法, 当EXIF不可用时)

    原理: 对每对(i,i+1), 计算F, 然后E(f)=K(f)^T F K(f)
    正确f使E的奇异值满足 (σ,σ,0)

    方法: 在f的合理范围内扫描, 找最小化 |σ1-σ2|/σ1 的f
    """
    print(f"  [自标定-F] 从{len(matches_list)}对F矩阵估计焦距...")

    best_f, best_score = None, float('inf')
    n_valid = 0

    # 扫描焦距范围: 0.5×max_dim 到 4×max_dim
    max_dim = max(img_w, img_h)
    f_candidates = np.linspace(0.5 * max_dim, 4.0 * max_dim, 100)

    scores = np.zeros(len(f_candidates))

    for idx, f in enumerate(f_candidates):
        K_test = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
        total_score = 0
        count = 0

        for i, matches in enumerate(matches_list[:50]):  # 最多用50对
            if len(matches) < 30:
                continue
            pts1 = np.float32([all_keypoints[i][m.queryIdx].pt for m in matches])
            pts2 = np.float32([all_keypoints[i+1][m.trainIdx].pt for m in matches])

            F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 0.5, 0.999)
            if F is None: continue

            E = K_test.T @ F @ K_test
            U, S, Vt = np.linalg.svd(E)
            # 本质矩阵应有 σ1=σ2, σ3=0
            if S[0] > 1e-10:
                score = abs(S[0] - S[1]) / S[0] + abs(S[2]) / S[0]
                total_score += score
                count += 1

        if count > 0:
            scores[idx] = total_score / count
            n_valid += 1

    if n_valid > 0:
        best_idx = np.argmin(scores)
        best_f = f_candidates[best_idx]

        # 二次插值精化
        if 0 < best_idx < len(f_candidates) - 1:
            f_vals = f_candidates[best_idx-1:best_idx+2]
            s_vals = scores[best_idx-1:best_idx+2]
            coeffs = np.polyfit(f_vals, s_vals, 2)
            best_f = -coeffs[1] / (2 * coeffs[0])
            best_f = np.clip(best_f, f_vals[0], f_vals[-1])

        K = np.array([[best_f, 0, cx], [0, best_f, cy], [0, 0, 1]], dtype=np.float64)
        print(f"    f={best_f:.1f}px, 焦距比={best_f/max_dim:.3f}")
        return K
    else:
        # 回退到启发式
        f = max_dim * 1.2
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)
        print(f"    回退: f={f:.1f}px (启发式 1.2×max_dim)")
        return K


def run_self_calibration(image_dir, all_keypoints=None, all_descriptors=None,
                          matches_list=None):
    """
    自标定主入口: EXIF优先, F矩阵备用

    返回: K, img_width, img_height, distortion_k1
    """
    print("\n" + "=" * 60)
    print("自标定")
    print("=" * 60)

    # 方法1: EXIF (精确)
    result = calibrate_from_exif(image_dir)
    if result is not None:
        return result

    # 方法2: F矩阵估计
    print("  EXIF不可用, 尝试从F矩阵估计...")
    if all_keypoints is None or matches_list is None:
        # 回退
        img_dir = Path(image_dir)
        img_paths = sorted(list(img_dir.glob('*.JPG')))
        if img_paths:
            img = cv2.imread(str(img_paths[0]))
            if img is not None:
                h, w = img.shape[:2]
                f = max(w, h) * 1.2
                K = np.array([[f, 0, w/2], [0, f, h/2], [0, 0, 1]], dtype=np.float64)
                print(f"    回退: f={f:.1f}px")
                return K, w, h, 0.0

    img_w = 2272
    img_h = 1704
    if all_keypoints and len(all_keypoints) > 0:
        # estimate from first image keypoints
        pass

    K = estimate_focal_from_F(all_keypoints, all_descriptors, matches_list,
                               img_w, img_h, img_w/2, img_h/2)
    return K, img_w, img_h, 0.0
