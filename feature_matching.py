"""
模块1: 特征提取与匹配
- 使用 SIFT 提取每张图片的关键点和描述子
- 使用 FLANN + Lowe's ratio test 在相邻图像对之间匹配
"""

import cv2
import numpy as np
from pathlib import Path
import time


def load_images(image_dir, max_dim=1200):
    """
    加载指定目录中的所有图片，并缩小至最长边不超过 max_dim 以加速
    返回: images (list of np.array), image_paths (list of Path), scale_factors (list of float)
    """
    img_dir = Path(image_dir)
    img_paths = sorted(img_dir.glob('*.JPG'))  # Windows glob 大小写不敏感
    if not img_paths:
        img_paths = sorted(img_dir.glob('*.jpg'))
    if not img_paths:
        img_paths = sorted(img_dir.glob('*.png'))
    img_paths = sorted(img_paths)

    images = []
    scale_factors = []

    print(f"  [加载] 找到 {len(img_paths)} 张图片，目标最长边 ≤ {max_dim}px")

    for p in img_paths:
        # 使用 np.fromfile 绕过 OpenCV 的 Unicode 路径问题 (Windows)
        img_data = np.fromfile(str(p), dtype=np.uint8)
        img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
        if img is None:
            print(f"  [警告] 无法读取 {p.name}，跳过")
            continue

        h, w = img.shape[:2]
        scale = max_dim / max(h, w)
        if scale < 1.0:
            new_w, new_h = int(w * scale), int(h * scale)
            img = cv2.resize(img, (new_w, new_h))
            scale_factors.append(scale)
        else:
            scale_factors.append(1.0)

        images.append(img)

    print(f"  [加载] 成功加载 {len(images)} 张图片，尺寸 {images[0].shape[1]}x{images[0].shape[0]}")
    return images, img_paths, scale_factors


def extract_features(images):
    """
    对每张图片提取 SIFT 关键点和描述子
    SIFT (Scale-Invariant Feature Transform) 具有尺度和旋转不变性
    返回: keypoints (list of list of cv2.KeyPoint), descriptors (list of np.array)
    """
    # SIFT 特征检测器: nfeatures=0 表示不限制数量, contrastThreshold 控制低对比度过滤
    sift = cv2.SIFT_create(nfeatures=0, contrastThreshold=0.04, edgeThreshold=10)

    all_keypoints = []
    all_descriptors = []

    print(f"  [SIFT] 开始提取特征...")
    t0 = time.time()

    for i, img in enumerate(images):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, des = sift.detectAndCompute(gray, None)

        all_keypoints.append(kp)
        all_descriptors.append(des)

        if (i + 1) % 20 == 0 or i == 0 or i == len(images) - 1:
            print(f"    图片 {i+1}/{len(images)}: {len(kp)} 个特征点")

    t1 = time.time()
    avg_kp = np.mean([len(k) for k in all_keypoints])
    print(f"  [SIFT] 特征提取完成，耗时 {t1-t0:.1f}s，平均每图 {avg_kp:.0f} 个特征点")
    return all_keypoints, all_descriptors


def match_sequential_pairs(all_keypoints, all_descriptors):
    """
    相邻图像对 (i, i+1) 的特征匹配
    使用 FLANN + Lowe's ratio test:
      - 对每个特征找最近邻和次近邻
      - 如果 distance_ratio = d_nn / d_2nn < threshold, 接受匹配
    返回: matches_list (list of list of cv2.DMatch)
    """
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches_list = []
    n = len(all_descriptors)
    ratio_thresh = 0.7

    print(f"  [匹配] FLANN匹配 (共 {n-1} 对)...")
    t0 = time.time()

    for i in range(n - 1):
        des1, des2 = all_descriptors[i], all_descriptors[i + 1]
        if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
            matches_list.append([])
            continue

        raw_matches = flann.knnMatch(des1, des2, k=2)
        good_matches = []
        for match_pair in raw_matches:
            if len(match_pair) == 2:
                m, n2 = match_pair
                if m.distance < ratio_thresh * n2.distance:
                    good_matches.append(m)

        matches_list.append(good_matches)

    t1 = time.time()
    avg_matches = np.mean([len(m) for m in matches_list])
    print(f"  [匹配] 完成, 耗时 {t1-t0:.1f}s, 平均每对 {avg_matches:.0f} 个匹配")
    return matches_list


def run_feature_matching(image_dir, max_dim=1200):
    """
    运行完整的特征提取与匹配流程
    返回: images, all_keypoints, all_descriptors, matches_list, scale_factors
    """
    print("=" * 60)
    print("阶段1: 特征提取与匹配")
    print("=" * 60)

    images, img_paths, scale_factors = load_images(image_dir, max_dim)
    all_keypoints, all_descriptors = extract_features(images)
    matches_list = match_sequential_pairs(all_keypoints, all_descriptors)

    return images, all_keypoints, all_descriptors, matches_list, scale_factors
