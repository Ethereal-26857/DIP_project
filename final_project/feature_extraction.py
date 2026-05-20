# ======================================================
# 成员 2 —— 颜色特征 + 纹理特征工程
# 功能：颜色矩、颜色直方图、GLCM纹理、LBP纹理提取
#       特征拼接、归一化、PCA降维、特征有效性分析
# ======================================================

import os
import cv2
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

# ====================== 路径配置 ======================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(BASE_DIR, "train")
TEST_DIR = os.path.join(BASE_DIR, "test")
TRAIN_LABEL = os.path.join(BASE_DIR, "train_label.csv")
TEST_LABEL = os.path.join(BASE_DIR, "test_label.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "features")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ====================== 1. 颜色特征提取 ======================

def extract_color_moments(img):
    """
    提取颜色矩特征（RGB三通道）
    每个通道：均值、标准差、偏度 → 9维特征
    """
    moments = []
    for c in range(3):
        channel = img[:, :, c].ravel()
        mean = np.mean(channel)
        std = np.std(channel)
        # 偏度 (skewness)，防止除零
        if std < 1e-6:
            skewness = 0.0
        else:
            skewness = np.mean(((channel - mean) / std) ** 3)
        moments.extend([mean, std, skewness])
    return np.array(moments)


def extract_color_histogram(img, bins=32):
    """
    提取颜色直方图特征（RGB三通道）
    每通道bins个区间的归一化直方图 → 3*bins维特征
    """
    hist_features = []
    for c in range(3):
        channel = img[:, :, c].ravel()
        hist, _ = np.histogram(channel, bins=bins, range=(0, 256), density=True)
        hist_features.extend(hist)
    return np.array(hist_features)


# ====================== 2. 纹理特征提取 ======================

def extract_glcm_features(gray_img, distances=(1, 2), angles=(0, np.pi/4, np.pi/2, 3*np.pi/4)):
    """
    提取GLCM纹理特征
    计算多距离多角度的灰度共生矩阵，取各属性的均值与标准差
    属性：contrast, dissimilarity, homogeneity, energy, correlation, ASM
    → 12维特征 (6个属性 × (均值+标准差))
    """
    glcm = graycomatrix(gray_img, distances=distances, angles=angles,
                        levels=256, symmetric=True, normed=True)

    properties = ['contrast', 'dissimilarity', 'homogeneity',
                  'energy', 'correlation', 'ASM']
    features = []
    for prop in properties:
        vals = graycoprops(glcm, prop).ravel()
        features.append(np.mean(vals))
        features.append(np.std(vals))
    return np.array(features)


def extract_lbp_features(gray_img, P=8, R=1, n_bins=10):
    """
    提取LBP纹理特征
    使用uniform LBP，统计归一化直方图
    → n_bins维特征（默认10）
    """
    # uniform LBP: P*8 neighbors, radius=R
    n_points = P * R
    if n_points < 8:
        n_points = 8
    lbp = local_binary_pattern(gray_img, n_points, R, method='uniform')
    lbp = lbp.astype(np.int32)

    n_patterns = int(lbp.max()) + 1
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_patterns), density=True)
    return hist


# ====================== 3. 综合特征提取 ======================

def extract_all_features(img):
    """
    提取图像的全部颜色+纹理特征
    输入：BGR图像 (H, W, 3)
    输出：特征向量
    """
    # 颜色特征（RGB空间）
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    color_moments = extract_color_moments(img_rgb)       # 9维
    color_hist = extract_color_histogram(img_rgb, bins=32)  # 96维

    # 纹理特征（灰度图）
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    glcm_feat = extract_glcm_features(gray)   # 12维
    lbp_feat = extract_lbp_features(gray, n_bins=10)  # 10维

    return np.concatenate([color_moments, color_hist, glcm_feat, lbp_feat])


def get_feature_names():
    """返回特征名称列表"""
    names = []
    # 颜色矩
    for ch in ['R', 'G', 'B']:
        for moment in ['mean', 'std', 'skewness']:
            names.append(f'color_moment_{ch}_{moment}')
    # 颜色直方图
    for ch in ['R', 'G', 'B']:
        for i in range(32):
            names.append(f'hist_{ch}_{i}')
    # GLCM
    for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']:
        for stat in ['mean', 'std']:
            names.append(f'glcm_{prop}_{stat}')
    # LBP
    for i in range(10):
        names.append(f'lbp_bin_{i}')
    return names


# ====================== 4. 批量特征提取 ======================

def process_dataset(image_dir, label_path):
    """
    对指定目录的所有图像提取特征
    返回：特征矩阵 X, 标签 y, 图像ID列表
    """
    df = pd.read_csv(label_path)
    X_list, y_list, ids = [], [], []

    for _, row in df.iterrows():
        image_id = row['image_id']
        img_path = os.path.join(image_dir, f"{image_id}.jpg")
        img = cv2.imread(img_path)
        if img is None:
            print(f"  ⚠ 跳过损坏图像: {image_id}")
            continue
        feat = extract_all_features(img)
        X_list.append(feat)
        y_list.append(row['dx'])
        ids.append(image_id)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list)
    print(f"  提取完成：{len(ids)} 张图像, 特征维度={X.shape[1]}")
    return X, y, ids


# ====================== 5. 主流程 ======================

def main():
    print("=" * 60)
    print("成员 2：颜色特征 + 纹理特征工程")
    print("=" * 60)

    # ---- 5.1 特征提取 ----
    print("\n[1/5] 提取训练集特征...")
    X_train, y_train, train_ids = process_dataset(TRAIN_DIR, TRAIN_LABEL)

    print("\n[2/5] 提取测试集特征...")
    X_test, y_test, test_ids = process_dataset(TEST_DIR, TEST_LABEL)

    # ---- 5.2 特征归一化 ----
    print("\n[3/5] 特征归一化 (StandardScaler)...")
    scaler = StandardScaler()
    X_train_norm = scaler.fit_transform(X_train)
    X_test_norm = scaler.transform(X_test)
    print(f"  训练集归一化完成, shape={X_train_norm.shape}")
    print(f"  测试集归一化完成, shape={X_test_norm.shape}")

    # ---- 5.3 PCA降维（可选） ----
    print("\n[4/5] PCA降维...")
    # 保留95%的方差
    pca = PCA(n_components=0.95, random_state=42)
    X_train_pca = pca.fit_transform(X_train_norm)
    X_test_pca = pca.transform(X_test_norm)
    print(f"  原始维度: {X_train_norm.shape[1]}, PCA后维度: {X_train_pca.shape[1]}")
    print(f"  累计解释方差: {np.sum(pca.explained_variance_ratio_):.4f}")

    # ---- 5.4 保存特征 ----
    print("\n[5/5] 保存特征矩阵...")
    np.save(os.path.join(OUTPUT_DIR, "X_train_raw.npy"), X_train)
    np.save(os.path.join(OUTPUT_DIR, "X_train_norm.npy"), X_train_norm)
    np.save(os.path.join(OUTPUT_DIR, "X_train_pca.npy"), X_train_pca)
    np.save(os.path.join(OUTPUT_DIR, "X_test_raw.npy"), X_test)
    np.save(os.path.join(OUTPUT_DIR, "X_test_norm.npy"), X_test_norm)
    np.save(os.path.join(OUTPUT_DIR, "X_test_pca.npy"), X_test_pca)
    np.save(os.path.join(OUTPUT_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(OUTPUT_DIR, "y_test.npy"), y_test)
    np.save(os.path.join(OUTPUT_DIR, "train_ids.npy"), np.array(train_ids))
    np.save(os.path.join(OUTPUT_DIR, "test_ids.npy"), np.array(test_ids))
    np.save(os.path.join(OUTPUT_DIR, "feature_names.npy"), np.array(get_feature_names()))

    # 保存PCA和scaler参数，供后续使用
    import pickle
    with open(os.path.join(OUTPUT_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(OUTPUT_DIR, "pca.pkl"), "wb") as f:
        pickle.dump(pca, f)

    print(f"  特征文件已全部保存到 {OUTPUT_DIR}/")
    print(f"  ├─ X_train_raw.npy   ({X_train.shape})")
    print(f"  ├─ X_train_norm.npy  ({X_train_norm.shape})")
    print(f"  ├─ X_train_pca.npy   ({X_train_pca.shape})")
    print(f"  ├─ X_test_raw.npy    ({X_test.shape})")
    print(f"  ├─ X_test_norm.npy   ({X_test_norm.shape})")
    print(f"  ├─ X_test_pca.npy    ({X_test_pca.shape})")
    print(f"  ├─ y_train.npy / y_test.npy")
    print(f"  ├─ scaler.pkl / pca.pkl")
    print(f"  └─ feature_names.npy")

    # ---- 5.5 特征有效性分析 ----
    print("\n" + "=" * 60)
    print("特征有效性分析")
    print("=" * 60)

    # 用随机森林快速评估原始特征
    clf_raw = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    scores_raw = cross_val_score(clf_raw, X_train_norm, y_train, cv=5)
    print(f"\n  [原始特征] 5折CV准确率: {scores_raw.mean():.4f} ± {scores_raw.std():.4f}")

    clf_pca = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    scores_pca = cross_val_score(clf_pca, X_train_pca, y_train, cv=5)
    print(f"  [PCA特征]  5折CV准确率: {scores_pca.mean():.4f} ± {scores_pca.std():.4f}")

    # 互信息分析
    mi_scores = mutual_info_classif(X_train_norm, y_train, random_state=42)
    feature_names = get_feature_names()
    top_indices = np.argsort(mi_scores)[::-1][:15]
    print("\n  [互信息 Top-15 特征]:")
    for rank, idx in enumerate(top_indices, 1):
        print(f"    {rank:2d}. {feature_names[idx]:30s}  MI={mi_scores[idx]:.4f}")

    # 特征类型分组评估
    print("\n  [各类特征维度与贡献]:")
    groups = {
        '颜色矩 (9维)': (0, 9),
        '颜色直方图 (96维)': (9, 105),
        'GLCM纹理 (12维)': (105, 117),
        'LBP纹理 (10维)': (117, 127),
    }
    for name, (start, end) in groups.items():
        group_mi = np.mean(mi_scores[start:end])
        print(f"    {name}: 平均互信息={group_mi:.4f}")

    # 在测试集上的最终评估
    clf_raw.fit(X_train_norm, y_train)
    test_acc = clf_raw.score(X_test_norm, y_test)
    print(f"\n  [测试集准确率] {test_acc:.4f}")

    print("\n" + "=" * 60)
    print("成员 2 特征工程完成！")
    print("=" * 60)
    return X_train_norm, X_test_norm, y_train, y_test


if __name__ == "__main__":
    X_train_norm, X_test_norm, y_train, y_test = main()
