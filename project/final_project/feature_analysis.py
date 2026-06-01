# ======================================================
# 成员 2 —— 特征有效性分析与可视化
# ======================================================

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 非交互模式，直接保存图片
import matplotlib.pyplot as plt
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import mutual_info_classif
import pickle
import os

# ====================== 加载特征 ======================
FEAT_DIR = "./features"
X_train = np.load(os.path.join(FEAT_DIR, "X_train_norm.npy"))
X_test = np.load(os.path.join(FEAT_DIR, "X_test_norm.npy"))
X_train_pca = np.load(os.path.join(FEAT_DIR, "X_train_pca.npy"))
X_test_pca = np.load(os.path.join(FEAT_DIR, "X_test_pca.npy"))
y_train = np.load(os.path.join(FEAT_DIR, "y_train.npy"), allow_pickle=True)
y_test = np.load(os.path.join(FEAT_DIR, "y_test.npy"), allow_pickle=True)
feature_names = np.load(os.path.join(FEAT_DIR, "feature_names.npy"), allow_pickle=True)

with open(os.path.join(FEAT_DIR, "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)
with open(os.path.join(FEAT_DIR, "pca.pkl"), "rb") as f:
    pca = pickle.load(f)

classes = ['mel', 'nv', 'vasc']
print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")
print(f"PCA后维度: {X_train_pca.shape[1]}")
print(f"类别分布 (train): {np.unique(y_train, return_counts=True)}")
print(f"类别分布 (test):  {np.unique(y_test, return_counts=True)}")

# ====================== 1. PCA方差解释 ======================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 累计方差
cumsum = np.cumsum(pca.explained_variance_ratio_)
axes[0].plot(range(1, len(cumsum) + 1), cumsum, 'b-', linewidth=2)
axes[0].axhline(y=0.95, color='r', linestyle='--', label='95% threshold')
axes[0].axvline(x=X_train_pca.shape[1], color='g', linestyle='--',
                label=f'n_components={X_train_pca.shape[1]}')
axes[0].set_xlabel('Number of Components')
axes[0].set_ylabel('Cumulative Explained Variance')
axes[0].set_title('PCA Cumulative Explained Variance')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 各成分方差
axes[1].bar(range(1, 21), pca.explained_variance_ratio_[:20], color='steelblue', edgecolor='white')
axes[1].set_xlabel('Principal Component')
axes[1].set_ylabel('Explained Variance Ratio')
axes[1].set_title('Top 20 PCA Components')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(FEAT_DIR, "pca_analysis.png"), dpi=150, bbox_inches='tight')
plt.close()

# ====================== 2. 互信息分析 ======================
mi_scores = mutual_info_classif(X_train, y_train, random_state=42)
top30_idx = np.argsort(mi_scores)[::-1][:30]

fig, ax = plt.subplots(figsize=(12, 7))
colors = []
for name in feature_names[top30_idx]:
    if 'color_moment' in name:
        colors.append('#e74c3c')
    elif 'hist_' in name:
        colors.append('#3498db')
    elif 'glcm' in name:
        colors.append('#2ecc71')
    else:
        colors.append('#f39c12')

bars = ax.barh(range(30), mi_scores[top30_idx][::-1], color=colors[::-1], edgecolor='white')
ax.set_yticks(range(30))
ax.set_yticklabels(feature_names[top30_idx][::-1], fontsize=8)
ax.set_xlabel('Mutual Information Score')
ax.set_title('Top 30 Features by Mutual Information')
ax.grid(True, alpha=0.3, axis='x')

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#e74c3c', label='Color Moments'),
    Patch(facecolor='#3498db', label='Color Histogram'),
    Patch(facecolor='#2ecc71', label='GLCM'),
    Patch(facecolor='#f39c12', label='LBP'),
]
ax.legend(handles=legend_elements, loc='lower right')
plt.tight_layout()
plt.savefig(os.path.join(FEAT_DIR, "mutual_info.png"), dpi=150, bbox_inches='tight')
plt.close()

# ====================== 3. t-SNE可视化 ======================
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_all = np.vstack([X_train, X_test])
y_all = np.hstack([y_train, y_test])
X_tsne = tsne.fit_transform(X_all)

fig, ax = plt.subplots(figsize=(10, 8))
colors_class = {'mel': '#e74c3c', 'nv': '#3498db', 'vasc': '#2ecc71'}
markers = {'train': 'o', 'test': 's'}

# 训练集
for cls in classes:
    mask = (y_train == cls)
    ax.scatter(X_tsne[:len(y_train)][mask, 0], X_tsne[:len(y_train)][mask, 1],
               c=colors_class[cls], marker='o', label=f'{cls} (train)',
               alpha=0.6, s=30, edgecolors='none')

# 测试集
for cls in classes:
    mask = (y_test == cls)
    ax.scatter(X_tsne[len(y_train):][mask, 0], X_tsne[len(y_train):][mask, 1],
               c=colors_class[cls], marker='s', label=f'{cls} (test)',
               alpha=0.8, s=40, edgecolors='black', linewidth=0.5)

ax.set_xlabel('t-SNE Component 1')
ax.set_ylabel('t-SNE Component 2')
ax.set_title('t-SNE Visualization of Extracted Features')
ax.legend(loc='upper right', fontsize=7, ncol=2)
plt.tight_layout()
plt.savefig(os.path.join(FEAT_DIR, "tsne_visualization.png"), dpi=150, bbox_inches='tight')
plt.close()

# ====================== 4. 多分类器对比 ======================
classifiers = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM (RBF)': SVC(kernel='rbf', random_state=42),
    'KNN (k=5)': KNeighborsClassifier(n_neighbors=5),
}

print("\n" + "=" * 60)
print("多分类器对比评估")
print("=" * 60)

results = []
for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    train_acc = clf.score(X_train, y_train)
    test_acc = clf.score(X_test, y_test)
    results.append({'Classifier': name, 'Train Acc': train_acc, 'Test Acc': test_acc})
    print(f"\n  [{name}]")
    print(f"    训练准确率: {train_acc:.4f}")
    print(f"    测试准确率: {test_acc:.4f}")

# ====================== 5. 最佳分类器详细评估 ======================
best_clf = RandomForestClassifier(n_estimators=100, random_state=42)
best_clf.fit(X_train, y_train)
y_pred = best_clf.predict(X_test)

print("\n" + "=" * 60)
print("Random Forest 详细评估")
print("=" * 60)
print(classification_report(y_test, y_pred, target_names=classes))

# 混淆矩阵
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes, ax=ax)
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
ax.set_title('Confusion Matrix - Random Forest')
plt.tight_layout()
plt.savefig(os.path.join(FEAT_DIR, "confusion_matrix.png"), dpi=150, bbox_inches='tight')
plt.close()

# ====================== 6. 特征组消融实验 ======================
print("\n" + "=" * 60)
print("特征组消融实验")
print("=" * 60)

feature_groups = {
    'color_moment': [i for i, n in enumerate(feature_names) if 'color_moment' in n],
    'hist': [i for i, n in enumerate(feature_names) if 'hist_' in n],
    'glcm': [i for i, n in enumerate(feature_names) if 'glcm' in n],
    'lbp': [i for i, n in enumerate(feature_names) if 'lbp' in n],
}

# 各组单独使用
for name, indices in feature_groups.items():
    X_train_sub = X_train[:, indices]
    X_test_sub = X_test[:, indices]
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train_sub, y_train)
    acc = clf.score(X_test_sub, y_test)
    print(f"  仅用 {name} ({len(indices)}维): 测试准确率 = {acc:.4f}")

# 全部特征
print(f"  全部特征 (127维): 测试准确率 = {best_clf.score(X_test, y_test):.4f}")
print(f"  PCA降维 ({X_train_pca.shape[1]}维): 测试准确率 = ", end="")
clf_pca = RandomForestClassifier(n_estimators=100, random_state=42)
clf_pca.fit(X_train_pca, y_train)
print(f"{clf_pca.score(X_test_pca, y_test):.4f}")

print("\n特征有效性分析完成！")
