# 生物医学图像处理 — 期末大作业 项目交付文档

## 项目概述

**课题**：Project 2 — 皮肤镜图像三分类（黑色素瘤 mel / 痣 nv / 血管病变 vasc）

**数据**：200 张原始皮肤镜图像 + 增强图像（旋转/翻转），共 600 张，80/20 分层划分为训练集（480 张）和测试集（120 张）

**任务分工**：

| 成员 | 模块 | 状态 |
|------|------|------|
| 成员1 | 数据预处理（去噪、去毛发、尺寸归一化、亮度归一化、数据集划分） | ✅ 完成 |
| 成员2 | 颜色特征 + 纹理特征工程（颜色矩、直方图、GLCM、LBP、PCA降维） | ✅ 完成 |
| 成员3 | 传统机器学习模型训练与评估（SVM / RF / KNN、网格搜索、指标评估） | ✅ 完成 |

---

## 目录结构

```
final_project/
├── train/                   # 训练集图像 (480 张 JPG)
├── test/                    # 测试集图像 (120 张 JPG)
├── train_label.csv          # 训练集标签 (image_id, dx)
├── test_label.csv           # 测试集标签 (image_id, dx)
│
├── features/                # 成员2输出 → 成员3输入
│   ├── X_train_norm.npy     #   训练集归一化特征 (480×127)
│   ├── X_test_norm.npy      #   测试集归一化特征 (120×127)
│   ├── X_train_pca.npy      #   PCA降维后特征 (480×36)
│   ├── X_test_pca.npy       #   PCA降维后特征 (120×36)
│   ├── y_train.npy          #   训练集标签
│   ├── y_test.npy           #   测试集标签
│   ├── scaler.pkl           #   StandardScaler 模型
│   ├── pca.pkl              #   PCA 模型
│   └── feature_names.npy    #   特征名称列表
│
├── results/                 # 成员3输出
│   ├── best_model_SVM.pkl   #   最佳模型 (SVM, F1_macro=0.736)
│   ├── model_comparison.csv #   三模型性能汇总表
│   ├── evaluation_results.pkl # 完整评估数据
│   ├── classification_report_*.txt  # 各模型详细分类报告
│   └── *.png                #   可视化图表
│
├── feature_extraction.py    # 成员2：特征提取主脚本
├── feature_analysis.py      # 成员2：特征分析 + 可视化
├── model_training.py        # 成员3：模型训练 + 评估主脚本
├── demo.ipynb               # 成员1/2：演示 Notebook
├── member2_feature_engineering.ipynb  # 成员2：特征工程 Notebook
│
├── report_member2.tex/pdf   # 成员2：实验报告
├── report_member3.tex/pdf   # 成员3：实验报告
│
├── .vene/                   # Python 虚拟环境（所有依赖已安装）
├── .vscode/settings.json    # VS Code 配置（自动选择 .vene 解释器）
└── Project/                 # 原始课题说明 & 示例代码
```

---

## 环境配置

### 方式一：使用已有虚拟环境（推荐）

```bash
# 激活虚拟环境
.vene/Scripts/activate      # Windows Git Bash
# 或
.vene\Scripts\activate      # Windows CMD / PowerShell

# 验证依赖
pip list | grep -E "numpy|pandas|scikit|matplotlib|seaborn"
```

### 方式二：重新创建环境

```bash
python -m venv venv
source venv/Scripts/activate
pip install numpy pandas scikit-learn scikit-image matplotlib seaborn opencv-python jupyter
```

### VS Code 设置

打开项目后，`Ctrl+Shift+P` → `Python: Select Interpreter` → 选 `.vene\Scripts\python.exe`。

---

## 如何运行

### 重新生成特征（成员2）

```bash
python feature_extraction.py
# 输出到 features/ 目录
```

### 重新训练模型（成员3）

```bash
python model_training.py
# 输出到 results/ 目录，耗时约 1-2 分钟
```

### 加载已训练模型进行推理

```python
import pickle
import numpy as np

# 加载模型
with open('results/best_model_SVM.pkl', 'rb') as f:
    model = pickle.load(f)

# 加载特征
X_test = np.load('features/X_test_norm.npy')
y_test = np.load('features/y_test.npy', allow_pickle=True)

# 预测
y_pred = model.predict(X_test)
```

---

## 核心结果（成员3）

| 模型 | 准确率 | F1(Macro) | 最佳参数 |
|------|--------|-----------|----------|
| Random Forest | **73.33%** | 0.736 | n=100, max_depth=None |
| SVM (RBF) | 72.50% | 0.736 | C=100, gamma=scale |
| KNN | 66.67% | 0.672 | k=1, metric=manhattan |

**关键结论**：
- SVM 和 RF 性能相当，推荐 SVM（泛化更好、对 vasc 少数类覆盖更优）
- mel ↔ nv 混淆是主要分类难点（两者视觉相似）
- vasc（血管病变）识别效果最好，F1≈0.79

---

## 后续改进方向

1. **深度学习特征**：用预训练 CNN（ResNet/EfficientNet）提取特征替代手工特征
2. **特征融合**：手工特征 + CNN 特征拼接
3. **更多分类器**：尝试 XGBoost / LightGBM / 多层感知机
4. **数据增强**：更多增强策略（颜色抖动、随机擦除）
5. **类别不平衡**：SMOTE 过采样或类别加权
6. **集成学习**：Stacking / Voting 融合多个模型

---

## 提交流程

- 提交文件：`model_training.py` + `report_member3.pdf`
- 代码运行说明见上方「如何运行」部分
- 如有问题联系成员3
