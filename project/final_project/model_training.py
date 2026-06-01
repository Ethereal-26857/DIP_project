# ======================================================
# 成员 3 —— 传统机器学习模型训练与评估
# 功能：SVM / 随机森林 / KNN 模型搭建
#       交叉验证、参数调优、测试集评估
#       输出准确率、精确率、召回率、F1、混淆矩阵
# ======================================================

import os
import numpy as np
import pandas as pd
import pickle
import time
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False
import seaborn as sns

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import (
    StratifiedKFold, GridSearchCV, cross_val_score,
    cross_validate, learning_curve
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_curve, auc,
    RocCurveDisplay
)
from sklearn.preprocessing import label_binarize
from itertools import cycle

# ====================== 路径配置 ======================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FEAT_DIR = os.path.join(BASE_DIR, "features")
RESULT_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULT_DIR, exist_ok=True)

CLASSES = ['mel', 'nv', 'vasc']
CLASS_LABELS = {'mel': 0, 'nv': 1, 'vasc': 2}


# ====================== 1. 加载数据 ======================

def load_data(use_pca=False):
    """加载预处理后的特征数据"""
    if use_pca:
        X_train = np.load(os.path.join(FEAT_DIR, "X_train_pca.npy"))
        X_test = np.load(os.path.join(FEAT_DIR, "X_test_pca.npy"))
    else:
        X_train = np.load(os.path.join(FEAT_DIR, "X_train_norm.npy"))
        X_test = np.load(os.path.join(FEAT_DIR, "X_test_norm.npy"))

    y_train = np.load(os.path.join(FEAT_DIR, "y_train.npy"), allow_pickle=True)
    y_test = np.load(os.path.join(FEAT_DIR, "y_test.npy"), allow_pickle=True)

    return X_train, X_test, y_train, y_test


# ====================== 2. 模型定义与参数网格 ======================

def build_models():
    """定义三种分类器及其超参数搜索空间"""
    models = {
        'SVM': {
            'estimator': SVC(probability=True, random_state=42),
            'param_grid': {
                'C': [0.1, 1, 10, 100],
                'kernel': ['linear', 'rbf'],
                'gamma': ['scale', 'auto', 0.01, 0.1],
            }
        },
        'Random Forest': {
            'estimator': RandomForestClassifier(random_state=42),
            'param_grid': {
                'n_estimators': [50, 100, 200, 300],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
            }
        },
        'KNN': {
            'estimator': KNeighborsClassifier(),
            'param_grid': {
                'n_neighbors': [1, 3, 5, 7, 9, 11, 15],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan', 'minkowski'],
            }
        },
    }
    return models


# ====================== 3. 网格搜索与交叉验证 ======================

def tune_model(clf, param_grid, X_train, y_train, model_name, cv=5):
    """使用GridSearchCV进行参数调优"""
    print(f"\n{'='*60}")
    print(f"  [{model_name}] 网格搜索 + {cv}折交叉验证")
    print(f"{'='*60}")

    start_time = time.time()

    grid = GridSearchCV(
        clf, param_grid, cv=StratifiedKFold(cv, shuffle=True, random_state=42),
        scoring='accuracy', n_jobs=-1, verbose=1
    )
    grid.fit(X_train, y_train)

    elapsed = time.time() - start_time
    print(f"  搜索完成, 耗时: {elapsed:.1f}s")
    print(f"  最佳参数: {grid.best_params_}")
    print(f"  最佳CV准确率: {grid.best_score_:.4f}")

    # 输出所有搜索结果（按rank排序）
    cv_results = pd.DataFrame(grid.cv_results_)
    top_results = cv_results.nsmallest(5, 'rank_test_score')[
        ['rank_test_score', 'mean_test_score', 'std_test_score', 'params']
    ]
    print(f"\n  Top-5 参数组合:")
    for _, row in top_results.iterrows():
        print(f"    Rank {int(row['rank_test_score'])}: "
              f"CV={row['mean_test_score']:.4f} ± {row['std_test_score']:.4f}")

    return grid.best_estimator_, grid.best_params_, grid.best_score_


# ====================== 4. 交叉验证详细评估 ======================

def cross_validate_detailed(model, X_train, y_train, model_name, cv=5):
    """执行交叉验证，输出多种指标"""
    print(f"\n  [{model_name}] {cv}折交叉验证详细指标:")

    scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
    cv_scores = cross_validate(
        model, X_train, y_train, cv=StratifiedKFold(cv, shuffle=True, random_state=42),
        scoring=scoring, n_jobs=-1
    )

    results = {}
    for metric in scoring:
        scores = cv_scores[f'test_{metric}']
        results[metric] = {'mean': scores.mean(), 'std': scores.std()}
        print(f"    {metric}: {scores.mean():.4f} ± {scores.std():.4f}")

    return results


# ====================== 5. 测试集完整评估 ======================

def evaluate_on_test(model, X_test, y_test, model_name):
    """在测试集上输出完整的评估指标"""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None

    acc = accuracy_score(y_test, y_pred)
    # 各类别详细指标: average=None 返回各类别值
    prec = precision_score(y_test, y_pred, average=None, labels=CLASSES)
    rec = recall_score(y_test, y_pred, average=None, labels=CLASSES)
    f1 = f1_score(y_test, y_pred, average=None, labels=CLASSES)
    # 宏平均 & 加权平均
    prec_macro = precision_score(y_test, y_pred, average='macro')
    rec_macro = recall_score(y_test, y_pred, average='macro')
    f1_macro = f1_score(y_test, y_pred, average='macro')
    prec_weighted = precision_score(y_test, y_pred, average='weighted')
    rec_weighted = recall_score(y_test, y_pred, average='weighted')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')

    cm = confusion_matrix(y_test, y_pred)

    print(f"\n{'='*60}")
    print(f"  [{model_name}] 测试集评估")
    print(f"{'='*60}")
    print(f"  准确率 (Accuracy):  {acc:.4f}")
    print(f"\n  各类别指标:")
    print(f"  {'类别':>8s}  {'精确率':>8s}  {'召回率':>8s}  {'F1-score':>8s}")
    for i, cls in enumerate(CLASSES):
        print(f"  {cls:>8s}  {prec[i]:>8.4f}  {rec[i]:>8.4f}  {f1[i]:>8.4f}")
    print(f"\n  宏平均 (Macro Avg):")
    print(f"    Precision: {prec_macro:.4f}  Recall: {rec_macro:.4f}  F1: {f1_macro:.4f}")
    print(f"  加权平均 (Weighted Avg):")
    print(f"    Precision: {prec_weighted:.4f}  Recall: {rec_weighted:.4f}  F1: {f1_weighted:.4f}")

    print(f"\n  混淆矩阵:")
    print(f"              Predicted")
    print(f"              {'mel':>6s}  {'nv':>6s}  {'vasc':>6s}")
    for i, cls in enumerate(CLASSES):
        print(f"  True {cls:>4s}  {cm[i, 0]:>6d}  {cm[i, 1]:>6d}  {cm[i, 2]:>6d}")

    # 汇总为字典
    metrics = {
        'model': model_name,
        'accuracy': acc,
        'precision_per_class': dict(zip(CLASSES, prec)),
        'recall_per_class': dict(zip(CLASSES, rec)),
        'f1_per_class': dict(zip(CLASSES, f1)),
        'precision_macro': prec_macro,
        'recall_macro': rec_macro,
        'f1_macro': f1_macro,
        'precision_weighted': prec_weighted,
        'recall_weighted': rec_weighted,
        'f1_weighted': f1_weighted,
        'confusion_matrix': cm,
        'y_pred': y_pred,
        'y_proba': y_proba,
    }
    return metrics


# ====================== 6. 可视化 ======================

def plot_confusion_matrices(all_metrics, save_path):
    """绘制所有模型的混淆矩阵"""
    n_models = len(all_metrics)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4))

    for ax, (name, m) in zip(axes, all_metrics.items()):
        sns.heatmap(m['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                    xticklabels=CLASSES, yticklabels=CLASSES, ax=ax,
                    cbar=False, annot_kws={'fontsize': 12})
        ax.set_title(f'{name}', fontsize=13)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n混淆矩阵已保存至: {save_path}")


def plot_metrics_comparison(all_metrics, save_path):
    """绘制所有模型的指标对比柱状图"""
    models = list(all_metrics.keys())
    metrics_names = ['Accuracy', 'Precision\n(Macro)', 'Recall\n(Macro)', 'F1\n(Macro)']
    x = np.arange(len(metrics_names))
    width = 0.25
    colors = ['#e74c3c', '#3498db', '#2ecc71']

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (name, m) in enumerate(all_metrics.items()):
        values = [m['accuracy'], m['precision_macro'], m['recall_macro'], m['f1_macro']]
        bars = ax.bar(x + i * width, values, width, label=name, color=colors[i],
                      edgecolor='white', linewidth=0.8)
        # 数值标注
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    ax.set_xticks(x + width)
    ax.set_xticklabels(metrics_names)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison', fontsize=14)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"模型对比图已保存至: {save_path}")


def plot_per_class_f1(all_metrics, save_path):
    """绘制各类别 F1-score 对比图"""
    models = list(all_metrics.keys())
    x = np.arange(len(CLASSES))
    width = 0.25
    colors = ['#e74c3c', '#3498db', '#2ecc71']

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (name, m) in enumerate(all_metrics.items()):
        values = [m['f1_per_class'][c] for c in CLASSES]
        bars = ax.bar(x + i * width, values, width, label=name, color=colors[i],
                      edgecolor='white', linewidth=0.8)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    ax.set_xticks(x + width)
    ax.set_xticklabels(CLASSES)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel('F1-Score')
    ax.set_title('Per-Class F1-Score Comparison', fontsize=14)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"各类别F1对比图已保存至: {save_path}")


def plot_roc_curves(all_metrics, X_test, y_test, save_path):
    """绘制每个模型的ROC曲线（OvR多分类）"""
    y_test_bin = label_binarize(y_test, classes=CLASSES)
    n_classes = len(CLASSES)
    n_models = len(all_metrics)

    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4))
    if n_models == 1:
        axes = [axes]

    colors = cycle(['#e74c3c', '#3498db', '#2ecc71'])

    for ax, (name, m) in zip(axes, all_metrics.items()):
        if m['y_proba'] is None:
            ax.text(0.5, 0.5, 'No probability available', ha='center', va='center')
            ax.set_title(name)
            continue

        for i, (cls, color) in enumerate(zip(CLASSES, colors)):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], m['y_proba'][:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=color, lw=2,
                    label=f'{cls} (AUC={roc_auc:.2f})')

        ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'{name} ROC', fontsize=13)
        ax.legend(loc='lower right', fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"ROC曲线已保存至: {save_path}")


def plot_learning_curves(best_model, X_train, y_train, model_name, save_path):
    """绘制最佳模型的学习曲线"""
    fig, ax = plt.subplots(figsize=(8, 5))

    train_sizes, train_scores, test_scores = learning_curve(
        best_model, X_train, y_train,
        cv=StratifiedKFold(5, shuffle=True, random_state=42),
        n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='accuracy'
    )

    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    test_mean = test_scores.mean(axis=1)
    test_std = test_scores.std(axis=1)

    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                    alpha=0.2, color='#3498db')
    ax.fill_between(train_sizes, test_mean - test_std, test_mean + test_std,
                    alpha=0.2, color='#e74c3c')
    ax.plot(train_sizes, train_mean, 'o-', color='#3498db', linewidth=2, label='Training Score')
    ax.plot(train_sizes, test_mean, 'o-', color='#e74c3c', linewidth=2, label='Validation Score')

    ax.set_xlabel('Training Samples')
    ax.set_ylabel('Accuracy')
    ax.set_title(f'{model_name} - Learning Curve', fontsize=13)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"学习曲线已保存至: {save_path}")


# ====================== 7. 结果汇总表 ======================

def generate_summary_table(all_metrics, save_path):
    """生成LaTeX格式的汇总表格"""
    rows = []
    for name, m in all_metrics.items():
        rows.append({
            '模型': name,
            '准确率': f"{m['accuracy']:.4f}",
            '精确率(Macro)': f"{m['precision_macro']:.4f}",
            '召回率(Macro)': f"{m['recall_macro']:.4f}",
            'F1(Macro)': f"{m['f1_macro']:.4f}",
            'F1(Weighted)': f"{m['f1_weighted']:.4f}",
        })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(save_path, "model_comparison.csv"), index=False)

    # 打印表格
    print(f"\n{'='*80}")
    print("模型性能汇总")
    print(f"{'='*80}")
    print(df.to_string(index=False))

    return df


# ====================== 8. 主流程 ======================

def main():
    print("=" * 60)
    print("成员 3：传统机器学习模型训练与评估")
    print("=" * 60)

    # 加载数据
    print("\n[1/6] 加载特征数据...")
    X_train, X_test, y_train, y_test = load_data(use_pca=False)
    print(f"  训练集: {X_train.shape}, 测试集: {X_test.shape}")
    for cls in CLASSES:
        cnt_train = np.sum(y_train == cls)
        cnt_test = np.sum(y_test == cls)
        print(f"  {cls}: train={cnt_train}, test={cnt_test}")

    # 构建模型
    models_cfg = build_models()

    # 存储所有结果
    all_metrics = {}
    best_models = {}

    # 依次训练每个模型
    for model_name, cfg in models_cfg.items():
        print(f"\n\n{'#'*60}")
        print(f"#  {model_name}")
        print(f"{'#'*60}")

        # Step A: 参数调优
        best_clf, best_params, best_cv_score = tune_model(
            cfg['estimator'], cfg['param_grid'],
            X_train, y_train, model_name, cv=5
        )
        best_models[model_name] = best_clf

        # Step B: 交叉验证详细指标
        cv_results = cross_validate_detailed(best_clf, X_train, y_train, model_name)

        # Step C: 测试集评估
        metrics = evaluate_on_test(best_clf, X_test, y_test, model_name)
        all_metrics[model_name] = metrics

        # 打印分类报告
        print(f"\n  sklearn分类报告:")
        print(classification_report(y_test, metrics['y_pred'],
                                    target_names=CLASSES, digits=4))

    # ---- 评估结束，开始输出结果 ----

    print(f"\n\n{'='*60}")
    print("[2/6] 生成可视化图表...")
    print("=" * 60)

    plot_confusion_matrices(all_metrics,
                            os.path.join(RESULT_DIR, "confusion_matrices.png"))
    plot_metrics_comparison(all_metrics,
                            os.path.join(RESULT_DIR, "metrics_comparison.png"))
    plot_per_class_f1(all_metrics,
                      os.path.join(RESULT_DIR, "per_class_f1.png"))
    plot_roc_curves(all_metrics, X_test, y_test,
                    os.path.join(RESULT_DIR, "roc_curves.png"))

    # 找出最佳模型
    best_name = max(all_metrics, key=lambda k: all_metrics[k]['f1_macro'])
    best_model = best_models[best_name]
    print(f"\n  最佳模型: {best_name} (F1_macro={all_metrics[best_name]['f1_macro']:.4f})")

    # 学习曲线
    plot_learning_curves(best_model, X_train, y_train, best_name,
                         os.path.join(RESULT_DIR, "learning_curve.png"))

    # ---- 汇总表 ----
    print(f"\n\n{'='*60}")
    print("[3/6] 生成结果汇总...")
    print("=" * 60)
    summary_df = generate_summary_table(all_metrics, RESULT_DIR)

    # ---- 保存模型 ----
    print(f"\n\n{'='*60}")
    print("[4/6] 保存最佳模型...")
    print("=" * 60)
    model_save_path = os.path.join(RESULT_DIR, f"best_model_{best_name.replace(' ', '_')}.pkl")
    with open(model_save_path, 'wb') as f:
        pickle.dump(best_model, f)
    print(f"  最佳模型已保存至: {model_save_path}")

    # 保存所有模型的评估结果
    results_save = {
        'all_metrics': all_metrics,
        'best_models': best_models,
        'summary': summary_df,
    }
    with open(os.path.join(RESULT_DIR, "evaluation_results.pkl"), 'wb') as f:
        pickle.dump(results_save, f)
    print(f"  评估结果已保存至: {os.path.join(RESULT_DIR, 'evaluation_results.pkl')}")

    # ---- 详细分类报告 ----
    print(f"\n\n{'='*60}")
    print("[5/6] 各模型分类报告...")
    print("=" * 60)

    for name, m in all_metrics.items():
        report = classification_report(y_test, m['y_pred'],
                                       target_names=CLASSES, digits=4)
        report_path = os.path.join(RESULT_DIR, f"classification_report_{name.replace(' ', '_')}.txt")
        with open(report_path, 'w') as f:
            f.write(f"Model: {name}\n")
            f.write(f"Best Params: {best_models[name].get_params()}\n")
            f.write("-" * 50 + "\n")
            f.write(report)
        print(f"  {name} 报告已保存至: {report_path}")

    # ---- 完成 ----
    print(f"\n\n{'='*60}")
    print("[6/6] 成员 3 模型训练与评估完成！")
    print("=" * 60)
    print(f"\n  最终结论:")
    for name, m in sorted(all_metrics.items(), key=lambda x: x[1]['f1_macro'], reverse=True):
        print(f"    {name:15s}  Acc={m['accuracy']:.4f}  "
              f"F1_macro={m['f1_macro']:.4f}  F1_weighted={m['f1_weighted']:.4f}")

    return all_metrics, best_models


if __name__ == "__main__":
    all_metrics, best_models = main()
