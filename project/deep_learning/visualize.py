"""训练曲线与错例可视化。"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager
from PIL import Image

from . import config

# 中文显示（Windows 常见字体）
for _font in ("Microsoft YaHei", "SimHei", "SimSun"):
    if any(_font in f.name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.sans-serif"] = [_font]
        break
plt.rcParams["axes.unicode_minus"] = False


def plot_training_curves(
    backbone: str,
    history_path: Path | None = None,
    save_dir: Path | None = None,
) -> Path | None:
    history_path = history_path or (
        config.RESULTS_DIR / f"{backbone}_history.json"
    )
    if not history_path.exists():
        print(f"[跳过] 未找到训练历史: {history_path}")
        return None

    with open(history_path, encoding="utf-8") as f:
        history = json.load(f)

    epochs = range(1, len(history["train_loss"]) + 1)
    save_dir = save_dir or config.FIGURES_DIR
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / f"{backbone}_training_curves.png"

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(epochs, history["train_loss"], "b-o", markersize=3, label="训练")
    axes[0].plot(epochs, history["val_loss"], "r-o", markersize=3, label="验证")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title(f"{backbone} — 损失曲线")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(
        epochs, history["train_acc"], "b-o", markersize=3, label="训练 Acc"
    )
    axes[1].plot(
        epochs, history["val_acc"], "r-o", markersize=3, label="验证 Acc"
    )
    if "val_macro_f1" in history:
        axes[1].plot(
            epochs,
            history["val_macro_f1"],
            "g-s",
            markersize=3,
            label="验证 Macro-F1",
        )
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_title(f"{backbone} — 准确率 / F1")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"训练曲线已保存: {out_path}")
    return out_path


def plot_misclassified(
    backbone: str,
    split: str = "test",
    max_total: int = 9,
    predictions_path: Path | None = None,
    image_dir: Path | None = None,
    save_dir: Path | None = None,
) -> Path | None:
    predictions_path = predictions_path or (
        config.RESULTS_DIR / f"{backbone}_{split}_predictions.csv"
    )
    if not predictions_path.exists():
        print(f"[跳过] 未找到预测文件: {predictions_path}")
        return None

    image_dir = image_dir or (
        config.TEST_IMAGE_DIR if split == "test" else config.TRAIN_IMAGE_DIR
    )

    df = pd.read_csv(predictions_path)
    wrong = df[df["correct"] == 0].copy()
    if wrong.empty:
        print(f"[{backbone}] {split} 无错例可展示")
        return None

    # 优先展示 mel 相关错例（临床意义更大）
    priority = wrong[wrong["true_dx"] == "mel"]
    others = wrong[wrong["true_dx"] != "mel"]
    wrong = pd.concat([priority, others]).head(max_total)

    n = len(wrong)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    if n == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n > 1 else [axes]

    for ax, (_, row) in zip(axes, wrong.iterrows()):
        img_path = image_dir / f"{row['image_id']}.jpg"
        img = Image.open(img_path).convert("RGB")
        ax.imshow(img)
        aug_tag = "增强" if row.get("is_augmented", False) else "原图"
        ax.set_title(
            f"{row['image_id']}\n真:{row['true_dx']}  pred:{row['pred_dx']} ({aug_tag})",
            fontsize=9,
        )
        ax.axis("off")

    for ax in axes[len(wrong) :]:
        ax.axis("off")

    save_dir = save_dir or config.FIGURES_DIR
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / f"{backbone}_{split}_misclassified.png"
    fig.suptitle(f"{backbone} — {split} 错例（共 {len(df[df['correct']==0])} 张）", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"错例图已保存: {out_path}")
    return out_path


def plot_all_figures(backbones: list[str] | None = None) -> list[Path]:
    backbones = backbones or list(config.BACKBONES)
    saved: list[Path] = []
    for backbone in backbones:
        p = plot_training_curves(backbone)
        if p:
            saved.append(p)
        p = plot_misclassified(backbone, split="test")
        if p:
            saved.append(p)
    return saved
