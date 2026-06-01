"""测试集评估，并按原图 / 增强图分别统计。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from . import config
from .dataset import SkinLesionDataset, build_transforms
from .models import build_model
from .utils import get_device, is_augmented


@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> pd.DataFrame:
    model.eval()
    rows = []
    for images, labels, image_ids in tqdm(loader, desc="predict", leave=False):
        images = images.to(device)
        outputs = model(images)
        preds = outputs.argmax(1).cpu().numpy()
        labels_np = labels.numpy()
        for i, image_id in enumerate(image_ids):
            rows.append(
                {
                    "image_id": image_id,
                    "true_dx": config.IDX_TO_LABEL[int(labels_np[i])],
                    "pred_dx": config.IDX_TO_LABEL[int(preds[i])],
                    "correct": int(preds[i] == labels_np[i]),
                    "is_augmented": is_augmented(image_id),
                }
            )
    return pd.DataFrame(rows)


def metrics_from_df(df: pd.DataFrame) -> dict:
    y_true = df["true_dx"]
    y_pred = df["pred_dx"]
    return {
        "n_samples": len(df),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=config.CLASS_NAMES
        ).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, labels=config.CLASS_NAMES, zero_division=0
        ),
    }


def evaluate_split(
    model: nn.Module,
    image_dir: Path,
    labels_df: pd.DataFrame,
    device: torch.device,
    batch_size: int = config.BATCH_SIZE,
) -> dict:
    """对某一划分（如 test）做整体 + 原图/增强图分组评估。"""
    ds = SkinLesionDataset(
        image_dir, labels_df, transform=build_transforms(train=False)
    )
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
    )
    pred_df = collect_predictions(model, loader, device)

    result = {
        "overall": metrics_from_df(pred_df),
        "original_only": metrics_from_df(pred_df[~pred_df["is_augmented"]]),
        "augmented_only": metrics_from_df(pred_df[pred_df["is_augmented"]]),
        "predictions": pred_df,
    }
    return result


def load_model_from_checkpoint(checkpoint_path: Path) -> nn.Module:
    ckpt = torch.load(checkpoint_path, map_location=get_device(), weights_only=False)
    backbone = ckpt["backbone"]
    model = build_model(backbone)
    model.load_state_dict(ckpt["model_state"])
    return model.to(get_device())


def run_full_evaluation(
    backbone: str,
    dfs: dict,
    checkpoint_path: Path | None = None,
    save_dir: Path | None = None,
) -> dict:
    """
    在 val / test 上评估，输出原图 vs 增强图对比表。
    """
    checkpoint_path = checkpoint_path or (
        config.CHECKPOINT_DIR / f"{backbone}_best.pt"
    )
    save_dir = save_dir or config.RESULTS_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    model = load_model_from_checkpoint(checkpoint_path)

    report = {"backbone": backbone, "checkpoint": str(checkpoint_path), "splits": {}}

    split_configs = [
        ("val", config.TRAIN_IMAGE_DIR, dfs["val"]),
        ("test", config.TEST_IMAGE_DIR, dfs["test"]),
    ]

    for split_name, image_dir, df in split_configs:
        eval_result = evaluate_split(model, image_dir, df, device)
        pred_df = eval_result.pop("predictions")
        report["splits"][split_name] = eval_result

        pred_df.to_csv(
            save_dir / f"{backbone}_{split_name}_predictions.csv", index=False
        )

    # 保存 JSON（去掉 classification_report 中的换行也可保留为字符串）
    json_path = save_dir / f"{backbone}_evaluation.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print_summary(report)
    return report


def print_summary(report: dict) -> None:
    backbone = report["backbone"]
    print(f"\n========== {backbone} 评估摘要 ==========")
    for split_name, split_data in report["splits"].items():
        print(f"\n--- {split_name} ---")
        for subset in ("overall", "original_only", "augmented_only"):
            m = split_data[subset]
            tag = {
                "overall": "全部",
                "original_only": "原图",
                "augmented_only": "增强图",
            }[subset]
            print(
                f"  [{tag}] n={m['n_samples']} "
                f"acc={m['accuracy']:.4f} macro_f1={m['macro_f1']:.4f}"
            )


def build_comparison_table(result_files: list[Path]) -> pd.DataFrame:
    """从多个 evaluation.json 汇总原图/增强图准确率对比表。"""
    rows = []
    for path in result_files:
        with open(path, encoding="utf-8") as f:
            report = json.load(f)
        backbone = report["backbone"]
        for split_name, split_data in report["splits"].items():
            for subset in ("overall", "original_only", "augmented_only"):
                m = split_data[subset]
                rows.append(
                    {
                        "model": backbone,
                        "split": split_name,
                        "subset": subset,
                        "n": m["n_samples"],
                        "accuracy": m["accuracy"],
                        "macro_f1": m["macro_f1"],
                    }
                )
    return pd.DataFrame(rows)
