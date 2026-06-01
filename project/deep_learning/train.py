"""训练与验证循环。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from . import config
from .models import build_model
from .utils import get_device


def compute_class_weights(train_df: pd.DataFrame) -> torch.Tensor:
    """逆频率类别权重，顺序与 CLASS_NAMES 一致。"""
    counts = train_df["dx"].value_counts()
    total = len(train_df)
    weights = []
    for name in config.CLASS_NAMES:
        c = counts.get(name, 1)
        weights.append(total / (config.NUM_CLASSES * c))
    return torch.tensor(weights, dtype=torch.float32)


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    all_preds: list[int] = []
    all_labels: list[int] = []
    total = 0

    for images, labels, _ in tqdm(loader, desc="eval", leave=False):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
        total += images.size(0)

    acc = sum(p == l for p, l in zip(all_preds, all_labels)) / total
    macro_f1 = f1_score(
        all_labels,
        all_preds,
        average="macro",
        labels=list(range(config.NUM_CLASSES)),
        zero_division=0,
    )
    return total_loss / total, acc, macro_f1


def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels, _ in tqdm(loader, desc="train", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


def _checkpoint_score(val_acc: float, val_macro_f1: float) -> float:
    if config.CHECKPOINT_METRIC == "macro_f1":
        return val_macro_f1
    return val_acc


def save_history(backbone: str, history: dict) -> Path:
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.RESULTS_DIR / f"{backbone}_history.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    return path


def fit(
    backbone: str,
    loaders: dict,
    train_df: pd.DataFrame | None = None,
    num_epochs: int = config.NUM_EPOCHS,
    lr: float = config.LEARNING_RATE,
    use_class_weights: bool = config.USE_CLASS_WEIGHTS,
    checkpoint_dir: Path | None = None,
) -> dict:
    """
    完整训练流程，返回 history 与最佳验证指标。
    checkpoint 保存到 checkpoint_dir/{backbone}_best.pt
    """
    device = get_device()
    checkpoint_dir = checkpoint_dir or config.CHECKPOINT_DIR
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(backbone).to(device)

    if use_class_weights and train_df is not None:
        class_weights = compute_class_weights(train_df).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print(f"  类别权重 ({config.CLASS_NAMES}): {class_weights.cpu().tolist()}")
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = Adam(
        model.parameters(), lr=lr, weight_decay=config.WEIGHT_DECAY
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    history = {
        "backbone": backbone,
        "use_class_weights": use_class_weights,
        "checkpoint_metric": config.CHECKPOINT_METRIC,
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_macro_f1": [],
    }
    best_score = -1.0
    best_val_acc = 0.0
    best_val_f1 = 0.0
    patience_counter = 0
    best_path = checkpoint_dir / f"{backbone}_best.pt"

    metric_name = config.CHECKPOINT_METRIC
    print(f"  最佳模型选择指标: {metric_name}")

    for epoch in range(1, num_epochs + 1):
        print(f"\n[{backbone}] Epoch {epoch}/{num_epochs}")
        train_loss, train_acc = train_one_epoch(
            model, loaders["train"], criterion, optimizer, device
        )
        val_loss, val_acc, val_macro_f1 = evaluate_epoch(
            model, loaders["val"], criterion, device
        )
        score = _checkpoint_score(val_acc, val_macro_f1)
        scheduler.step(score)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_macro_f1"].append(val_macro_f1)

        print(
            f"  train loss={train_loss:.4f} acc={train_acc:.4f} | "
            f"val loss={val_loss:.4f} acc={val_acc:.4f} macro_f1={val_macro_f1:.4f}"
        )

        if score > best_score:
            best_score = score
            best_val_acc = val_acc
            best_val_f1 = val_macro_f1
            patience_counter = 0
            torch.save(
                {
                    "backbone": backbone,
                    "model_state": model.state_dict(),
                    "val_acc": val_acc,
                    "val_macro_f1": val_macro_f1,
                    "checkpoint_metric": metric_name,
                    "epoch": epoch,
                    "class_names": config.CLASS_NAMES,
                    "use_class_weights": use_class_weights,
                },
                best_path,
            )
            print(f"  -> 保存最佳模型 ({metric_name}={score:.4f}): {best_path}")
        else:
            patience_counter += 1
            if patience_counter >= config.EARLY_STOP_PATIENCE:
                print(f"  早停于 epoch {epoch}")
                break

    history_path = save_history(backbone, history)
    print(f"  训练曲线数据已保存: {history_path}")

    return {
        "history": history,
        "history_path": history_path,
        "best_val_acc": best_val_acc,
        "best_val_macro_f1": best_val_f1,
        "best_path": best_path,
    }
