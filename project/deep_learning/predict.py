"""推理并生成 Submitdemo 格式的 output.csv。"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from . import config
from .dataset import SkinLesionDataset, build_transforms
from .evaluate import load_model_from_checkpoint
from .utils import get_device


@torch.no_grad()
def predict_dataloader(model, loader, device) -> list[tuple[str, str]]:
    model.eval()
    rows: list[tuple[str, str]] = []
    for images, _, image_ids in tqdm(loader, desc="inference"):
        images = images.to(device)
        preds = model(images).argmax(1).cpu().tolist()
        for image_id, pred_idx in zip(image_ids, preds):
            rows.append((image_id, config.IDX_TO_LABEL[pred_idx]))
    return rows


def generate_output_csv(
    backbone: str | None = None,
    checkpoint_path: Path | None = None,
    image_dir: Path | None = None,
    label_csv: Path | None = None,
    labels_df: pd.DataFrame | None = None,
    output_path: Path | None = None,
) -> Path:
    """
    对给定样本列表推理，写出 output.csv（image_id, dx）。

    默认：processed dataset 测试集 + 最佳 MobileNetV2 权重。
    可传入 labels_df 或 label_csv 指定 image_id 列表。
    """
    backbone = backbone or config.DEFAULT_BACKBONE_FOR_INFERENCE
    checkpoint_path = checkpoint_path or (
        config.CHECKPOINT_DIR / f"{backbone}_best.pt"
    )
    image_dir = image_dir or config.TEST_IMAGE_DIR
    output_path = output_path or (config.RESULTS_DIR / f"output_{backbone}.csv")

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"未找到权重: {checkpoint_path}")

    if labels_df is None:
        label_csv = label_csv or config.TEST_LABEL_CSV
        labels_df = pd.read_csv(label_csv)[["image_id"]]
    else:
        labels_df = labels_df[["image_id"]].copy()

    ds = SkinLesionDataset(
        image_dir, labels_df, transform=build_transforms(train=False)
    )
    loader = DataLoader(
        ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
    )

    device = get_device()
    model = load_model_from_checkpoint(checkpoint_path)
    rows = predict_dataloader(model, loader, device)

    out_df = pd.DataFrame(rows, columns=["image_id", "dx"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)
    print(f"已写入 {len(out_df)} 条预测: {output_path}")
    return output_path


def generate_submitdemo_output(
    backbone: str | None = None,
    submit_dir: Path | None = None,
) -> Path:
    """
    读取 Submitdemo_Proj2/image 下所有 jpg，按文件名推理。
    输出到该目录下的 output.csv（评测 demo 格式）。
    """
    submit_dir = submit_dir or (config.PROJECT_ROOT / "Submitdemo_Proj2")
    image_dir = submit_dir / "image"
    if not image_dir.exists():
        raise FileNotFoundError(f"未找到图像目录: {image_dir}")

    image_ids = sorted(p.stem for p in image_dir.glob("*.jpg"))
    labels_df = pd.DataFrame({"image_id": image_ids})
    output_path = submit_dir / "output.csv"

    return generate_output_csv(
        backbone=backbone,
        image_dir=image_dir,
        labels_df=labels_df,
        output_path=output_path,
    )
