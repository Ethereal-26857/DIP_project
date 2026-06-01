"""PyTorch 数据集与 DataLoader 构建。"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from . import config
from .utils import base_lesion_id, set_seed


def build_transforms(train: bool) -> transforms.Compose:
    """训练时可加轻度在线增强；验证/测试仅 resize + 归一化。"""
    if train:
        return transforms.Compose(
            [
                transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                transforms.ToTensor(),
                transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
        ]
    )


class SkinLesionDataset(Dataset):
    def __init__(
        self,
        image_dir: Path,
        labels_df: pd.DataFrame,
        transform: transforms.Compose | None = None,
    ):
        self.image_dir = Path(image_dir)
        self.labels_df = labels_df.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels_df)

    def __getitem__(self, idx: int):
        row = self.labels_df.iloc[idx]
        image_id = str(row["image_id"])
        label = config.LABEL_TO_IDX[str(row["dx"])]

        path = self.image_dir / f"{image_id}.jpg"
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, label, image_id


def split_train_val(
    train_df: pd.DataFrame,
    val_ratio: float = config.VAL_RATIO,
    seed: int = config.SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    按 lesion 原图 ID 分组划分，避免同一病灶的原图/增强图同时出现在 train 与 val。
    """
    groups = train_df["image_id"].map(base_lesion_id).values
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=val_ratio, random_state=seed
    )
    train_idx, val_idx = next(splitter.split(train_df, groups=groups))
    return (
        train_df.iloc[train_idx].reset_index(drop=True),
        train_df.iloc[val_idx].reset_index(drop=True),
    )


def create_dataloaders(
    batch_size: int = config.BATCH_SIZE,
    num_workers: int = config.NUM_WORKERS,
) -> dict:
    """
    返回 train / val / test 三个 DataLoader，以及对应的 DataFrame（便于评估时分组）。
    """
    set_seed(config.SEED)
    train_df = pd.read_csv(config.TRAIN_LABEL_CSV)
    test_df = pd.read_csv(config.TEST_LABEL_CSV)
    train_split_df, val_df = split_train_val(train_df)

    loaders = {}
    dfs = {"train": train_split_df, "val": val_df, "test": test_df}

    for split, df, train_mode in [
        ("train", train_split_df, True),
        ("val", val_df, False),
        ("test", test_df, False),
    ]:
        image_dir = (
            config.TRAIN_IMAGE_DIR if split != "test" else config.TEST_IMAGE_DIR
        )
        ds = SkinLesionDataset(
            image_dir, df, transform=build_transforms(train=train_mode)
        )
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    return loaders, dfs
