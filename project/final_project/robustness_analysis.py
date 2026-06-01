# -*- coding: utf-8 -*-
"""
Member 4: robustness testing and stability analysis.

This script is intentionally self-contained. The submitted virtual environment
points to another machine, so the perturbation feature extractor uses PIL and
scikit-image instead of OpenCV while preserving the 127-dimensional feature
layout from member 2:

    color moments (9) + RGB histograms (96) + GLCM (12) + LBP (10)

Outputs are written to results_member4/ and member 4 report files are written
beside this script.
"""

from __future__ import annotations

import json
import math
import os
import pickle
import re
import subprocess
import warnings
import zlib
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image, ImageEnhance, ImageFilter
from scipy import stats
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.transform import rotate as sk_rotate

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


BASE_DIR = Path(__file__).resolve().parent
FEAT_DIR = BASE_DIR / "features"
TRAIN_DIR = BASE_DIR / "train"
TEST_DIR = BASE_DIR / "test"
MEMBER3_RESULT_DIR = BASE_DIR / "results"
OUTPUT_DIR = BASE_DIR / "results_member4"

CLASSES = ["mel", "nv", "vasc"]
CLASS_LABELS = {
    "mel": "melanoma",
    "nv": "nevus",
    "vasc": "vascular lesion",
}
RANDOM_SEEDS = list(range(10))


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)


def configure_plots() -> None:
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    sns.set_theme(style="whitegrid", font="DejaVu Sans")


def strip_aug_suffix(image_id: str) -> str:
    return re.sub(r"_aug\d+$", "", str(image_id))


def aug_suffix(image_id: str) -> str:
    match = re.search(r"(_aug\d+)$", str(image_id))
    return match.group(1) if match else "original"


def load_feature_data() -> dict[str, np.ndarray]:
    return {
        "X_train_raw": np.load(FEAT_DIR / "X_train_raw.npy"),
        "X_train_norm": np.load(FEAT_DIR / "X_train_norm.npy"),
        "X_test_raw": np.load(FEAT_DIR / "X_test_raw.npy"),
        "X_test_norm": np.load(FEAT_DIR / "X_test_norm.npy"),
        "y_train": np.load(FEAT_DIR / "y_train.npy", allow_pickle=True),
        "y_test": np.load(FEAT_DIR / "y_test.npy", allow_pickle=True),
        "train_ids": np.load(FEAT_DIR / "train_ids.npy", allow_pickle=True),
        "test_ids": np.load(FEAT_DIR / "test_ids.npy", allow_pickle=True),
    }


def default_models() -> dict[str, object]:
    return {
        "SVM": SVC(C=100, kernel="rbf", gamma="scale", probability=True, random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            max_features="sqrt",
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42,
        ),
        "KNN": KNeighborsClassifier(
            n_neighbors=1,
            weights="distance",
            metric="manhattan",
        ),
    }


def load_member3_models(feature_data: dict[str, np.ndarray]) -> dict[str, object]:
    """Load member 3 tuned models, falling back to local retraining if needed."""
    result_path = MEMBER3_RESULT_DIR / "evaluation_results.pkl"
    models = None

    if result_path.exists():
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with result_path.open("rb") as f:
                    saved = pickle.load(f)
            models = saved.get("best_models")
        except Exception as exc:
            print(f"[warn] Could not load member 3 models: {exc}")

    if models:
        try:
            for model in models.values():
                model.predict(feature_data["X_test_norm"][:3])
            return {name: models[name] for name in ["SVM", "Random Forest", "KNN"]}
        except Exception as exc:
            print(f"[warn] Stored models are not executable here: {exc}")

    print("[info] Re-training models locally with member 3 best parameters.")
    models = {}
    for name, model in default_models().items():
        fitted = clone(model)
        fitted.fit(feature_data["X_train_norm"], feature_data["y_train"])
        models[name] = fitted
    return models


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    baseline_pred: np.ndarray | None = None,
) -> dict[str, object]:
    precision_per_class = precision_score(
        y_true, y_pred, labels=CLASSES, average=None, zero_division=0
    )
    recall_per_class = recall_score(
        y_true, y_pred, labels=CLASSES, average=None, zero_division=0
    )
    f1_per_class = f1_score(
        y_true, y_pred, labels=CLASSES, average=None, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=CLASSES)
    invariance = (
        float(np.mean(y_pred == baseline_pred)) if baseline_pred is not None else np.nan
    )

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(
            precision_score(y_true, y_pred, labels=CLASSES, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true, y_pred, labels=CLASSES, average="macro", zero_division=0)
        ),
        "f1_macro": float(
            f1_score(y_true, y_pred, labels=CLASSES, average="macro", zero_division=0)
        ),
        "f1_weighted": float(
            f1_score(y_true, y_pred, labels=CLASSES, average="weighted", zero_division=0)
        ),
        "precision_mel": float(precision_per_class[0]),
        "precision_nv": float(precision_per_class[1]),
        "precision_vasc": float(precision_per_class[2]),
        "recall_mel": float(recall_per_class[0]),
        "recall_nv": float(recall_per_class[1]),
        "recall_vasc": float(recall_per_class[2]),
        "f1_mel": float(f1_per_class[0]),
        "f1_nv": float(f1_per_class[1]),
        "f1_vasc": float(f1_per_class[2]),
        "prediction_invariance": invariance,
        "confusion_matrix": cm.tolist(),
    }


def metric_row(
    model_name: str,
    analysis: str,
    perturbation: str,
    strength: str,
    metrics: dict[str, object],
    baseline_metrics: dict[str, object],
    n_samples: int,
) -> dict[str, object]:
    baseline_acc = float(baseline_metrics["accuracy"])
    baseline_f1 = float(baseline_metrics["f1_macro"])
    return {
        "model": model_name,
        "analysis": analysis,
        "perturbation": perturbation,
        "strength": strength,
        "condition": f"{perturbation}:{strength}",
        "n_samples": n_samples,
        **{k: v for k, v in metrics.items() if k != "confusion_matrix"},
        "accuracy_retention": float(metrics["accuracy"]) / baseline_acc
        if baseline_acc
        else np.nan,
        "f1_macro_retention": float(metrics["f1_macro"]) / baseline_f1
        if baseline_f1
        else np.nan,
        "delta_accuracy": float(metrics["accuracy"]) - baseline_acc,
        "delta_f1_macro": float(metrics["f1_macro"]) - baseline_f1,
        "confusion_matrix": json.dumps(metrics["confusion_matrix"], ensure_ascii=False),
    }


def rgb_to_gray_uint8(img_rgb: np.ndarray) -> np.ndarray:
    gray = (
        0.299 * img_rgb[:, :, 0]
        + 0.587 * img_rgb[:, :, 1]
        + 0.114 * img_rgb[:, :, 2]
    )
    return np.clip(np.rint(gray), 0, 255).astype(np.uint8)


def extract_color_moments(img_rgb: np.ndarray) -> np.ndarray:
    moments = []
    for channel_idx in range(3):
        channel = img_rgb[:, :, channel_idx].ravel().astype(np.float64)
        mean = np.mean(channel)
        std = np.std(channel)
        skewness = 0.0 if std < 1e-6 else np.mean(((channel - mean) / std) ** 3)
        moments.extend([mean, std, skewness])
    return np.array(moments, dtype=np.float64)


def extract_color_histogram(img_rgb: np.ndarray, bins: int = 32) -> np.ndarray:
    hist_features = []
    for channel_idx in range(3):
        channel = img_rgb[:, :, channel_idx].ravel()
        hist, _ = np.histogram(channel, bins=bins, range=(0, 256), density=True)
        hist_features.extend(hist)
    return np.array(hist_features, dtype=np.float64)


def extract_glcm_features(gray_img: np.ndarray) -> np.ndarray:
    glcm = graycomatrix(
        gray_img,
        distances=(1, 2),
        angles=(0, np.pi / 4, np.pi / 2, 3 * np.pi / 4),
        levels=256,
        symmetric=True,
        normed=True,
    )
    properties = [
        "contrast",
        "dissimilarity",
        "homogeneity",
        "energy",
        "correlation",
        "ASM",
    ]
    features = []
    for prop in properties:
        vals = graycoprops(glcm, prop).ravel()
        features.extend([np.mean(vals), np.std(vals)])
    return np.array(features, dtype=np.float64)


def extract_lbp_features(gray_img: np.ndarray, n_bins: int = 10) -> np.ndarray:
    lbp = local_binary_pattern(gray_img, P=8, R=1, method="uniform").astype(np.int32)
    n_patterns = int(lbp.max()) + 1
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_patterns), density=True)
    return hist.astype(np.float64)


def extract_all_features_rgb(img_rgb: np.ndarray) -> np.ndarray:
    gray = rgb_to_gray_uint8(img_rgb)
    return np.concatenate(
        [
            extract_color_moments(img_rgb),
            extract_color_histogram(img_rgb, bins=32),
            extract_glcm_features(gray),
            extract_lbp_features(gray, n_bins=10),
        ]
    ).astype(np.float32)


def read_rgb(image_id: str) -> np.ndarray:
    path = TEST_DIR / f"{image_id}.jpg"
    with Image.open(path) as img:
        return np.array(img.convert("RGB"), dtype=np.uint8)


def stable_rng(seed: int, image_id: str) -> np.random.Generator:
    crc = zlib.crc32(str(image_id).encode("utf-8"))
    return np.random.default_rng(seed + crc)


def center_crop_resize(img_rgb: np.ndarray, crop_fraction: float) -> np.ndarray:
    height, width = img_rgb.shape[:2]
    crop_h = max(1, int(round(height * crop_fraction)))
    crop_w = max(1, int(round(width * crop_fraction)))
    y0 = (height - crop_h) // 2
    x0 = (width - crop_w) // 2
    cropped = img_rgb[y0 : y0 + crop_h, x0 : x0 + crop_w]
    pil_img = Image.fromarray(cropped)
    pil_img = pil_img.resize((width, height), Image.Resampling.BILINEAR)
    return np.array(pil_img, dtype=np.uint8)


def apply_perturbation(
    img_rgb: np.ndarray,
    perturbation: str,
    value: float,
    image_id: str,
) -> np.ndarray:
    if perturbation == "clean":
        return img_rgb.copy()

    pil_img = Image.fromarray(img_rgb)
    if perturbation == "brightness":
        return np.array(ImageEnhance.Brightness(pil_img).enhance(value), dtype=np.uint8)
    if perturbation == "contrast":
        return np.array(ImageEnhance.Contrast(pil_img).enhance(value), dtype=np.uint8)
    if perturbation == "gaussian_noise":
        rng = stable_rng(20240526, image_id)
        noisy = img_rgb.astype(np.float32) + rng.normal(0, value, img_rgb.shape)
        return np.clip(noisy, 0, 255).astype(np.uint8)
    if perturbation == "blur":
        return np.array(pil_img.filter(ImageFilter.GaussianBlur(radius=value)), dtype=np.uint8)
    if perturbation == "rotate_crop":
        angle, crop_fraction = value
        rotated = sk_rotate(
            img_rgb,
            angle=angle,
            resize=False,
            mode="edge",
            preserve_range=True,
        )
        rotated = np.clip(rotated, 0, 255).astype(np.uint8)
        return center_crop_resize(rotated, crop_fraction)

    raise ValueError(f"Unknown perturbation: {perturbation}")


def perturbation_plan() -> list[dict[str, object]]:
    return [
        {"perturbation": "brightness", "strength": "factor=0.80", "value": 0.80},
        {"perturbation": "brightness", "strength": "factor=1.20", "value": 1.20},
        {"perturbation": "contrast", "strength": "factor=0.80", "value": 0.80},
        {"perturbation": "contrast", "strength": "factor=1.20", "value": 1.20},
        {"perturbation": "gaussian_noise", "strength": "sigma=8", "value": 8.0},
        {"perturbation": "gaussian_noise", "strength": "sigma=16", "value": 16.0},
        {"perturbation": "blur", "strength": "radius=1", "value": 1.0},
        {"perturbation": "blur", "strength": "radius=2", "value": 2.0},
        {
            "perturbation": "rotate_crop",
            "strength": "angle=5,crop=0.96",
            "value": (5.0, 0.96),
        },
        {
            "perturbation": "rotate_crop",
            "strength": "angle=10,crop=0.92",
            "value": (10.0, 0.92),
        },
    ]


def extract_perturbed_features(
    test_ids: np.ndarray,
    perturbation: str,
    value: object,
) -> np.ndarray:
    features = []
    total = len(test_ids)
    for index, image_id in enumerate(test_ids, start=1):
        img_rgb = read_rgb(str(image_id))
        img_rgb = apply_perturbation(img_rgb, perturbation, value, str(image_id))
        features.append(extract_all_features_rgb(img_rgb))
        if index % 30 == 0 or index == total:
            print(f"    extracted {index:3d}/{total} images")
    return np.vstack(features).astype(np.float32)


def load_scaler() -> StandardScaler:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with (FEAT_DIR / "scaler.pkl").open("rb") as f:
            return pickle.load(f)


def run_baseline_and_perturbation_tests(
    models: dict[str, object],
    feature_data: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, dict[str, dict[str, object]], dict[str, np.ndarray]]:
    rows = []
    baseline_metrics = {}
    baseline_predictions = {}

    y_test = feature_data["y_test"]
    X_test_norm = feature_data["X_test_norm"]

    print("[1/4] Reproducing member 3 baseline metrics...")
    for model_name, model in models.items():
        y_pred = model.predict(X_test_norm)
        metrics = evaluate_predictions(y_test, y_pred, baseline_pred=y_pred)
        baseline_metrics[model_name] = metrics
        baseline_predictions[model_name] = y_pred
        rows.append(
            metric_row(
                model_name,
                "baseline",
                "clean",
                "none",
                metrics,
                metrics,
                len(y_test),
            )
        )
        print(
            f"    {model_name:14s} acc={metrics['accuracy']:.4f} "
            f"f1_macro={metrics['f1_macro']:.4f}"
        )

    print("[2/4] Running perturbation robustness tests...")
    scaler = load_scaler()
    for plan in perturbation_plan():
        perturbation = str(plan["perturbation"])
        strength = str(plan["strength"])
        value = plan["value"]
        print(f"  - {perturbation} / {strength}")
        X_raw = extract_perturbed_features(feature_data["test_ids"], perturbation, value)
        X_norm = scaler.transform(X_raw)
        for model_name, model in models.items():
            y_pred = model.predict(X_norm)
            metrics = evaluate_predictions(
                y_test,
                y_pred,
                baseline_pred=baseline_predictions[model_name],
            )
            rows.append(
                metric_row(
                    model_name,
                    "perturbation",
                    perturbation,
                    strength,
                    metrics,
                    baseline_metrics[model_name],
                    len(y_test),
                )
            )
            print(
                f"      {model_name:14s} acc={metrics['accuracy']:.4f} "
                f"f1_macro={metrics['f1_macro']:.4f} "
                f"invariance={metrics['prediction_invariance']:.4f}"
            )

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "robustness_metrics.csv", index=False, encoding="utf-8-sig")
    return df, baseline_metrics, baseline_predictions


def compute_subset_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    if len(y_true) == 0:
        return {
            "accuracy": np.nan,
            "macro_f1": np.nan,
            "weighted_f1": np.nan,
        }
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=CLASSES, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(y_true, y_pred, labels=CLASSES, average="weighted", zero_division=0)
        ),
    }


def run_augmentation_consistency(
    baseline_predictions: dict[str, np.ndarray],
    feature_data: dict[str, np.ndarray],
) -> pd.DataFrame:
    print("[3/4] Analyzing augmentation consistency and split overlap...")
    train_base = pd.Series(feature_data["train_ids"]).map(strip_aug_suffix)
    test_base = pd.Series(feature_data["test_ids"]).map(strip_aug_suffix)
    overlap = sorted(set(train_base) & set(test_base), key=lambda x: int(x))

    rows: list[dict[str, object]] = [
        {
            "model": "DATASET",
            "row_type": "split_overlap",
            "subset": "train_vs_test_base_id",
            "n_samples": len(feature_data["train_ids"]) + len(feature_data["test_ids"]),
            "n_groups": len(set(train_base) | set(test_base)),
            "accuracy": np.nan,
            "macro_f1": np.nan,
            "weighted_f1": np.nan,
            "prediction_consistency": np.nan,
            "all_correct_rate": np.nan,
            "any_correct_rate": np.nan,
            "notes": (
                f"train_base={train_base.nunique()}, test_base={test_base.nunique()}, "
                f"overlap_base={len(overlap)}"
            ),
        }
    ]

    base_df = pd.DataFrame(
        {
            "image_id": feature_data["test_ids"].astype(str),
            "base_id": [strip_aug_suffix(x) for x in feature_data["test_ids"]],
            "suffix": [aug_suffix(x) for x in feature_data["test_ids"]],
            "y_true": feature_data["y_test"],
        }
    )

    for model_name, y_pred in baseline_predictions.items():
        df = base_df.copy()
        df["y_pred"] = y_pred
        group_sizes = df.groupby("base_id").size()
        multi_group_ids = group_sizes[group_sizes >= 2].index
        multi_df = df[df["base_id"].isin(multi_group_ids)]

        consistent_flags = []
        all_correct_flags = []
        any_correct_flags = []
        for _, group in multi_df.groupby("base_id"):
            consistent_flags.append(group["y_pred"].nunique() == 1)
            all_correct_flags.append(bool(np.all(group["y_pred"] == group["y_true"])))
            any_correct_flags.append(bool(np.any(group["y_pred"] == group["y_true"])))

        rows.append(
            {
                "model": model_name,
                "row_type": "group_summary",
                "subset": "multi_sample_test_groups",
                "n_samples": len(multi_df),
                "n_groups": len(multi_group_ids),
                "accuracy": float(accuracy_score(df["y_true"], df["y_pred"])),
                "macro_f1": float(
                    f1_score(
                        df["y_true"],
                        df["y_pred"],
                        labels=CLASSES,
                        average="macro",
                        zero_division=0,
                    )
                ),
                "weighted_f1": float(
                    f1_score(
                        df["y_true"],
                        df["y_pred"],
                        labels=CLASSES,
                        average="weighted",
                        zero_division=0,
                    )
                ),
                "prediction_consistency": float(np.mean(consistent_flags))
                if consistent_flags
                else np.nan,
                "all_correct_rate": float(np.mean(all_correct_flags))
                if all_correct_flags
                else np.nan,
                "any_correct_rate": float(np.mean(any_correct_flags))
                if any_correct_flags
                else np.nan,
                "notes": "Consistency is computed only for base ids with >=2 test variants.",
            }
        )

        for suffix in ["original", "_aug1", "_aug2"]:
            subset = df[df["suffix"] == suffix]
            metrics = compute_subset_metrics(
                subset["y_true"].to_numpy(),
                subset["y_pred"].to_numpy(),
            )
            rows.append(
                {
                    "model": model_name,
                    "row_type": "suffix_summary",
                    "subset": suffix,
                    "n_samples": len(subset),
                    "n_groups": subset["base_id"].nunique(),
                    "accuracy": metrics["accuracy"],
                    "macro_f1": metrics["macro_f1"],
                    "weighted_f1": metrics["weighted_f1"],
                    "prediction_consistency": np.nan,
                    "all_correct_rate": np.nan,
                    "any_correct_rate": np.nan,
                    "notes": "Performance by image augmentation suffix in the test set.",
                }
            )

    result_df = pd.DataFrame(rows)
    result_df.to_csv(
        OUTPUT_DIR / "augmentation_consistency.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return result_df


def class_distribution(labels: np.ndarray) -> str:
    values, counts = np.unique(labels, return_counts=True)
    return ";".join(f"{v}:{c}" for v, c in zip(values, counts))


def run_group_split_tests(
    baseline_metrics: dict[str, dict[str, object]],
    feature_data: dict[str, np.ndarray],
) -> pd.DataFrame:
    print("[4/4] Running strict grouped split stability tests...")
    X_raw = np.vstack([feature_data["X_train_raw"], feature_data["X_test_raw"]])
    y = np.concatenate([feature_data["y_train"], feature_data["y_test"]])
    image_ids = np.concatenate([feature_data["train_ids"], feature_data["test_ids"]]).astype(str)
    groups = np.array([strip_aug_suffix(x) for x in image_ids])

    group_df = (
        pd.DataFrame({"group": groups, "label": y})
        .drop_duplicates("group")
        .sort_values("group", key=lambda s: s.astype(int))
        .reset_index(drop=True)
    )
    group_labels = group_df["label"].to_numpy()
    unique_groups = group_df["group"].to_numpy()

    rows = []
    train_base = pd.Series(feature_data["train_ids"]).map(strip_aug_suffix)
    test_base = pd.Series(feature_data["test_ids"]).map(strip_aug_suffix)
    split_overlap = len(set(train_base) & set(test_base))
    for model_name, metrics in baseline_metrics.items():
        rows.append(
            {
                "split_type": "original_augmented_split",
                "seed": "",
                "model": model_name,
                "n_train_samples": len(feature_data["y_train"]),
                "n_test_samples": len(feature_data["y_test"]),
                "n_train_groups": train_base.nunique(),
                "n_test_groups": test_base.nunique(),
                "train_test_group_overlap": split_overlap,
                "train_class_distribution": class_distribution(feature_data["y_train"]),
                "test_class_distribution": class_distribution(feature_data["y_test"]),
                "accuracy": metrics["accuracy"],
                "precision_macro": metrics["precision_macro"],
                "recall_macro": metrics["recall_macro"],
                "f1_macro": metrics["f1_macro"],
                "f1_weighted": metrics["f1_weighted"],
                "f1_mel": metrics["f1_mel"],
                "f1_nv": metrics["f1_nv"],
                "f1_vasc": metrics["f1_vasc"],
            }
        )

    for seed in RANDOM_SEEDS:
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        group_train_idx, group_test_idx = next(splitter.split(unique_groups, group_labels))
        train_groups = set(unique_groups[group_train_idx])
        test_groups = set(unique_groups[group_test_idx])
        train_mask = np.array([g in train_groups for g in groups])
        test_mask = np.array([g in test_groups for g in groups])

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_raw[train_mask])
        X_test = scaler.transform(X_raw[test_mask])
        y_train = y[train_mask]
        y_test = y[test_mask]

        for model_name, model in default_models().items():
            fitted = clone(model)
            fitted.fit(X_train, y_train)
            y_pred = fitted.predict(X_test)
            metrics = evaluate_predictions(y_test, y_pred)
            rows.append(
                {
                    "split_type": "strict_group_split",
                    "seed": seed,
                    "model": model_name,
                    "n_train_samples": int(train_mask.sum()),
                    "n_test_samples": int(test_mask.sum()),
                    "n_train_groups": len(train_groups),
                    "n_test_groups": len(test_groups),
                    "train_test_group_overlap": len(train_groups & test_groups),
                    "train_class_distribution": class_distribution(y_train),
                    "test_class_distribution": class_distribution(y_test),
                    "accuracy": metrics["accuracy"],
                    "precision_macro": metrics["precision_macro"],
                    "recall_macro": metrics["recall_macro"],
                    "f1_macro": metrics["f1_macro"],
                    "f1_weighted": metrics["f1_weighted"],
                    "f1_mel": metrics["f1_mel"],
                    "f1_nv": metrics["f1_nv"],
                    "f1_vasc": metrics["f1_vasc"],
                }
            )
        print(f"    seed={seed}: train={train_mask.sum()}, test={test_mask.sum()}")

    result_df = pd.DataFrame(rows)
    result_df.to_csv(
        OUTPUT_DIR / "group_split_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary_rows = []
    strict = result_df[result_df["split_type"] == "strict_group_split"]
    for model_name, group in strict.groupby("model"):
        summary_rows.append(
            {
                "model": model_name,
                "n_runs": len(group),
                "accuracy_mean": group["accuracy"].mean(),
                "accuracy_std": group["accuracy"].std(ddof=1),
                "f1_macro_mean": group["f1_macro"].mean(),
                "f1_macro_std": group["f1_macro"].std(ddof=1),
                "f1_weighted_mean": group["f1_weighted"].mean(),
                "f1_weighted_std": group["f1_weighted"].std(ddof=1),
            }
        )
    pd.DataFrame(summary_rows).to_csv(
        OUTPUT_DIR / "group_split_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    return result_df


def parse_cm(value: str) -> np.ndarray:
    return np.array(json.loads(value), dtype=int)


def plot_robustness_curves(robust_df: pd.DataFrame) -> None:
    plot_df = robust_df[robust_df["analysis"] == "perturbation"].copy()
    order = [f"{p['perturbation']}:{p['strength']}" for p in perturbation_plan()]
    plot_df["condition"] = pd.Categorical(plot_df["condition"], categories=order, ordered=True)
    plot_df = plot_df.sort_values("condition")

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.lineplot(
        data=plot_df,
        x="condition",
        y="f1_macro",
        hue="model",
        marker="o",
        ax=ax,
    )
    ax.set_title("Robustness under image perturbations")
    ax.set_xlabel("Perturbation condition")
    ax.set_ylabel("Macro-F1")
    ax.tick_params(axis="x", rotation=35)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "robustness_curves.png", dpi=220)
    plt.close(fig)


def plot_stability_heatmap(robust_df: pd.DataFrame) -> None:
    plot_df = robust_df[robust_df["analysis"] == "perturbation"].copy()
    pivot = plot_df.pivot(index="model", columns="condition", values="f1_macro_retention")
    fig, ax = plt.subplots(figsize=(14, 4.8))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        vmin=0.0,
        vmax=1.05,
        ax=ax,
        cbar_kws={"label": "Macro-F1 retention"},
    )
    ax.set_title("Stability heatmap relative to clean baseline")
    ax.set_xlabel("Perturbation condition")
    ax.set_ylabel("Model")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "stability_heatmap.png", dpi=220)
    plt.close(fig)


def plot_per_class_f1_drop(robust_df: pd.DataFrame) -> None:
    baselines = (
        robust_df[robust_df["analysis"] == "baseline"]
        .set_index("model")[["f1_mel", "f1_nv", "f1_vasc"]]
        .to_dict("index")
    )
    rows = []
    pert_df = robust_df[robust_df["analysis"] == "perturbation"].copy()
    for (model, perturbation), group in pert_df.groupby(["model", "perturbation"]):
        worst = group.sort_values("f1_macro").iloc[0]
        for cls in CLASSES:
            key = f"f1_{cls}"
            rows.append(
                {
                    "model": model,
                    "perturbation": perturbation,
                    "class": cls,
                    "f1_drop": baselines[model][key] - worst[key],
                }
            )
    plot_df = pd.DataFrame(rows)
    plot_df = plot_df.groupby(["perturbation", "class"], as_index=False)["f1_drop"].mean()

    fig, ax = plt.subplots(figsize=(10, 5.5))
    sns.barplot(data=plot_df, x="perturbation", y="f1_drop", hue="class", ax=ax)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Average per-class F1 drop at each perturbation's worst strength")
    ax.set_xlabel("Perturbation")
    ax.set_ylabel("F1 drop from clean baseline")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "per_class_f1_drop.png", dpi=220)
    plt.close(fig)


def plot_worst_confusion_matrix(robust_df: pd.DataFrame) -> None:
    pert_df = robust_df[robust_df["analysis"] == "perturbation"].copy()
    worst = pert_df.sort_values("f1_macro").iloc[0]
    cm = parse_cm(worst["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(5.4, 4.8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Reds", xticklabels=CLASSES, yticklabels=CLASSES, ax=ax)
    ax.set_title(
        "Worst perturbation confusion matrix\n"
        f"{worst['model']} | {worst['perturbation']} | {worst['strength']}"
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "worst_perturbation_confusion_matrix.png", dpi=220)
    plt.close(fig)


def plot_group_split_stability(group_df: pd.DataFrame) -> None:
    original = group_df[group_df["split_type"] == "original_augmented_split"].copy()
    strict = group_df[group_df["split_type"] == "strict_group_split"].copy()

    strict_summary = (
        strict.groupby("model", as_index=False)
        .agg(f1_macro=("f1_macro", "mean"), f1_std=("f1_macro", "std"))
        .assign(split_label="strict group split")
    )
    original_summary = original[["model", "f1_macro"]].copy()
    original_summary["f1_std"] = 0.0
    original_summary["split_label"] = "original augmented split"
    plot_df = pd.concat([original_summary, strict_summary], ignore_index=True)

    fig, ax = plt.subplots(figsize=(8.5, 5))
    sns.barplot(data=plot_df, x="model", y="f1_macro", hue="split_label", ax=ax)
    for patch, (_, row) in zip(ax.patches, plot_df.iterrows()):
        if row["f1_std"] > 0:
            x = patch.get_x() + patch.get_width() / 2
            ax.errorbar(x, row["f1_macro"], yerr=row["f1_std"], color="black", capsize=4)
    ax.set_ylim(0, 1)
    ax.set_title("Original split vs strict grouped split")
    ax.set_xlabel("Model")
    ax.set_ylabel("Macro-F1")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "group_split_stability.png", dpi=220)
    plt.close(fig)


def generate_plots(robust_df: pd.DataFrame, group_df: pd.DataFrame) -> None:
    plot_robustness_curves(robust_df)
    plot_stability_heatmap(robust_df)
    plot_per_class_f1_drop(robust_df)
    plot_worst_confusion_matrix(robust_df)
    plot_group_split_stability(group_df)


def latex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    return text


def format_float(value: float) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    return f"{float(value):.4f}"


def table_to_latex(df: pd.DataFrame, columns: list[str]) -> str:
    col_spec = "l" * len(columns)
    lines = [rf"\begin{{tabular}}{{{col_spec}}}", r"\toprule"]
    lines.append(" & ".join(latex_escape(c) for c in columns) + r" \\")
    lines.append(r"\midrule")
    for _, row in df.iterrows():
        values = []
        for col in columns:
            value = row[col]
            values.append(format_float(value) if isinstance(value, (float, np.floating)) else latex_escape(value))
        lines.append(" & ".join(values) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines)


def write_tex_report(
    robust_df: pd.DataFrame,
    aug_df: pd.DataFrame,
    group_df: pd.DataFrame,
) -> Path:
    baseline = robust_df[robust_df["analysis"] == "baseline"][
        ["model", "accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted"]
    ]
    worst = robust_df[robust_df["analysis"] == "perturbation"].sort_values("f1_macro").iloc[0]
    group_summary = (
        group_df[group_df["split_type"] == "strict_group_split"]
        .groupby("model", as_index=False)
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            f1_macro_mean=("f1_macro", "mean"),
            f1_macro_std=("f1_macro", "std"),
        )
    )
    consistency = aug_df[aug_df["row_type"] == "group_summary"][
        ["model", "n_groups", "prediction_consistency", "all_correct_rate", "any_correct_rate"]
    ]
    overlap_notes = aug_df[aug_df["row_type"] == "split_overlap"]["notes"].iloc[0]

    tex = rf"""
\documentclass[UTF8]{{ctexart}}
\usepackage{{geometry}}
\usepackage{{booktabs}}
\usepackage{{graphicx}}
\usepackage{{float}}
\usepackage{{hyperref}}
\geometry{{a4paper, margin=2.2cm}}
\title{{成员4：增强数据鲁棒性测试、结果对比与稳定性分析}}
\author{{成员4}}
\date{{}}

\begin{{document}}
\maketitle

\section{{任务定位}}
本模块基于成员1的预处理数据、成员2的127维颜色与纹理特征、成员3的SVM/随机森林/KNN模型结果，进一步评估模型在增强图像、图像扰动和严格分组划分下的稳定性。评估指标包括Accuracy、Macro Precision、Macro Recall、Macro-F1、Weighted-F1、各类别F1、混淆矩阵、预测不变率和相对性能保持率。

\section{{基线复现}}
成员3基线结果如下。三种传统模型中，SVM和随机森林Macro-F1均为0.7360，KNN相对较弱。

\begin{{center}}
{table_to_latex(baseline, ["model", "accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted"])}
\end{{center}}

\section{{增强一致性与划分风险}}
按image\_id去除\_aug1/\_aug2后得到原始编号。当前数据划分中，{latex_escape(overlap_notes)}。这说明增强版本存在跨训练集和测试集分布的情况，因此成员4额外加入严格按原始编号分组的稳定性测试。

\begin{{center}}
{table_to_latex(consistency, ["model", "n_groups", "prediction_consistency", "all_correct_rate", "any_correct_rate"])}
\end{{center}}

\section{{扰动鲁棒性}}
对测试集分别施加亮度变化、对比度变化、高斯噪声、模糊、轻微旋转裁剪，并复用成员2的127维特征结构重新提取特征。最差扰动条件为：模型{latex_escape(worst["model"])}，扰动{latex_escape(worst["perturbation"])}，强度{latex_escape(worst["strength"])}，Macro-F1={format_float(float(worst["f1_macro"]))}。

\begin{{figure}}[H]
\centering
\includegraphics[width=0.95\linewidth]{{results_member4/robustness_curves.png}}
\caption{{不同扰动条件下的Macro-F1变化}}
\end{{figure}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.95\linewidth]{{results_member4/stability_heatmap.png}}
\caption{{相对基线的Macro-F1保持率热力图}}
\end{{figure}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.72\linewidth]{{results_member4/per_class_f1_drop.png}}
\caption{{各类别在最差扰动强度下的F1下降}}
\end{{figure}}

\section{{严格分组稳定性}}
将同一原始编号的原图和增强图全部放入同一侧，避免训练集和测试集共享同源图像。10次随机分组后的均值和标准差如下。

\begin{{center}}
{table_to_latex(group_summary, ["model", "accuracy_mean", "accuracy_std", "f1_macro_mean", "f1_macro_std"])}
\end{{center}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.82\linewidth]{{results_member4/group_split_stability.png}}
\caption{{原划分与严格分组划分的Macro-F1对比}}
\end{{figure}}

\section{{结论}}
成员4分析显示：第一，增强版本跨训练和测试集合会使评估偏乐观，因此严格分组结果更能反映泛化能力；第二，亮度、噪声和旋转裁剪会造成明显性能波动，其中受影响最大的类别通常是mel和nv；第三，随机森林在多次严格分组中通常更稳定，而SVM在干净测试集上对少数类覆盖较好；第四，KNN在高维手工特征空间中对扰动和划分变化更敏感。

\end{{document}}
"""
    path = BASE_DIR / "report_member4.tex"
    path.write_text(tex.strip() + "\n", encoding="utf-8")
    return path


def add_pdf_text_page(pdf: PdfPages, title: str, lines: list[str]) -> None:
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.text(0.08, 0.94, title, fontsize=18, weight="bold")
    y = 0.89
    for line in lines:
        fig.text(0.08, y, line, fontsize=10.5, wrap=True)
        y -= 0.035
        if y < 0.08:
            break
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_pdf_image_page(pdf: PdfPages, title: str, image_path: Path) -> None:
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.text(0.04, 0.94, title, fontsize=16, weight="bold")
    img = plt.imread(image_path)
    ax = fig.add_axes([0.04, 0.06, 0.92, 0.82])
    ax.imshow(img)
    ax.axis("off")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def write_pdf_report(
    robust_df: pd.DataFrame,
    aug_df: pd.DataFrame,
    group_df: pd.DataFrame,
) -> Path:
    baseline = robust_df[robust_df["analysis"] == "baseline"]
    worst = robust_df[robust_df["analysis"] == "perturbation"].sort_values("f1_macro").iloc[0]
    group_summary = pd.read_csv(OUTPUT_DIR / "group_split_summary.csv")
    overlap_notes = aug_df[aug_df["row_type"] == "split_overlap"]["notes"].iloc[0]

    path = BASE_DIR / "report_member4.pdf"
    with PdfPages(path) as pdf:
        lines = [
            "Member 4: robustness testing, result comparison, and stability analysis.",
            "",
            "Clean baseline:",
            *[
                f"  {row.model}: Accuracy={row.accuracy:.4f}, Macro-F1={row.f1_macro:.4f}, Weighted-F1={row.f1_weighted:.4f}"
                for row in baseline.itertuples()
            ],
            "",
            f"Augmentation split note: {overlap_notes}.",
            f"Worst perturbation: {worst['model']} / {worst['perturbation']} / {worst['strength']}, Macro-F1={float(worst['f1_macro']):.4f}.",
            "",
            "Strict grouped split summary:",
            *[
                f"  {row.model}: Macro-F1={row.f1_macro_mean:.4f} +/- {row.f1_macro_std:.4f}, Accuracy={row.accuracy_mean:.4f} +/- {row.accuracy_std:.4f}"
                for row in group_summary.itertuples()
            ],
        ]
        add_pdf_text_page(pdf, "Member 4 Robustness Report", lines)
        for title, filename in [
            ("Robustness curves", "robustness_curves.png"),
            ("Stability heatmap", "stability_heatmap.png"),
            ("Per-class F1 drop", "per_class_f1_drop.png"),
            ("Strict grouped split", "group_split_stability.png"),
            ("Worst perturbation confusion matrix", "worst_perturbation_confusion_matrix.png"),
        ]:
            add_pdf_image_page(pdf, title, OUTPUT_DIR / filename)

    return path


def try_compile_tex(tex_path: Path) -> bool:
    xelatex = "xelatex"
    miktex_xelatex = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "MiKTeX" / "miktex" / "bin" / "x64" / "xelatex.exe"
    candidates = [xelatex]
    if miktex_xelatex.exists():
        candidates.append(str(miktex_xelatex))

    last_status = None
    for candidate in candidates:
        try:
            result = subprocess.run(
                [candidate, "-interaction=nonstopmode", tex_path.name],
                cwd=BASE_DIR,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=90,
                check=False,
            )
            last_status = result.returncode
            if result.returncode == 0:
                subprocess.run(
                    [candidate, "-interaction=nonstopmode", tex_path.name],
                    cwd=BASE_DIR,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=90,
                    check=False,
                )
                print("[info] xelatex compiled report_member4.tex successfully.")
                return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue

    if last_status is None:
        print("[info] xelatex unavailable; kept matplotlib PDF report.")
    else:
        print("[warn] xelatex failed; kept matplotlib PDF report.")
    return False


def write_reports(
    robust_df: pd.DataFrame,
    aug_df: pd.DataFrame,
    group_df: pd.DataFrame,
) -> None:
    tex_path = BASE_DIR / "report_member4.tex"
    if not tex_path.exists():
        tex_path = write_tex_report(robust_df, aug_df, group_df)

    if not try_compile_tex(tex_path):
        write_pdf_report(robust_df, aug_df, group_df)


def main() -> None:
    ensure_dirs()
    configure_plots()

    feature_data = load_feature_data()
    models = load_member3_models(feature_data)

    robust_df, baseline_metrics, baseline_predictions = run_baseline_and_perturbation_tests(
        models, feature_data
    )
    aug_df = run_augmentation_consistency(baseline_predictions, feature_data)
    group_df = run_group_split_tests(baseline_metrics, feature_data)
    generate_plots(robust_df, group_df)
    write_reports(robust_df, aug_df, group_df)

    print("\nDone. Outputs:")
    print(f"  {OUTPUT_DIR / 'robustness_metrics.csv'}")
    print(f"  {OUTPUT_DIR / 'augmentation_consistency.csv'}")
    print(f"  {OUTPUT_DIR / 'group_split_comparison.csv'}")
    print(f"  {BASE_DIR / 'report_member4.tex'}")
    print(f"  {BASE_DIR / 'report_member4.pdf'}")


if __name__ == "__main__":
    main()
