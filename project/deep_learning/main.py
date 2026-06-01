"""
皮肤病变分类 — 深度学习实验入口

用法（在项目根目录「大作业」下执行）:

  py -m deep_learning.main --stats
  py -m deep_learning.main --train --eval --backbone all
  py -m deep_learning.main --plot
  py -m deep_learning.main --predict
  py -m deep_learning.main --predict --submitdemo
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from . import config
from .dataset import create_dataloaders, split_train_val
from .evaluate import build_comparison_table, run_full_evaluation
from .predict import generate_output_csv, generate_submitdemo_output
from .train import fit
from .utils import is_augmented, set_seed
from .visualize import plot_all_figures


def print_dataset_stats() -> None:
    train_df = pd.read_csv(config.TRAIN_LABEL_CSV)
    test_df = pd.read_csv(config.TEST_LABEL_CSV)
    train_split, val_df = split_train_val(train_df)

    print("=" * 60)
    print("processed dataset 统计")
    print("=" * 60)
    for name, df in [
        ("官方 train (480)", train_df),
        ("划分后 train", train_split),
        ("划分后 val", val_df),
        ("官方 test (120)", test_df),
    ]:
        aug = df["image_id"].map(is_augmented)
        print(f"\n[{name}] 总数={len(df)}")
        print(f"  原图={ (~aug).sum() }, 增强图={ aug.sum() }")
        print(f"  类别分布:\n{df['dx'].value_counts().to_string()}")

    print("\n说明:")
    print("  - 训练/验证/测试图像已预处理 (256×256, 去毛, 亮度归一化)")
    print("  - val 按 lesion 原图 ID 分组划分，减轻增强样本泄漏")
    print("  - 测试集评估时分别报告「原图」「增强图」指标")
    print(f"  - 类别权重: {config.USE_CLASS_WEIGHTS}")
    print(f"  - 最佳模型指标: {config.CHECKPOINT_METRIC}")


def run_train(
    backbones: list[str],
    loaders,
    dfs: dict,
    use_class_weights: bool,
) -> None:
    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    for backbone in backbones:
        print(f"\n{'#' * 60}\n开始训练: {backbone}\n{'#' * 60}")
        fit(
            backbone,
            loaders,
            train_df=dfs["train"],
            use_class_weights=use_class_weights,
        )


def run_eval(backbones: list[str], dfs: dict) -> None:
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_paths = []
    for backbone in backbones:
        ckpt = config.CHECKPOINT_DIR / f"{backbone}_best.pt"
        if not ckpt.exists():
            print(f"[跳过] 未找到权重: {ckpt}")
            continue
        run_full_evaluation(backbone, dfs, ckpt)
        json_paths.append(config.RESULTS_DIR / f"{backbone}_evaluation.json")

    if json_paths:
        table = build_comparison_table(json_paths)
        out_csv = config.RESULTS_DIR / "model_comparison.csv"
        table.to_csv(out_csv, index=False)
        print(f"\n对比表已保存: {out_csv}")
        print(table.to_string(index=False))


def run_predict(backbone: str, submitdemo: bool) -> None:
    if submitdemo:
        generate_submitdemo_output(backbone=backbone)
    else:
        path = generate_output_csv(backbone=backbone)
        # 同时写一份通用名 output.csv 便于提交说明引用
        generic = config.RESULTS_DIR / "output.csv"
        pd.read_csv(path).to_csv(generic, index=False)
        print(f"副本: {generic}")


def parse_args():
    parser = argparse.ArgumentParser(description="皮肤病变深度学习分类")
    parser.add_argument("--train", action="store_true", help="执行训练")
    parser.add_argument("--eval", action="store_true", help="执行评估")
    parser.add_argument("--plot", action="store_true", help="绘制训练曲线与错例图")
    parser.add_argument("--predict", action="store_true", help="生成 output.csv")
    parser.add_argument(
        "--submitdemo",
        action="store_true",
        help="对 Submitdemo_Proj2/image 推理并写入该目录 output.csv",
    )
    parser.add_argument("--stats", action="store_true", help="仅打印数据统计")
    parser.add_argument(
        "--backbone",
        type=str,
        default=config.DEFAULT_BACKBONE_FOR_INFERENCE,
        choices=[*config.BACKBONES, "all"],
        help="骨干网络",
    )
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument(
        "--no-class-weights",
        action="store_true",
        help="关闭类别加权损失",
    )
    return parser.parse_args()


def resolve_backbones(name: str) -> list[str]:
    return list(config.BACKBONES) if name == "all" else [name]


def main():
    args = parse_args()
    set_seed(config.SEED)

    backbones_for_plot = (
        list(config.BACKBONES)
        if args.backbone == "all"
        else [args.backbone]
    )

    if args.stats:
        print_dataset_stats()
        return

    if args.plot and not (args.train or args.eval or args.predict):
        plot_all_figures(backbones_for_plot)
        return

    if args.predict and not (args.train or args.eval):
        backbone = (
            config.DEFAULT_BACKBONE_FOR_INFERENCE
            if args.backbone == "all"
            else args.backbone
        )
        run_predict(backbone, args.submitdemo)
        return

    if not args.train and not args.eval:
        print_dataset_stats()
        print(
            "\n提示: --train / --eval / --plot / --predict\n"
            "  例: py -m deep_learning.main --plot\n"
            "      py -m deep_learning.main --predict --backbone mobilenet_v2"
        )
        return

    config.NUM_EPOCHS = args.epochs
    loaders, dfs = create_dataloaders(batch_size=args.batch_size)
    backbones = resolve_backbones(args.backbone)
    use_class_weights = not args.no_class_weights

    if args.train:
        run_train(backbones, loaders, dfs, use_class_weights)

    if args.eval:
        run_eval(backbones, dfs)

    if args.plot or args.train:
        plot_all_figures(backbones if args.backbone == "all" else [args.backbone])

    if args.predict:
        bb = (
            config.DEFAULT_BACKBONE_FOR_INFERENCE
            if args.backbone == "all"
            else args.backbone
        )
        run_predict(bb, args.submitdemo)


if __name__ == "__main__":
    main()
