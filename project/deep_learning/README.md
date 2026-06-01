# 皮肤病变深度学习分类

## 依赖

```bash
py -m pip install -r deep_learning/requirements.txt
```

## 常用命令（在大作业根目录执行）

| 命令 | 说明 |
|------|------|
| `py -m deep_learning.main --stats` | 数据统计 |
| `py -m deep_learning.main --train --eval --plot --backbone all` | 训练、评估、出图 |
| `py -m deep_learning.main --plot` | 仅根据已有结果绘制曲线/错例 |
| `py -m deep_learning.main --predict` | 测试集推理 → `outputs/results/output_mobilenet_v2.csv` |
| `py -m deep_learning.main --predict --submitdemo` | Submitdemo 目录推理 → `Submitdemo_Proj2/output.csv` |

## 训练相关配置（`config.py`）

- `USE_CLASS_WEIGHTS = True`：类别加权交叉熵
- `CHECKPOINT_METRIC = "macro_f1"`：按验证 Macro-F1 保存最佳模型
- `--no-class-weights`：关闭类别权重

## 输出目录

```
outputs/
├── checkpoints/     # 模型权重
├── results/         # JSON、CSV、history、output.csv
└── figures/         # 训练曲线、错例图（供 report 引用）
```

## 报告插图

LaTeX 报告 `report/experiment_report.tex` 引用 `outputs/figures/` 下 PNG。
