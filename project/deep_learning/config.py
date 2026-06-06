"""深度学习实验配置（路径、超参数、类别映射）。"""
from pathlib import Path

# 项目根目录（大作业文件夹）
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 加工后的数据集
PROCESSED_DIR = PROJECT_ROOT / "final_project"
TRAIN_IMAGE_DIR = PROCESSED_DIR / "train"
TEST_IMAGE_DIR = PROCESSED_DIR / "test"
TRAIN_LABEL_CSV = PROCESSED_DIR / "train_label.csv"
TEST_LABEL_CSV = PROCESSED_DIR / "test_label.csv"

# 实验输出
OUTPUT_DIR = PROJECT_ROOT / "deep_learning" / "outputs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
RESULTS_DIR = OUTPUT_DIR / "results"
FIGURES_DIR = OUTPUT_DIR / "figures"

# 类别
CLASS_NAMES = ["mel", "nv", "vasc"]
NUM_CLASSES = len(CLASS_NAMES)
LABEL_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}
IDX_TO_LABEL = {i: name for name, i in LABEL_TO_IDX.items()}

# 图像
IMAGE_SIZE = 256  # 与预处理 demo 一致
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# 训练
SEED = 42
VAL_RATIO = 0.2  # 从 train_label 中再划分验证集（按原图 ID 分组）
BATCH_SIZE = 16
NUM_EPOCHS = 30
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 0  # Windows 上建议 0，若 Linux 可改为 4
EARLY_STOP_PATIENCE = 8
USE_CLASS_WEIGHTS = True  # 按训练集类别频率的逆频率加权
CHECKPOINT_METRIC = "macro_f1"  # 选最佳模型: "macro_f1" 或 "accuracy"

# 支持的骨干网络
BACKBONES = ("resnet18", "mobilenet_v2")
DEFAULT_BACKBONE_FOR_INFERENCE = "mobilenet_v2"
