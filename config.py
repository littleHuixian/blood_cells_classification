# -*- coding: gbk -*-
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
# 项目路径配置
DATA_DIR = ROOT_DIR / "bloodcells_dataset"
MODEL_DIR = ROOT_DIR / "models"
OUTPUT_DIR = ROOT_DIR / "outputs"
MODEL_PATH = MODEL_DIR / "best_blood_cell_model.pth"

# 训练超参数
IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 50
EARLY_STOP_PATIENCE = 5
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
SEED = 42
DEVICE = "cuda"