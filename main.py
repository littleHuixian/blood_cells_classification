# -*- coding: gbk -*-
import torch

import config
from src.dataset import create_dataloaders
from src.evaluate import evaluate
from src.model import BloodCellCNN
from src.train import save_curves, train_model


def main():
    # 创建模型和结果目录。
    config.MODEL_DIR.mkdir(exist_ok=True)
    config.OUTPUT_DIR.mkdir(exist_ok=True)
    # 优先使用 GPU；没有 CUDA 时自动切换到 CPU。
    device = torch.device("cuda" if config.DEVICE == "cuda" and torch.cuda.is_available() else "cpu")
    print(f"使用设备：{device}")
    (train_loader, val_loader, test_loader), class_names = create_dataloaders(
        config.DATA_DIR, config.BATCH_SIZE, config.IMAGE_SIZE, config.SEED
    )
    print(f"检测到 {len(class_names)} 个类别：{class_names}")
    model = BloodCellCNN(len(class_names)).to(device)
    model, history = train_model(model, train_loader, val_loader, device, config)
    torch.save({"model": model.state_dict(), "classes": class_names}, config.MODEL_PATH)
    save_curves(history, config.OUTPUT_DIR)
    evaluate(model, test_loader, class_names, device, config.OUTPUT_DIR)


if __name__ == "__main__":
    main()