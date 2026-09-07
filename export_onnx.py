# -*- coding: gbk -*-
"""将训练好的 PyTorch 血细胞分类模型导出为 ONNX。"""

from pathlib import Path

import torch

from src.model import BloodCellCNN


ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "bloodcells_dataset"
WEIGHT_PATH = ROOT_DIR / "best_blood_cell_model.pth"
OUTPUT_DIR = ROOT_DIR / "models"
ONNX_PATH = OUTPUT_DIR / "best_blood_cell_model.onnx"
IMAGE_SIZE = 224


def load_weights(model, weight_path, device):
    """加载旧版纯 state_dict 或新版 checkpoint 中的模型权重。"""
    checkpoint = torch.load(weight_path, map_location=device)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        checkpoint = checkpoint["model"]
    model.load_state_dict(checkpoint)
    return model


def main():
    if not WEIGHT_PATH.exists():
        raise FileNotFoundError(f"找不到权重文件：{WEIGHT_PATH}")
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"找不到数据集目录：{DATA_DIR}")

    # 根据数据集类别文件夹数量创建完全匹配的模型结构。
    class_names = sorted(
        folder.name for folder in DATA_DIR.iterdir() if folder.is_dir()
    )
    if not class_names:
        raise ValueError(f"数据集目录中没有类别文件夹：{DATA_DIR}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BloodCellCNN(num_classes=len(class_names)).to(device)
    model = load_weights(model, WEIGHT_PATH, device)
    model.eval()

    # 使用动态 batch 维度，导出后可以输入任意批次大小的图片。
    dummy_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE, device=device)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy_input,
        ONNX_PATH,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["images"],
        output_names=["logits"],
        dynamic_axes={
            "images": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
    )

    print(f"ONNX 模型导出完成：{ONNX_PATH}")
    print(f"类别数量：{len(class_names)}")
    print(f"类别顺序：{class_names}")


if __name__ == "__main__":
    main()