# -*- coding: gbk -*-
from PIL import Image
import torch


def predict(model, image_path, transform, class_names, device):
    """预测单张图片，并返回类别名称和置信度。"""
    image = transform(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        probabilities = model(image).softmax(dim=1)[0]
    index = probabilities.argmax().item()
    return class_names[index], probabilities[index].item()