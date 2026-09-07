# -*- coding: gbk -*-
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def evaluate(model, test_loader, class_names, device, output_dir):
    """在测试集上评估模型，并保存混淆矩阵。"""
    model.eval()
    targets, predictions = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images.to(device))
            targets.extend(labels.numpy())
            predictions.extend(outputs.argmax(1).cpu().numpy())

    print(f"测试集准确率：{accuracy_score(targets, predictions):.4f}")
    print(classification_report(targets, predictions, target_names=class_names, zero_division=0))
    matrix = confusion_matrix(targets, predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("预测类别")
    plt.ylabel("真实类别")
    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "confusion_matrix.png")
    plt.close()