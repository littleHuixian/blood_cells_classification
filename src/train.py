# -*- coding: gbk -*-
import copy

import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from tqdm.auto import tqdm


def train_model(model, train_loader, val_loader, device, config):
    """训练模型，并返回验证损失最低时的模型权重。"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )
    best_loss = float("inf")
    best_weights = copy.deepcopy(model.state_dict())
    patience = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(config.NUM_EPOCHS):
        train_loss, train_acc = _run_epoch(
            model, train_loader, criterion, optimizer, device, training=True
        )
        val_loss, val_acc = _run_epoch(
            model, val_loader, criterion, optimizer, device, training=False
        )
        scheduler.step(val_loss)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        print(
            f"第 {epoch + 1}/{config.NUM_EPOCHS} 轮 | "
            f"训练损失 {train_loss:.4f}，验证损失 {val_loss:.4f}，"
            f"验证准确率 {val_acc:.2%}"
        )

        if val_loss < best_loss:
            best_loss = val_loss
            best_weights = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
            if patience >= config.EARLY_STOP_PATIENCE:
                break

    model.load_state_dict(best_weights)
    return model, history


def _run_epoch(model, loader, criterion, optimizer, device, training):
    """执行一个训练或验证轮次。"""
    model.train(training)
    total_loss = 0.0
    correct = 0
    total = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for images, labels in tqdm(loader, leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total


def save_curves(history, output_dir):
    """保存训练损失曲线和准确率曲线。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    for metric, filename, title in [
        ("loss", "loss_curve.png", "损失曲线"),
        ("acc", "accuracy_curve.png", "准确率曲线"),
    ]:
        plt.figure(figsize=(8, 5))
        plt.plot(history[f"train_{metric}"], label="训练集")
        plt.plot(history[f"val_{metric}"], label="验证集")
        plt.title(title)
        plt.xlabel("训练轮次")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / filename)
        plt.close()