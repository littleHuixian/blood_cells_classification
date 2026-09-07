# -*- coding: gbk -*-
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms


class TransformSubset(Dataset):
    """为数据子集单独应用训练或验证变换。"""

    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        image, label = self.subset[index]
        return self.transform(image), label

    def __len__(self):
        return len(self.subset)


def set_seed(seed):
    """固定随机种子，保证数据划分和训练结果尽量可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_dataloaders(data_dir, batch_size, image_size, seed):
    """读取 ImageFolder 数据，并按 8:1:1 划分训练、验证和测试集。"""
    set_seed(seed)
    base_dataset = datasets.ImageFolder(data_dir)
    train_size = int(0.8 * len(base_dataset))
    val_size = int(0.1 * len(base_dataset))
    test_size = len(base_dataset) - train_size - val_size
    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset, test_subset = random_split(
        base_dataset,
        [train_size, val_size, test_size],
        generator=generator,
    )

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    # 训练集使用随机增强，验证集和测试集只做尺寸和标准化处理。
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
        normalize,
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize,
    ])

    train_dataset = TransformSubset(train_subset, train_transform)
    val_dataset = TransformSubset(val_subset, eval_transform)
    test_dataset = TransformSubset(test_subset, eval_transform)
    loaders = (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        DataLoader(val_dataset, batch_size=batch_size, shuffle=False),
        DataLoader(test_dataset, batch_size=batch_size, shuffle=False),
    )
    return loaders, base_dataset.classes