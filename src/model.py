# -*- coding: gbk -*-
import torch.nn as nn


class BloodCellCNN(nn.Module):
    """用于血细胞分类的卷积神经网络。"""

    def __init__(self, num_classes):
        super().__init__()

        def block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        # 逐步增加通道数，提取从边缘到细胞形状的高级特征。
        self.features = nn.Sequential(
            block(3, 32),
            block(32, 64),
            block(64, 128),
            block(128, 256),
            block(256, 512),
        )
        # 将空间特征压缩为固定长度，减少全连接层参数量。
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, inputs):
        features = self.avgpool(self.features(inputs))
        return self.classifier(features)