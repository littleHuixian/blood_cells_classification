# -*- coding: gbk -*-
import os
import random
import time
import copy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from tabulate import tabulate

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import datasets, transforms, models

from torchinfo import summary
from tqdm.auto import tqdm

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    cohen_kappa_score, matthews_corrcoef, roc_auc_score,
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import label_binarize
from sklearn.manifold import TSNE
from sklearn.calibration import calibration_curve

# from pytorch_grad_cam import GradCAM  # 如需 Grad-CAM，请取消注释
from pytorch_grad_cam.utils.image import show_cam_on_image
from captum.attr import IntegratedGradients, Occlusion

# import gradio as gr  # 如需 Gradio 界面，请取消注释

# %matplotlib inline  # 在 Jupyter 中显示 Matplotlib 图像
def set_all_seeds(seed=42):
    """
    Forces all random number generators to use a fixed seed, enabling
    fully reproducible experiments across CPU and GPU.
    """
    # 1. 设置环境变量
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # PyTorch 确定性算法所需配置（CUDA >= 10.2）
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8' 
    
    # 2. 设置 Python 内置随机数生成器
    random.seed(seed)
    
    # 3. 设置 NumPy 随机数生成器
    np.random.seed(seed)
    
    # 4. 设置 PyTorch 随机数生成器
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # 多 GPU 环境使用
    
    # 5. 配置 CuDNN 后端以保证确定性
    # deterministic=True 确保卷积算法可复现
    torch.backends.cudnn.deterministic = True 
    # benchmark=False 禁用可能产生差异的卷积算法自动调优
    torch.backends.cudnn.benchmark = False
    
    # 6. 在可用时强制 PyTorch 使用确定性算法
    torch.use_deterministic_algorithms(True, warn_only=True)
    
    print(f"Random seed set to {seed} across all environments.")

# 执行随机种子设置
SEED = 42
set_all_seeds(SEED)


# 设置超参数和计算设备
DATA_DIR = './bloodcells_dataset' # 确保与数据集文件夹名称一致
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4

# 如果可用则自动使用 GPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using compute device: {DEVICE}")


# 1. 分别定义训练集和验证集/测试集的数据处理流程
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5), # 数据增强：随机水平翻转
    transforms.RandomVerticalFlip(p=0.5),   # 数据增强：随机垂直翻转
    transforms.RandomRotation(20),          # 数据增强：随机旋转最多 20 度
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    # 注意：验证集和测试集不使用随机翻转或旋转
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 2. 加载不带变换的基础数据集（读取原始 PIL 图片）
full_dataset = datasets.ImageFolder(root=DATA_DIR)
classes = full_dataset.classes
print(f"Detected {len(classes)} classes: {classes}")

# 3. 计算 80% 训练集、10% 验证集和 10% 测试集的大小
total_size = len(full_dataset)
train_size = int(0.8 * total_size)
val_size = int(0.1 * total_size)
test_size = total_size - train_size - val_size 

# 4. 划分原始数据集
train_subset, val_subset, test_subset = random_split(
    full_dataset, [train_size, val_size, test_size]
)

# 5. 创建包装类，为不同子集应用指定的数据变换
class TransformWrapper(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, index):
        image, label = self.subset[index]
        if self.transform:
            image = self.transform(image)
        return image, label
        
    def __len__(self):
        return len(self.subset)

# 为各数据集划分应用对应的数据变换
train_dataset = TransformWrapper(train_subset, transform=train_transforms)
val_dataset = TransformWrapper(val_subset, transform=val_test_transforms)
test_dataset = TransformWrapper(test_subset, transform=val_test_transforms)

# 6. 创建数据加载器
BATCH_SIZE = 32
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Training samples: {len(train_dataset)} | Validation samples: {len(val_dataset)} | Testing samples: {len(test_dataset)}")


# 使用 iter() 和 next() 获取训练集中的一个批次
images, labels = next(iter(train_loader))

print("=== IMAGES TENSOR ===")
# 预期形状：[32, 3, 224, 224]（批次、颜色通道、高度、宽度）
print(f"Shape:       {images.shape}") 
print(f"Data Type:   {images.dtype}")
# 由于进行了标准化，数值大约在 -2.5 到 +2.5 之间，而不是 0 到 1
print(f"Value Range: Min {images.min():.4f} | Max {images.max():.4f}")

print("\n=== LABELS TENSOR ===")
# 预期形状：[32]（批次中的每张图片对应一个整数标签）
print(f"Shape:       {labels.shape}")
print(f"Data Type:   {labels.dtype}")
print(f"First 10:    {labels[:10].tolist()}")

class BloodCellCNN(nn.Module):
    def __init__(self, num_classes):
        super(BloodCellCNN, self).__init__()
        
        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )

        self.features = nn.Sequential(
            conv_block(3, 32),    
            conv_block(32, 64),   
            conv_block(64, 128),  
            conv_block(128, 256), 
            conv_block(256, 512)  
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.5), 
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(128, num_classes)
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


# 动态计算类别文件夹的数量
dynamic_class_names = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
dynamic_num_classes = len(dynamic_class_names)

# 根据类别数量创建模型，并移动到 GPU 或 CPU
model = BloodCellCNN(num_classes=dynamic_num_classes).to(DEVICE)

# 在 Jupyter 中整洁地输出模型摘要
summary(
    model,
    input_size=(32, 3, 224, 224), 
    col_names=[
        "input_size",
        "output_size",
        "num_params",
        "kernel_size",
        "mult_adds",
        "trainable"
    ],
    col_width=18,
    row_settings=["var_names"],
    verbose=0 
)


# 损失函数
criterion = nn.CrossEntropyLoss()

# 使用带 L2 权重衰减的 AdamW 优化器
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

# 学习率调度器：验证损失停滞时降低学习率
scheduler = lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='min', 
    factor=0.5, 
    patience=2)

# ==========================================
# 训练配置
# ==========================================
NUM_EPOCHS = 50
EARLY_STOP_PATIENCE = 5  

best_val_loss = float('inf')
epochs_no_improve = 0
best_model_weights = copy.deepcopy(model.state_dict())
history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

# ==========================================
# 主训练循环
# ==========================================
for epoch in range(1, NUM_EPOCHS + 1):
    start_time = time.time()
    
    # ---------------------------------------
    # 1. 训练阶段
    # ---------------------------------------
    model.train()
    running_train_loss, train_correct, train_total = 0.0, 0, 0
    
    train_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS} [Train]", leave=False)
    
    for images, labels in train_bar:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_train_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        train_correct += torch.sum(preds == labels.data).item()
        train_total += labels.size(0)
        
        train_bar.set_postfix({'loss': f"{loss.item():.4f}"})

    epoch_train_loss = running_train_loss / train_total
    epoch_train_acc = train_correct / train_total

    # ---------------------------------------
    # 2. 验证阶段
    # ---------------------------------------
    model.eval()
    running_val_loss, val_correct, val_total = 0.0, 0, 0
    
    val_bar = tqdm(val_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS} [Val]", leave=False)
    
    with torch.no_grad():
        for images, labels in val_bar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_val_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            val_correct += torch.sum(preds == labels.data).item()
            val_total += labels.size(0)

    epoch_val_loss = running_val_loss / val_total
    epoch_val_acc = val_correct / val_total

    # ---------------------------------------
    # 3. 学习率调度和模型保存
    # ---------------------------------------
    scheduler.step(epoch_val_loss)

    history['train_loss'].append(epoch_train_loss)
    history['train_acc'].append(epoch_train_acc)
    history['val_loss'].append(epoch_val_loss)
    history['val_acc'].append(epoch_val_acc)
    
    epoch_time = time.time() - start_time
    status = ""

    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        best_model_weights = copy.deepcopy(model.state_dict())
        torch.save(model.state_dict(), 'best_blood_cell_model.pth')
        epochs_no_improve = 0
        status = "Saved Best"
    else:
        epochs_no_improve += 1
        status = "No Improve"

    # ---------------------------------------
    # 4. 表格输出（使用 Tabulate）
    # ---------------------------------------
    current_lr = optimizer.param_groups[0]['lr']
    
    # 准备用于 Tabulate 的数据
    table_data = [
        ["Epoch", f"{epoch}"],
        ["Train Loss", f"{epoch_train_loss:.4f}"],
        ["Train Acc", f"{epoch_train_acc*100:.2f}%"],
        ["Val Loss", f"{epoch_val_loss:.4f}"],
        ["Val Acc", f"{epoch_val_acc*100:.2f}%"],
        ["Learning Rate", f"{current_lr:.2e}"],
        ["Epoch Time", f"{epoch_time:.2f}s"],
        ["Patience", f"{epochs_no_improve}/{EARLY_STOP_PATIENCE}"],
        ["Status", status]
    ]
    
    print(f"\nEpoch {epoch}/{NUM_EPOCHS}")
    # rounded_grid 提供清晰的圆角网格样式
    print(tabulate(table_data, headers=["Metric", "Value"], tablefmt="rounded_grid"))
    print("\n")

    if epochs_no_improve >= EARLY_STOP_PATIENCE:
        print("Early stopping triggered!")
        break

# ==========================================
# 训练结束后的处理
# ==========================================
print("Training Complete.")
model.load_state_dict(best_model_weights)



# 设置简洁、专业的绘图风格
sns.set_theme(style="whitegrid", palette="muted", context="notebook")

# 从 history 字典中提取训练记录
epochs = range(1, len(history['train_loss']) + 1)
train_loss = history['train_loss']
val_loss = history['val_loss']

# 将准确率转换为百分比，便于阅读
train_acc = [acc * 100 for acc in history['train_acc']]
val_acc = [acc * 100 for acc in history['val_acc']]

# 创建一个 1 行 2 列的画布
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# ==========================================
# 1. 损失曲线
# ==========================================
axes[0].plot(epochs, train_loss, label='Training Loss', marker='o', linewidth=2.5)
axes[0].plot(epochs, val_loss, label='Validation Loss', marker='s', linewidth=2.5, linestyle='--')
axes[0].set_title('Model Loss Over Time', fontsize=14, fontweight='bold', pad=15)
axes[0].set_xlabel('Epoch', fontsize=12, fontweight='500')
axes[0].set_ylabel('Cross-Entropy Loss', fontsize=12, fontweight='500')
# 强制横坐标使用整数刻度
axes[0].xaxis.set_major_locator(MaxNLocator(integer=True))
axes[0].legend(fontsize=11, loc='upper right')

# ==========================================
# 2. 准确率曲线
# ==========================================
axes[1].plot(epochs, train_acc, label='Training Accuracy', marker='o', linewidth=2.5)
axes[1].plot(epochs, val_acc, label='Validation Accuracy', marker='s', linewidth=2.5, linestyle='--')
axes[1].set_title('Model Accuracy Over Time', fontsize=14, fontweight='bold', pad=15)
axes[1].set_xlabel('Epoch', fontsize=12, fontweight='500')
axes[1].set_ylabel('Accuracy (%)', fontsize=12, fontweight='500')
axes[1].xaxis.set_major_locator(MaxNLocator(integer=True)) 
axes[1].legend(fontsize=11, loc='lower right')

# 调整布局，防止内容重叠并显示图像
plt.tight_layout()
plt.show()


# ==========================================
# 1. 加载已保存的模型
# ==========================================
BEST_MODEL = "/kaggle/working/best_blood_cell_model.pth"

# 将权重加载到已有的模型结构中
model.load_state_dict(torch.load(BEST_MODEL, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# ==========================================
# 2. 在测试集上执行推理
# ==========================================
all_targets = []
all_preds = []
all_probs = []

print("Running inference on test loader...")

with torch.no_grad():
    # 使用 tqdm 为测试集添加进度条
    for images, labels in tqdm(test_loader, desc="Evaluating Test Set"):
        images = images.to(DEVICE)
        
        # 前向传播
        logits = model(images)
        
        # 使用 Softmax 获取概率（ROC AUC 需要概率值）
        probs = F.softmax(logits, dim=1)
        
        # 使用 Argmax 获取最终类别预测
        _, preds = torch.max(logits, 1)
        
        # 移回 CPU 并保存结果
        all_targets.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

# 转换为 NumPy 数组，供 scikit-learn 使用
y_true = np.array(all_targets)
y_pred = np.array(all_preds)
y_prob = np.array(all_probs)

# ==========================================
# 3. 计算评估指标
# ==========================================
# 单值指标
acc = accuracy_score(y_true, y_pred)
kappa = cohen_kappa_score(y_true, y_pred)
mcc = matthews_corrcoef(y_true, y_pred)

# 宏平均（不考虑样本数量，每个类别权重相同）
prec_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
rec_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
auc_macro = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')

# 微平均（根据所有类别的总体结果计算指标）
prec_micro = precision_score(y_true, y_pred, average='micro', zero_division=0)
rec_micro = recall_score(y_true, y_pred, average='micro', zero_division=0)
f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)
auc_micro = roc_auc_score(y_true, y_prob, multi_class='ovr', average='micro')

# ==========================================
# 4. 表格输出
# ==========================================
table_data = [
    ["Accuracy", f"{acc:.4f}", "-"],
    ["Precision", f"{prec_macro:.4f}", f"{prec_micro:.4f}"],
    ["Recall", f"{rec_macro:.4f}", f"{rec_micro:.4f}"],
    ["F1 Score", f"{f1_macro:.4f}", f"{f1_micro:.4f}"],
    ["ROC AUC", f"{auc_macro:.4f}", f"{auc_micro:.4f}"],
    ["Cohen's Kappa", f"{kappa:.4f}", "-"],
    ["MCC", f"{mcc:.4f}", "-"]
]

print("\n")
print(tabulate(table_data, headers=["Metric", "Macro Average", "Micro Average"], tablefmt="double_grid"))


# ==========================================
# 生成并绘制混淆矩阵
# ==========================================

# 计算原始混淆矩阵
cm = confusion_matrix(y_true, y_pred)

# 获取类别名称；如果动态类别名称不存在，则使用数字标签作为后备
try:
    labels = dynamic_class_names
except NameError:
    try:
        labels = class_names
    except NameError:
        labels = [f"Class {i}" for i in range(cm.shape[0])]

# 设置图像样式
plt.figure(figsize=(12, 10))
sns.set_theme(style="white") # 移除热力图背景网格

# 绘制热力图
ax = sns.heatmap(
    cm, 
    annot=True,          # 显示样本数量
    fmt='d',             # 使用整数格式
    cmap='Dark2',        # 使用专业配色
    cbar=True,           # 显示颜色条
    square=True,         # 保持单元格为正方形
    linewidths=0.5,      # 添加细网格线
    linecolor='lightgray',
    xticklabels=labels, 
    yticklabels=labels,
    annot_kws={"size": 12} # 设置单元格数字字号
)

# 设置坐标轴和标题样式
ax.set_title('Blood Cell Classification Confusion Matrix', fontsize=18, fontweight='bold', pad=20)
ax.set_xlabel('Predicted Label', fontsize=14, fontweight='bold', labelpad=15)
ax.set_ylabel('True Label', fontsize=14, fontweight='bold', labelpad=15)

# 调整刻度格式，方便阅读
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.yticks(rotation=0, fontsize=12)

# 确保所有内容完整显示
plt.tight_layout()
plt.show()


# ==========================================
# 详细分类报告
# ==========================================

# 安全获取类别名称，逻辑与混淆矩阵部分保持一致
try:
    class_list = dynamic_class_names
except NameError:
    try:
        class_list = class_names
    except NameError:
        # 如果类别名称丢失，则使用数字索引作为后备
        class_list = [f"Class {i}" for i in range(len(set(y_true)))]

# 生成分类报告文本
report_text = classification_report(
    y_true, 
    y_pred, 
    target_names=class_list, 
    digits=4 # 保留 4 位小数，与前面的表格保持一致
)

# 使用清晰、专业的标题打印报告
print("=" * 60)
print(f"{'PER-CLASS CLASSIFICATION REPORT':^60}")
print("=" * 60)
print(report_text)
print("=" * 60)

# ==========================================
# 多分类 ROC 曲线绘制
# ==========================================

# 根据概率数组确定类别数量
n_classes = y_prob.shape[1]

# 将真实标签二值化，用于 One-vs-Rest ROC 计算
y_true_bin = label_binarize(y_true, classes=range(n_classes))

# 安全获取类别名称
try:
    class_list = dynamic_class_names
except NameError:
    try:
        class_list = class_names
    except NameError:
        class_list = [f"Class {i}" for i in range(n_classes)]

# 设置图像样式
plt.figure(figsize=(10, 8))
sns.set_theme(style="whitegrid")

# 定义包含足够区分度的类别配色
colors = sns.color_palette("tab10", n_colors=n_classes)

# 计算并绘制每个类别的 ROC 曲线和 AUC 面积
for i, color in zip(range(n_classes), colors):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
    roc_auc = auc(fpr, tpr)
    
    plt.plot(
        fpr, 
        tpr, 
        color=color, 
        lw=2, 
        label=f"{class_list[i]} (AUC = {roc_auc:.4f})"
    )

# 绘制对角线基准线（随机猜测）
plt.plot([0, 1], [0, 1], 'k--', lw=2, label="Random Guess (AUC = 0.5000)")

# 设置图像样式
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
plt.title('One-vs-Rest ROC Curves per Class', fontsize=16, fontweight='bold', pad=20)
plt.legend(loc="lower right", fontsize=10, frameon=True, shadow=True)

# 确保所有内容完整显示
plt.tight_layout()
plt.show()

# ==========================================
# 1. 设置和选择目标层
# ==========================================
# 安全获取类别名称
try:
    class_list = dynamic_class_names
except NameError:
    try:
        class_list = class_names
    except NameError:
        class_list = [f"Class {i}" for i in range(8)]

# 选择 Grad-CAM 的目标层
# 自定义 BloodCellCNN 使用 model.features[-1]
# 标准 ResNet18 使用 model.layer4[-1]
if hasattr(model, 'features'):
    target_layers = [model.features[-1]]
elif hasattr(model, 'layer4'):
    target_layers = [model.layer4[-1]]
else:
    raise ValueError("Could not automatically determine the target convolutional layer for Grad-CAM.")

# ==========================================
# 2. 获取一批 25 张测试图片
# ==========================================
images_to_show = []
labels_to_show = []

# 临时打乱测试加载器，随机获取 25 张图片
temp_loader = torch.utils.data.DataLoader(test_dataset, batch_size=25, shuffle=True)
images_tensor, true_labels = next(iter(temp_loader))
images_tensor = images_tensor.to(DEVICE)
true_labels = true_labels.to(DEVICE)

# ==========================================
# 3. 获取预测结果和置信度
# ==========================================
model.eval()
with torch.no_grad():
    logits = model(images_tensor)
    probs = F.softmax(logits, dim=1)
    confidences, preds = torch.max(probs, 1)

# ==========================================
# 4. 生成 Grad-CAM 热力图
# ==========================================
# 初始化 CAM 对象
cam = GradCAM(model=model, target_layers=target_layers)

# targets=None 时，自动使用每张图片得分最高的类别
grayscale_cams = cam(input_tensor=images_tensor, targets=None)

# 辅助函数：还原 Normalize()，以便显示原始 RGB 图片
def denormalize(img_tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(DEVICE)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(DEVICE)
    img = img_tensor * std + mean
    img = torch.clamp(img, 0, 1) # 确保数值严格位于 0 到 1 之间
    return img.cpu().numpy().transpose(1, 2, 0) # 转换为 Matplotlib 使用的 HxWxC 格式

# ==========================================
# 5. 绘制 5x5 图片网格
# ==========================================
fig, axes = plt.subplots(5, 5, figsize=(18, 20))
axes = axes.flatten()

for i in range(25):
    # 1. 获取原始 RGB 图片
    rgb_img = denormalize(images_tensor[i])
    
    # 2. 将热力图叠加到 RGB 图片上
    cam_image = show_cam_on_image(rgb_img, grayscale_cams[i], use_rgb=True)
    
    # 3. 绘制图片
    axes[i].imshow(cam_image)
    axes[i].axis('off')
    
    # 4. 设置标题文本格式
    pred_class = class_list[preds[i].item()]
    true_class = class_list[true_labels[i].item()]
    conf = confidences[i].item() * 100
    
    # 判断预测是否正确
    is_correct = (preds[i] == true_labels[i])
    title_color = 'green' if is_correct else 'red'
    
    # 5. 设置标题
    title_text = f"Pred: {pred_class} ({conf:.1f}%)\nTrue: {true_class}"
    axes[i].set_title(title_text, color=title_color, fontsize=11, fontweight='bold')

plt.suptitle("Grad-CAM: Where is the model looking?", fontsize=20, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()


# ==========================================
# 1. 从测试集提取特征向量
# ==========================================
model.eval()
features_list = []
labels_list = []

print("Extracting latent space features from the test set...")

# 去掉最后的分类层，创建特征提取模型
# BloodCellCNN 的 features 和 avgpool 输出 512 维向量，
# classifier 的第一个全连接层将其降为 128 维
feature_extractor = torch.nn.Sequential(
    model.features,
    model.avgpool,
    torch.nn.Flatten(),
    model.classifier[0], # Flatten
    model.classifier[1], # Dropout
    model.classifier[2], # Linear（512 -> 128）
    model.classifier[3]  # ReLU
).to(DEVICE)
feature_extractor.eval()

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)
        # 将图片输入特征提取器
        feats = feature_extractor(images)
        
        features_list.append(feats.cpu().numpy())
        labels_list.append(labels.numpy())

# 将所有批次拼接为 NumPy 数组
X_features = np.concatenate(features_list, axis=0)
y_labels = np.concatenate(labels_list, axis=0)

# ==========================================
# 2. 执行 t-SNE 降维
# ==========================================
print("Running t-SNE projection (this may take a moment)...")
tsne = TSNE(
    n_components=2, 
    perplexity=30, 
    max_iter=1000, 
    random_state=42, 
    n_jobs=-1
)
X_embedded = tsne.fit_transform(X_features)

# ==========================================
# 3. 绘制 t-SNE 散点图
# ==========================================
# 安全获取类别名称
try:
    class_list = dynamic_class_names
except NameError:
    try:
        class_list = class_names
    except NameError:
        class_list = [f"Class {i}" for i in range(len(np.unique(y_labels)))]

plt.figure(figsize=(12, 10))
sns.set_theme(style="whitegrid")

# 根据真实类别绘制不同颜色的散点图
scatter = sns.scatterplot(
    x=X_embedded[:, 0],
    y=X_embedded[:, 1],
    hue=[class_list[label] for label in y_labels],
    palette="tab10",
    alpha=0.8,
    s=40,
    edgecolor=None
)

# 设置图像样式
plt.title('t-SNE Latent Space Projections of Blood Cells', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('t-SNE Dimension 1', fontsize=12, fontweight='bold')
plt.ylabel('t-SNE Dimension 2', fontsize=12, fontweight='bold')

# 整齐设置图例
plt.legend(
    title='Blood Cell Classes', 
    bbox_to_anchor=(1.05, 1), 
    loc='upper left', 
    fontsize=11, 
    title_fontsize=12,
    frameon=True, 
    shadow=True
)

plt.tight_layout()
plt.show()

# ==========================================
# 1. 查找置信度最高的错误预测
# ==========================================
model.eval()

all_images = []
all_true = []
all_pred = []
all_conf = []

print("Scanning test set for model errors...")

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        
        logits = model(images)
        probs = F.softmax(logits, dim=1)
        confidences, preds = torch.max(probs, dim=1)
        
        # 将数据移到 CPU 保存
        for i in range(images.size(0)):
            if preds[i].item() != labels[i].item():  # 只保存错误预测
                all_images.append(images[i].cpu())
                all_true.append(labels[i].item())
                all_pred.append(preds[i].item())
                all_conf.append(confidences[i].item())

# 检查是否发现错误预测
if len(all_images) == 0:
    print("Amazing! The model made zero mistakes on the test set.")
else:
    # 按置信度从高到低排序，优先查看最自信的错误
    sorted_indices = np.argsort(all_conf)[::-1]
    
    # 最多选择 16 个高置信度错误（错误少于 16 个时全部选择）
    k = min(16, len(sorted_indices))
    
    # 安全获取类别名称
    try:
        class_list = dynamic_class_names
    except NameError:
        try:
            class_list = class_names
        except NameError:
            class_list = [f"Class {i}" for i in range(8)]

    # 辅助函数：反标准化图片以便绘图
    def denormalize(img_tensor):
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img = img_tensor * std + mean
        img = torch.clamp(img, 0, 1) # 将像素值限制在 0 到 1 之间
        return img.numpy().transpose(1, 2, 0)

    # ==========================================
    # 2. 绘制高置信度错误图片网格
    # ==========================================
    # 动态计算网格尺寸，最多为 4x4
    ncols = 4
    nrows = (k + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4 * nrows))
    axes = np.array(axes).flatten()

    for i in range(k):
        idx = sorted_indices[i]
        img = denormalize(all_images[idx])
        
        true_name = class_list[all_true[idx]]
        pred_name = class_list[all_pred[idx]]
        conf = all_conf[idx] * 100
        
        axes[i].imshow(img)
        axes[i].axis('off')
        
        # 使用红色标题突出高置信度错误
        title_str = f"Pred: {pred_name} ({conf:.1f}%)\nTrue: {true_name}"
        axes[i].set_title(title_str, color='red', fontsize=11, fontweight='bold')

    # 如果错误少于 16 个，关闭多余的子图
    for j in range(k, len(axes)):
        axes[j].axis('off')

    plt.suptitle("模型最自信的错误预测", fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()

    # ==========================================
# 1. 从测试集获取一张图片
# ==========================================
model.eval()
temp_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True)
single_image, true_label = next(iter(temp_loader))
single_image = single_image.to(DEVICE)

# 安全获取类别名称
try:
    class_list = dynamic_class_names
except NameError:
    try:
        class_list = class_names
    except NameError:
        class_list = [f"Class {i}" for i in range(8)]

true_class_name = class_list[true_label.item()]

# ==========================================
# 2. 从第一层卷积提取特征图
# ==========================================
# 在 BloodCellCNN 中，第一层位于 features[0]（第一个卷积模块）中
# 具体的 Conv2d 层是 features[0][0]
try:
    first_conv_layer = model.features[0][0]
except (AttributeError, IndexError):
    raise ValueError("无法自动定位第一层卷积层，请检查模型结构。")

# 直接前向传播通过第一个卷积模块
with torch.no_grad():
    # 将图片通过第一个模块，查看原始激活结果
    # model.features[0] 包含：[Conv2d, BatchNorm2d, ReLU, MaxPool2d]
    # 这里只通过 Conv2d、BatchNorm2d 和 ReLU，以获得清晰的特征图
    x = single_image
    x = model.features[0][0](x) # Conv2d
    x = model.features[0][1](x) # BatchNorm2d
    activation_maps = model.features[0][2](x) # ReLU

# 将激活图移到 CPU 并从计算图中分离
activation_maps = activation_maps.squeeze(0).cpu().numpy() # 形状：(32, H, W)
num_filters = activation_maps.shape[0]

# 辅助函数：反标准化原始输入图片以便显示
def denormalize(img_tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(DEVICE)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(DEVICE)
    img = img_tensor * std + mean
    img = torch.clamp(img, 0, 1)
    return img.cpu().squeeze(0).numpy().transpose(1, 2, 0)

original_img = denormalize(single_image)

# ==========================================
# 3. 绘制原图和特征图
# ==========================================
# 共有 32 个滤波器，创建包含原图的 6 列 x 6 行网格
fig = plt.figure(figsize=(14, 14))

# 首先在第 1 个位置绘制原始图片
ax = plt.subplot(6, 6, 1)
ax.imshow(original_img)
ax.set_title(f"Input Image\n({true_class_name})", fontsize=10, fontweight='bold', color='blue')
ax.axis('off')

# 绘制 32 个滤波器对应的激活图
for i in range(num_filters):
    ax = plt.subplot(6, 6, i + 2)
    # 使用鲜明的颜色映射绘制每个滤波器的二维激活图
    ax.imshow(activation_maps[i], cmap='viridis')
    ax.set_title(f"Filter {i+1}", fontsize=9)
    ax.axis('off')

plt.suptitle("First Layer Convolutional Feature Maps (What the model 'sees')", fontsize=18, fontweight='bold', y=0.95)
plt.tight_layout()
plt.show()

# ==========================================
# 1. 设置 Integrated Gradients
# ==========================================
model.eval()
ig = IntegratedGradients(model)

# 安全获取类别名称
try:
    class_list = dynamic_class_names
except NameError:
    try:
        class_list = class_names
    except NameError:
        class_list = [f"Class {i}" for i in range(8)]

# 辅助函数：反标准化图片以便可视化
def denormalize(img_tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(DEVICE)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(DEVICE)
    img = img_tensor * std + mean
    return torch.clamp(img, 0, 1).cpu().squeeze(0).permute(1, 2, 0).numpy()

# ==========================================
# 2. 获取一批 15 张测试图片
# ==========================================
temp_loader = torch.utils.data.DataLoader(test_dataset, batch_size=15, shuffle=True)
images_tensor, true_labels = next(iter(temp_loader))
images_tensor = images_tensor.to(DEVICE)
true_labels = true_labels.to(DEVICE)

print("正在计算 15 张测试图片的 Integrated Gradients，请稍候……")

# ==========================================
# 3. 设置 3x5 绘图网格
# ==========================================
fig, axes = plt.subplots(3, 5, figsize=(20, 12))
axes = axes.flatten()

for i in range(15):
    single_img = images_tensor[i].unsqueeze(0) # 为 Captum 保留批次维度（1, C, H, W）
    true_lbl = true_labels[i].item()
    
    # 前向传播获取预测结果
    with torch.no_grad():
        logits = model(single_img)
        probs = F.softmax(logits, dim=1)
        conf, pred_lbl = torch.max(probs, dim=1)
        pred_lbl = pred_lbl.item()
        conf = conf.item() * 100

    # 计算当前图片的特征归因
    attributions = ig.attribute(
        single_img, 
        target=pred_lbl, 
        baselines=torch.zeros_like(single_img), 
        n_steps=30 # 降低步数以加快 15 张图片的批量处理
    )
    
    # 将原图和归因结果转换为 NumPy 数组
    orig_np = denormalize(single_img)
    attr_np = attributions.cpu().squeeze(0).permute(1, 2, 0).numpy()
    
    # 对颜色通道的归因取平均，生成清晰的二维热力图
    attr_heatmap = np.mean(np.abs(attr_np), axis=2)
    # 将热力图归一化到 0 到 1，便于显示
    if attr_heatmap.max() > 0:
        attr_heatmap /= attr_heatmap.max()

    # 将原图和热力图叠加，适配网格布局
    axes[i].imshow(orig_np)
    axes[i].imshow(attr_heatmap, cmap='jet', alpha=0.4) # 叠加热力图
    axes[i].axis('off')
    
    # 标题颜色：预测正确为绿色，错误为红色
    is_correct = (pred_lbl == true_lbl)
    title_color = 'green' if is_correct else 'red'
    
    pred_name = class_list[pred_lbl]
    true_name = class_list[true_lbl]
    
    axes[i].set_title(
        f"Pred: {pred_name} ({conf:.1f}%)\nTrue: {true_name}", 
        color=title_color, 
        fontsize=10, 
        fontweight='bold'
    )

plt.suptitle("Integrated Gradients 特征归因（3x5 网格）", fontsize=18, fontweight='bold', y=0.98)
plt.tight_layout()
plt.show()