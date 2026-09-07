# 血细胞图像分类

这是一个基于 PyTorch 的血细胞图像分类项目，使用自定义 CNN 模型识别数据集中的 8 类血细胞。

## 项目结构

```text
bloodcells_project/
├── bloodcells_dataset/       # 原始数据集，每个类别对应一个文件夹
├── data/                     # 预留的数据目录
├── models/                   # 训练后保存的模型权重
├── outputs/                  # 训练曲线和混淆矩阵
├── src/
│   ├── dataset.py            # 数据加载、增强和数据划分
│   ├── model.py              # BloodCellCNN 模型
│   ├── train.py              # 训练流程
│   ├── evaluate.py           # 测试评估
│   └── predict.py            # 单张图片预测
├── config.py                 # 路径和训练参数
├── main.py                   # 程序入口
└── requirements.txt          # Python 依赖
```

## 安装依赖

建议使用项目虚拟环境：

```powershell
.\blood_cell_env\Scripts\python.exe -m pip install -r requirements.txt
```

## 运行训练

```powershell
.\blood_cell_env\Scripts\python.exe main.py
```

程序会自动完成以下工作：

1. 从 `bloodcells_dataset/` 读取图片。
2. 按 80%、10%、10% 划分训练集、验证集和测试集。
3. 训练 `BloodCellCNN` 模型，并使用早停策略。
4. 将最佳权重保存到 `models/best_blood_cell_model.pth`。
5. 将损失曲线、准确率曲线和混淆矩阵保存到 `outputs/`。

## 导出 ONNX

根目录下的 `best_blood_cell_model.pth` 是旧版训练脚本生成的权重时，执行：

```powershell
.\blood_cell_env\Scripts\python.exe export_onnx.py
```

导出的文件位于 `models/best_blood_cell_model.onnx`。

模型输入名称为 `images`，形状为 `N x 3 x 224 x 224`；输出名称为 `logits`，最后一维对应 8 个血细胞类别。导出脚本会打印类别顺序，部署时应使用相同的类别映射。

## 修改训练参数

在 `config.py` 中可以修改：

- `BATCH_SIZE`：批次大小
- `NUM_EPOCHS`：最大训练轮数
- `LEARNING_RATE`：学习率
- `EARLY_STOP_PATIENCE`：早停等待轮数
- `DEVICE`：计算设备，默认优先使用 CUDA

## 数据集格式

`bloodcells_dataset/` 必须使用 `ImageFolder` 格式：

```text
bloodcells_dataset/
├── basophil/
├── eosinophil/
├── erythroblast/
├── ig/
├── lymphocyte/
├── monocyte/
├── neutrophil/
└── platelet/
```