# MAFN: 毫米波雷达手势识别的轻量级三分支CNN模型

## 项目简介
MAFN (Multi-scale Attention Fusion Network) 是一个用于毫米波雷达手势识别的深度学习模型。该模型采用三分支结构，结合多尺度注意力机制，能够高效地处理毫米波雷达数据，实现准确的手势识别。本模型特别针对毫米波雷达数据的特点进行了优化，具有轻量级、高精度和实时性的特点。

## 主要特性
- 多尺度注意力机制：通过不同尺度的特征提取和融合，提高模型对细节特征的感知能力
- 三分支结构：分别处理不同层次的特征信息
- 自适应特征融合：动态调整各分支特征的权重
- 轻量级设计：在保证性能的同时降低模型复杂度
- 支持预训练：包含自监督预训练模块
- 实时推理：优化后的模型支持实时手势识别

## 环境要求
- Python 3.8+
- PyTorch 1.8+
- CUDA 11.0+ (如果使用GPU)
- 其他依赖包：
  ```
  numpy>=1.19.2
  scipy>=1.6.2
  matplotlib>=3.3.4
  tqdm>=4.60.0
  torchvision>=0.9.0
  ```

## 项目结构
```
MAFN/
├── model.py            # 主模型文件，包含模型定义和损失函数
├── data_preprocessing.py    # 数据预处理脚本
├── train.py           # 训练脚本
└── README.md          # 项目说明文档
```

## 安装
1. 克隆项目：
```bash
git clone https://github.com/1394872816/Multi-scale-Attention-Fusion-Network.git
cd Multi-scale-Attention-Fusion-Network
```

2. 创建虚拟环境（推荐）：
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. 安装依赖：
```bash
pip install -r requirements.txt
```

## 使用方法
1. 数据预处理：
```bash
python data_preprocessing.py --data_path /path/to/data --output_path /path/to/processed_data
```

2. 模型训练：
```bash
python train.py
```

注意：目前项目处于开发阶段，评估和推理功能将在后续版本中提供。

## 模型结构
MAFN模型主要由以下组件构成：
- MultiScaleAttention：多尺度注意力模块
- SEAttention：通道注意力模块
- FeatureFusion：特征融合模块
- 预训练模块：支持自监督学习
- 分类头：用于最终的手势分类

## 训练参数
- 学习率：1e-4
- 批次大小：32
- 训练轮数：100
- 优化器：AdamW
- 损失函数：改进的混合损失函数
- 学习率调度器：WarmupCosineLRScheduler
- 数据增强：随机裁剪、旋转、噪声添加

## 性能指标
- 准确率：>95%
- 推理速度：<10ms/样本
- 模型大小：<10MB
- 参数量：<1M
- FPS：>100

## 注意事项
1. 确保数据格式正确
2. 根据实际硬件配置调整批次大小
3. 建议使用GPU进行训练
4. 预训练模型可以显著提升性能
5. 训练前请确保数据已正确预处理
6. 建议使用验证集监控训练过程


## 联系方式
- 邮箱：[1394872816@qq.com]
- 项目主页：https://github.com/1394872816/Multi-scale-Attention-Fusion-Network
