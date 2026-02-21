# 🛢️ Oilfield Leak Detection System

基于多模态融合（RGB + 红外）的油田泄漏检测系统，采用 MobileCLIP 特征提取与弱监督学习策略。

## 📋 项目概述

本项目旨在解决冬季油田巡检中的泄漏点自动检测问题，核心挑战包括：
- **干扰源多**：裸露黑土、枯草、车辙印、抽油平台等在红外/RGB下与泄漏相似
- **样本不平衡**：泄漏样本稀少（约200张泄漏 vs 2800张正常）
- **需要高召回率**：漏检比误检更严重

## 🎯 技术路线

```
┌─────────────────────────────────────────────────────────────┐
│                    多模态融合检测Pipeline                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   RGB图像 ──┬──→ MobileCLIP ──→ 视觉特征                    │
│             │                      │                        │
│             │                      ▼                        │
│             │              ┌──────────────┐                 │
│   红外图像 ─┴──→ 温度纹理 ──→│  特征融合    │──→ 泄漏概率    │
│                  特征提取    │  + 分类器   │                 │
│                             └──────────────┘                │
│                                    │                        │
│   文本Prompt ───→ 语义权重 ────────┘                        │
│   (oil spill vs dark soil)                                 │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 📁 项目结构

```
oilfield-leak-detection/
├── README.md                 # 项目说明
├── requirements.txt          # 依赖包
├── setup.py                  # 安装配置
├── configs/
│   └── default.yaml          # 默认配置文件
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── video_sampler.py      # 模块1: 视频抽帧采样
│   │   ├── frame_splitter.py     # RGB-IR分离
│   │   ├── dataset.py            # PyTorch数据集类
│   │   └── augmentation.py       # 数据增强
│   ├── models/
│   │   ├── __init__.py
│   │   ├── mobileclip_extractor.py  # 模块2: MobileCLIP特征提取
│   │   ├── fusion_classifier.py     # 多模态融合分类器
│   │   └── ema_smoother.py          # 时序EMA平滑
│   ├── features/
│   │   ├── __init__.py
│   │   └── thermal_texture.py       # 模块3: 红外温度纹理特征
│   └── utils/
│       ├── __init__.py
│       ├── metrics.py            # 评估指标（召回率优先）
│       ├── visualization.py      # 可视化工具
│       └── registry.py           # 配准工具
├── scripts/
│   ├── 01_sample_frames.py       # 步骤1: 数据采样
│   ├── 02_validate_clip.py       # 步骤2: 验证CLIP区分能力
│   ├── 03_extract_features.py    # 步骤3: 特征提取
│   ├── 04_train_classifier.py    # 步骤4: 训练分类器
│   └── 05_evaluate.py            # 步骤5: 评估
├── notebooks/
│   └── exploration.ipynb         # 数据探索notebook
├── tests/
│   └── test_pipeline.py          # 单元测试
└── docs/
    └── experiment_log.md         # 实验记录
```

## 🚀 快速开始

### 1. 环境安装

```bash
# 克隆项目
git clone https://github.com/your-username/oilfield-leak-detection.git
cd oilfield-leak-detection

# 创建虚拟环境
conda create -n oilfield python=3.10
conda activate oilfield

# 安装依赖
pip install -r requirements.txt
```

### 2. 数据准备

```bash
# 从视频中抽帧采样
python scripts/01_sample_frames.py \
    --video_dir /path/to/videos \
    --output_dir ./data/sampled \
    --num_leak 160 \
    --num_normal 1600

# RGB-IR分离（如果是拼接视频）
python scripts/01_sample_frames.py --split_mode horizontal
```

### 3. 验证CLIP区分能力

```bash
# 零样本测试CLIP对泄漏vs黑土的区分
python scripts/02_validate_clip.py \
    --data_dir ./data/sampled \
    --output_dir ./results/clip_validation
```

### 4. 特征提取与训练

```bash
# 提取多模态特征
python scripts/03_extract_features.py \
    --data_dir ./data/sampled \
    --output_dir ./data/features

# 训练分类器
python scripts/04_train_classifier.py \
    --feature_dir ./data/features \
    --output_dir ./checkpoints
```

## ⚙️ 配置说明

主要配置项在 `configs/default.yaml`:

```yaml
data:
  image_size: [384, 384]
  split_mode: horizontal  # RGB-IR拼接方式: horizontal/vertical
  
model:
  clip_model: "mobileclip_s0"  # MobileCLIP变体
  fusion_method: "concat"       # 特征融合方式
  
training:
  batch_size: 16
  learning_rate: 1e-4
  focal_loss_gamma: 2.0        # Focal Loss参数
  pos_weight: 10.0             # 正样本权重（应对不平衡）
  
inference:
  threshold: 0.3               # 决策阈值（偏低以提高召回）
  ema_alpha: 0.7               # EMA平滑系数
```

## 📊 评估指标

本项目以**召回率优先**，同时关注准确率：

| 指标 | 目标 | 说明 |
|------|------|------|
| Recall | >85% | 漏检率 <15%（核心指标） |
| Accuracy | >80% | 整体准确率 |
| Precision | >60% | 可接受一定误报 |
| F1-Score | >70% | 平衡指标 |

## 🔬 核心算法

### 泄漏概率计算

$$P_{leak}(x,y) = A_{rgb}(x,y) \cdot A_{ir}(x,y) \cdot W(x,y)$$

其中：
- $A_{rgb}$: RGB异常置信度（MobileCLIP特征）
- $A_{ir}$: 红外异常置信度（温度纹理特征）
- $W$: 语义权重（CLIP文本引导）

### 干扰源区分策略

| 干扰类型 | RGB | 红外 | 区分方法 |
|----------|-----|------|----------|
| 裸露黑土 | 深色 | 偏热(分散) | 温度方差高、道路附近 |
| 枯草 | 黄褐 | 偏冷 | RGB+IR双重不匹配 |
| 车辙印 | 条状 | 零星热点 | 形状特征 |
| 抽油平台 | 金属 | 偏热 | 规则形状、语义过滤 |

## 📝 实验记录

详见 [docs/experiment_log.md](docs/experiment_log.md)

## 🤝 贡献指南

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 📄 License

MIT License - 详见 [LICENSE](LICENSE)

## 📮 联系方式

如有问题，请提交 Issue 或联系项目维护者。
