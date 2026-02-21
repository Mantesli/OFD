# 油田泄漏检测 - 标注规范

## 1. 概述

本文档定义了油田泄漏检测项目的数据标注规范。

**核心原则**：仅标注IR（红外）图像，RGB图像自动从拼接图像中提取。

## 2. 准备工作

### 2.1 图像预处理

对于DJI Matrice 300 RTK的左右拼接图像：

```
原始图像: [IR区域 | RGB区域]
         左半部分  右半部分

处理方式:
1. 裁剪左半部分作为IR图像进行标注
2. 保留原始文件名用于后续匹配
```

**推荐的预处理脚本**：

```python
import cv2
from pathlib import Path

def split_and_save(image_path, output_dir):
    """分割双模态图像并保存IR部分"""
    img = cv2.imread(str(image_path))
    h, w = img.shape[:2]
    
    # 左半部分是IR
    ir_img = img[:, :w//2]
    
    # 保存
    output_path = Path(output_dir) / f"{Path(image_path).stem}_ir.jpg"
    cv2.imwrite(str(output_path), ir_img)
    
    return output_path
```

### 2.2 标注工具

推荐使用 **LabelMe**：

```bash
pip install labelme
labelme  # 启动GUI
```

## 3. 标注类别定义

### 3.1 泄漏类别

| 类别名 | 描述 | 典型形态 | IR特征 |
|--------|------|----------|--------|
| `leak` | 液池扩散型泄漏 | 扇形/舌状 | 中心高温，边缘突然降温 |
| `leak_pipeline` | 管道穿孔型泄漏 | 线性/带状 | 沿管道方向的高温带 |

### 3.2 干扰类别

| 类别名 | 描述 | 典型形态 | IR特征 |
|--------|------|----------|--------|
| `soil` | 黑色裸露土壤 | 随机块状 | 整体较均匀的高温 |
| `equipment` | 油井/泵站设备 | 规则形状（圆/矩形） | 高温，形状规整 |
| `tire_track` | 轮胎痕迹/车辙 | 细长线性 | 高温，非常细长 |
| `other` | 其他热异常 | 不确定 | 用于无法确定的区域 |

### 3.3 视觉示例

```
泄漏 (leak):
    ╭───────╮
   ╱  源点   ╲      特点：
  ╱    ●      ╲     - 有明确的热源中心
 ╱   (高温)    ╲    - 向外扩散
╱_______________╲   - 边缘温度骤降
     (低温)

管道泄漏 (leak_pipeline):
════════●════════    特点：
   (沿管道高温带)     - 线性分布
                     - 可能有多个热点

土壤 (soil):
┌─────────────┐      特点：
│  ～～～～～  │     - 温度均匀
│  ～～～～～  │     - 形状不规则但无方向性
│  ～～～～～  │     - 无中心-边缘差异
└─────────────┘

设备 (equipment):
    ┌───────┐        特点：
    │   ●   │        - 形状规则（圆/方）
    │  热源  │        - 固定位置
    └───────┘        - 持续高温
```

## 4. 标注流程

### 4.1 使用LabelMe标注

1. **打开图像**
   ```
   File → Open Dir → 选择IR图像文件夹
   ```

2. **创建多边形标注**
   ```
   右键 → Create Polygon
   沿着热异常区域边界点击创建多边形
   ```

3. **选择类别**
   ```
   弹出对话框中输入类别名：
   leak / leak_pipeline / soil / equipment / tire_track / other
   ```

4. **保存**
   ```
   File → Save (自动保存为同名.json文件)
   ```

### 4.2 标注质量检查清单

- [ ] 多边形完全包围热异常区域
- [ ] 边界尽量贴合实际轮廓
- [ ] 类别选择正确
- [ ] 没有遗漏明显的热异常区域
- [ ] 对于不确定的区域使用 `other` 类别

## 5. 标注示例

### 5.1 LabelMe JSON格式

```json
{
  "version": "5.0.1",
  "flags": {},
  "shapes": [
    {
      "label": "leak",
      "points": [
        [120.5, 80.2],
        [150.3, 75.8],
        [180.1, 90.5],
        [175.2, 130.3],
        [140.8, 145.6],
        [110.2, 120.4]
      ],
      "shape_type": "polygon"
    },
    {
      "label": "soil",
      "points": [
        [250.0, 50.0],
        [300.0, 55.0],
        [310.0, 100.0],
        [245.0, 95.0]
      ],
      "shape_type": "polygon"
    }
  ],
  "imagePath": "001_ir.jpg",
  "imageHeight": 480,
  "imageWidth": 640
}
```

### 5.2 目录结构

```
data/
├── images/
│   ├── 001_ir.jpg
│   ├── 002_ir.jpg
│   └── ...
├── annotations/
│   ├── 001_ir.json
│   ├── 002_ir.json
│   └── ...
└── original/           # 可选：保留原始拼接图像
    ├── 001.jpg
    └── ...
```

## 6. 判别标准

### 6.1 如何区分泄漏和土壤？

| 特征 | 泄漏 | 土壤 |
|------|------|------|
| 温度分布 | 中心高、边缘低 | 整体均匀 |
| 边缘 | 突然降温（明显边界） | 渐变或无边界 |
| 形状 | 扇形/舌状（有方向） | 随机块状 |
| 位置 | 通常靠近管道 | 随机分布 |

**决策流程**：
```
该区域是否有明显的中心-边缘温差？
├─ 是 → 是否呈扇形/舌状扩散？
│       ├─ 是 → 标注为 leak
│       └─ 否（线性）→ 标注为 leak_pipeline
└─ 否（均匀温度）→ 标注为 soil
```

### 6.2 如何区分管道泄漏和轮胎痕迹？

| 特征 | 管道泄漏 | 轮胎痕迹 |
|------|----------|----------|
| 宽度 | 较宽（管道直径） | 很窄（轮胎宽度） |
| 温度 | 持续高温 | 可能间断 |
| 位置 | 沿已知管道走向 | 沿道路/车辙 |
| 形状 | 可能有热点/破口 | 均匀细长 |

### 6.3 不确定时怎么办？

1. 如果**略微不确定**：根据最可能的类别标注
2. 如果**非常不确定**：标注为 `other`
3. 如果**需要讨论**：在标注备注中说明

## 7. 常见问题

### Q1: RGB图像需要标注吗？

**不需要**。仅标注IR图像。RGB用于CLIP特征提取（自动处理）。

### Q2: 图像中没有泄漏怎么办？

仍然标注所有可见的热异常区域（soil, equipment等）。这些是重要的负样本。

### Q3: 多个泄漏区域怎么标注？

每个区域单独标注，都使用 `leak` 类别。

### Q4: 边界不清晰怎么办？

尽量标注可见的范围，不需要过分精确。模型会学习适应一定的标注噪声。

### Q5: 需要标注多少样本？

**建议**：
- 泄漏样本：100-200张（你现有的）
- 干扰样本：按相同比例标注 soil, equipment 等
- 总计：300-500张足够开始训练

## 8. 验证标注

使用以下脚本验证标注质量：

```python
from pathlib import Path
from src.features.region_analyzer import AnnotationLoader

def validate_annotations(annotation_dir):
    """验证标注文件"""
    ann_dir = Path(annotation_dir)
    
    stats = {'leak': 0, 'leak_pipeline': 0, 'soil': 0, 
             'equipment': 0, 'tire_track': 0, 'other': 0}
    
    for json_file in ann_dir.glob("*.json"):
        annotations = AnnotationLoader.load_labelme(str(json_file))
        for ann in annotations:
            label = ann['label']
            if label in stats:
                stats[label] += 1
            else:
                print(f"⚠️ 未知类别: {label} in {json_file.name}")
    
    print("标注统计:")
    for label, count in stats.items():
        print(f"  {label}: {count}")
    
    return stats

# 使用
validate_annotations("./data/annotations")
```

## 9. 下一步

标注完成后，运行分析脚本验证：

```bash
python scripts/06_analyze_thermal.py \
    --image_dir ./data/images \
    --annotation_dir ./data/annotations \
    --output_dir ./results/validation
```

查看输出结果，确认标注与模型预测的一致性。
