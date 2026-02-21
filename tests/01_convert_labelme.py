# scripts/01_convert_labelme.py
import json
import shutil
import random
import os
from pathlib import Path
from tqdm import tqdm


def main():
    # 配置路径（根据您的描述）
    DATA_ROOT = Path("./data")
    ORIGINAL_DIR = DATA_ROOT / "original"  # 原始双模态拼接图
    ANNOTATION_DIR = DATA_ROOT / "annotations"  # LabelMe JSON文件
    OUTPUT_DIR = DATA_ROOT / "sampled"  # 输出的标准数据集目录

    # 划分比例
    SPLIT_RATIO = {"train": 0.7, "val": 0.15, "test": 0.15}

    # 确保输出目录存在
    for split in ["train", "val", "test"]:
        for label in ["leak", "normal"]:
            (OUTPUT_DIR / split / label).mkdir(parents=True, exist_ok=True)

    # 1. 扫描所有标注文件并建立映射
    print("正在扫描标注文件...")
    dataset_items = []  # 存储 (image_path, label)

    # 获取所有 json 文件
    json_files = list(ANNOTATION_DIR.glob("*.json"))

    for json_file in tqdm(json_files):
        try:
            # 解析 JSON 找标签
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            is_leak = False
            # 检查 shapes 中是否有 leak 标签
            for shape in data.get("shapes", []):
                label_name = shape.get("label", "").lower()
                if label_name in ["leak", "leak_pipeline"]:
                    is_leak = True
                    break

            # 2. 找到对应的原始拼接图
            # 假设标注文件名为 xxx_ir.json，对应原始图为 xxx.jpg
            base_name = json_file.stem
            if base_name.endswith("_ir"):
                original_name = base_name[:-3]  # 去掉 _ir
            else:
                original_name = base_name

            # 在 original 目录查找对应的图片（支持 jpg/png）
            image_path = None
            for ext in [".jpg", ".jpeg", ".png"]:
                probe_path = ORIGINAL_DIR / (original_name + ext)
                if probe_path.exists():
                    image_path = probe_path
                    break

            if image_path:
                dataset_items.append({
                    "src": image_path,
                    "label": "leak" if is_leak else "normal",
                    "json": json_file.name
                })
            else:
                print(f"⚠️ 警告: 找不到对应的原始图片: {original_name}")

        except Exception as e:
            print(f"❌ 错误处理 {json_file}: {e}")

    # 3. 随机打乱并划分数据集
    random.seed(42)
    random.shuffle(dataset_items)

    # 分离泄漏和正常样本分别进行划分（保证验证集也有泄漏样本）
    leaks = [x for x in dataset_items if x["label"] == "leak"]
    normals = [x for x in dataset_items if x["label"] == "normal"]

    print(f"\n统计: 泄漏样本 {len(leaks)} 张, 正常样本 {len(normals)} 张")

    def split_and_copy(items, name_prefix):
        n = len(items)
        n_train = int(n * SPLIT_RATIO["train"])
        n_val = int(n * SPLIT_RATIO["val"])

        splits = {
            "train": items[:n_train],
            "val": items[n_train:n_train + n_val],
            "test": items[n_train + n_val:]
        }

        for split_name, split_items in splits.items():
            for item in split_items:
                dest_dir = OUTPUT_DIR / split_name / item["label"]
                shutil.copy2(item["src"], dest_dir / item["src"].name)

    split_and_copy(leaks, "leak")
    split_and_copy(normals, "normal")

    print(f"\n✅ 数据集构建完成！输出目录: {OUTPUT_DIR}")
    print(f"下一步请运行: python scripts/03_extract_features.py --data_dir {OUTPUT_DIR}")


if __name__ == "__main__":
    main()