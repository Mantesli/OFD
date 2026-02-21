import json
import shutil
import random
import os
from pathlib import Path
from tqdm import tqdm


def main():
    # 1. 定义路径
    # 假设脚本在 scripts/ 下，向上两级是项目根目录
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    DATA_ROOT = PROJECT_ROOT / "data"

    ANNOTATIONS_DIR = DATA_ROOT / "annotations"
    ORIGINAL_DIR = DATA_ROOT / "original"
    OUTPUT_DIR = DATA_ROOT / "sampled"

    print(f"📂 项目根目录: {PROJECT_ROOT}")
    print(f"📂 标注目录: {ANNOTATIONS_DIR}")
    print(f"📂 原图目录: {ORIGINAL_DIR}")

    # 2. 检查源目录是否存在
    if not ANNOTATIONS_DIR.exists():
        print(f"❌ 错误: 找不到标注目录 {ANNOTATIONS_DIR}")
        return
    if not ORIGINAL_DIR.exists():
        print(f"❌ 错误: 找不到原图目录 {ORIGINAL_DIR}")
        return

    # 3. 【关键步骤】清理旧数据 (Clean Reset)
    if OUTPUT_DIR.exists():
        print(f"🧹 正在清理旧的 sampled 目录: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)

    # 重建目录结构
    for split in ['train', 'val', 'test']:
        for label in ['leak', 'normal']:
            (OUTPUT_DIR / split / label).mkdir(parents=True, exist_ok=True)

    # 4. 扫描并匹配文件
    dataset_items = []
    json_files = list(ANNOTATIONS_DIR.glob("*.json"))

    print(f"🔍 找到 {len(json_files)} 个标注文件，开始解析...")

    for json_file in tqdm(json_files):
        try:
            # --- A. 解析标签 ---
            with open(json_file, 'r', encoding='utf-8') as f:
                content = json.load(f)

            # 只要有一个 shape 的 label 是 leak，就算泄漏样本
            is_leak = False
            for shape in content.get('shapes', []):
                label_name = shape.get('label', '').lower()
                if 'leak' in label_name:
                    is_leak = True
                    break

            label = "leak" if is_leak else "normal"

            # --- B. 寻找对应的原图 ---
            # 逻辑：标注文件名通常是 "001_ir.json"，原图是 "001.jpg"
            # 1. 去掉 "_ir" 后缀
            file_stem = json_file.stem.replace("_ir", "")

            # 2. 尝试匹配不同扩展名的图片
            image_path = None
            for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.bmp']:
                probe_path = ORIGINAL_DIR / (file_stem + ext)
                if probe_path.exists():
                    image_path = probe_path
                    break

            if image_path:
                dataset_items.append({
                    "src_path": image_path,
                    "label": label,
                    "stem": file_stem
                })
            else:
                # 仅在找不到时打印警告（可选）
                # print(f"⚠️ 未找到原图: {file_stem} (JSON: {json_file.name})")
                pass

        except Exception as e:
            print(f"❌ 读取错误 {json_file.name}: {e}")

    # 5. 随机打乱并划分
    if not dataset_items:
        print("❌ 未匹配到任何图片，请检查文件名对应关系！")
        return

    random.seed(42)  # 固定种子保证可复现
    random.shuffle(dataset_items)

    # 分离正负样本，分别划分，保证验证集里一定有泄漏样本
    leaks = [x for x in dataset_items if x['label'] == 'leak']
    normals = [x for x in dataset_items if x['label'] == 'normal']

    print(f"\n📊 统计结果:")
    print(f"   泄漏样本 (Leak): {len(leaks)}")
    print(f"   正常样本 (Normal): {len(normals)}")

    def split_and_copy(items, ratio=[0.7, 0.15, 0.15]):
        n = len(items)
        n_train = int(n * ratio[0])
        n_val = int(n * ratio[1])

        splits = {
            "train": items[:n_train],
            "val": items[n_train:n_train + n_val],
            "test": items[n_train + n_val:]
        }

        for split_name, split_items in splits.items():
            for item in split_items:
                src = item['src_path']
                dst = OUTPUT_DIR / split_name / item['label'] / src.name
                shutil.copy2(src, dst)

    print("\n🚀 正在复制文件...")
    split_and_copy(leaks)
    split_and_copy(normals)

    print(f"\n✅ 数据集重置完成！目录: {OUTPUT_DIR}")
    print("   Train/Leak 数量已恢复为原始真实数量。")


if __name__ == "__main__":
    main()