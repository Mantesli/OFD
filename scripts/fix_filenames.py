import os
from pathlib import Path


def rename_files():
    # 修改 scripts/fix_filenames.py 中的这一行
    data_dir = Path(r"E:\work\oilfield-leak-detection-v4\data\sampled\train\leak")

    if not data_dir.exists():
        print(f"❌ 找不到目录: {data_dir}")
        return

    print(f"📂 正在扫描目录: {data_dir}")

    count = 0
    for file_path in data_dir.glob("*"):
        # 获取文件名
        old_name = file_path.name

        # 检查是否包含非ASCII字符（比如中文）
        try:
            old_name.encode('ascii')
        except UnicodeEncodeError:
            # 如果包含中文，或者包含 "副本" 字样
            new_name = old_name

            # 替换常见的 Windows 复制后缀
            replacements = {
                " - 副本": "_copy",
                " - Copy": "_copy",
                " (2)": "_2",
                " (3)": "_3",
                " (4)": "_4",
                " (5)": "_5",
                " (6)": "_6",
                " ": "_"  # 把空格换成下划线
            }

            for old_str, new_str in replacements.items():
                new_name = new_name.replace(old_str, new_str)

            # 如果还是有中文（比如乱码），强制改名
            try:
                new_name.encode('ascii')
            except UnicodeEncodeError:
                # 强制重命名为 safe_xxx.jpg
                suffix = file_path.suffix
                new_name = f"aug_{count:04d}{suffix}"

            # 执行重命名
            if new_name != old_name:
                try:
                    new_path = file_path.parent / new_name
                    os.rename(file_path, new_path)
                    print(f"✅ 重命名: {old_name} -> {new_name}")
                    count += 1
                except Exception as e:
                    print(f"❌ 重命名失败 {old_name}: {e}")

    print(f"\n🎉 完成！共修复了 {count} 个文件名。")


if __name__ == "__main__":
    rename_files()