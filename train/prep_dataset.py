"""
数据集 JSON 生成脚本
扫描多视角数据集目录，生成 DiffSynth-Studio LoRA edit 训练所需的 metadata JSON 文件。

用法:
  python prep_dataset.py --dataset datasets/dataA_1 --out datasets/dataA_1/dataA_1.json

输出格式 (每行一条训练数据):
  {
    "image": "10001/top_left_near.png",      ← 目标视角图
    "prompt": "top left near",                ← 视角 prompt
    "edit_image": "10001/center_medium.png"   ← 输入参考图
  }
"""

import json
import os
import sys
from pathlib import Path
from argparse import ArgumentParser
from tqdm import tqdm

IMAGE_FORMATS = (".jpg", ".jpeg", ".png", ".bmp")


def collect_subdirs(dataset_path: str) -> list[str]:
    """收集所有包含图片的子目录路径。"""
    subdirs = []
    for root, dirs, files in os.walk(dataset_path):
        for f in files:
            if f.lower().endswith(IMAGE_FORMATS):
                subdirs.append(root)
                break
    return subdirs


def make_json(dataset_path: str, json_output_path: str):
    """扫描数据集并生成 training metadata JSON。"""
    dataset_base = Path(dataset_path).resolve()
    subdirs = collect_subdirs(str(dataset_base))

    json_content = []
    skipped = 0

    for subdir in tqdm(subdirs, desc="处理样本"):
        images = os.listdir(subdir)

        # 定位 center_medium 参考图
        center_candidates = [
            p for p in images
            if Path(p).stem == "center_medium" and p.lower().endswith(IMAGE_FORMATS)
        ]
        if len(center_candidates) != 1:
            print(f"  跳过 {subdir}: center_medium 不存在或重复")
            skipped += 1
            continue

        center_file = center_candidates[0]
        rel_dir = Path(subdir).relative_to(dataset_base)

        for image in images:
            if not image.lower().endswith(IMAGE_FORMATS):
                continue
            image_name = Path(image).stem
            if image_name == "center_medium":
                continue  # 参考图不作为训练目标

            entry = {
                "image": str(rel_dir / image),
                "prompt": " ".join(image_name.split("_")),
                "edit_image": str(rel_dir / center_file),
            }
            json_content.append(entry)

    with open(json_output_path, "w", encoding="utf-8") as f:
        json.dump(json_content, f, ensure_ascii=False, indent=2)

    print(f"\n完成: {len(json_content)} 条训练数据 → {json_output_path}")
    if skipped:
        print(f"跳过 {skipped} 个无效样本")


if __name__ == "__main__":
    parser = ArgumentParser(description="生成 LoRA edit 训练所需的 metadata JSON")
    parser.add_argument("--dataset", required=True, help="数据集目录路径")
    parser.add_argument("--out", required=True, help="输出 JSON 文件路径")
    args = parser.parse_args()

    make_json(args.dataset, args.out)
