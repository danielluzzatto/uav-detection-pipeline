"""
This file gives an idea of the starting dataset, divided into train, validation and test. 
"""
import os
import argparse
from pathlib import Path
from collections import defaultdict

def analyze_split(split_dir: Path, class_names: list[str]):
    img_dir = split_dir / "images"
    lbl_dir = split_dir / "labels"

    if not img_dir.exists():
        print(f"  [skip] {split_dir} — images/ not found")
        return

    image_files = list(img_dir.glob("*.*"))
    label_files = list(lbl_dir.glob("*.txt")) if lbl_dir.exists() else []

    images_with_label = 0
    images_without_label = 0
    images_empty_label = 0   # label file exists but is empty (background-only)
    class_counts = defaultdict(int)
    instance_counts = defaultdict(int)
    polygon_vertex_counts = []  # to distinguish seg vs bbox labels

    for img_path in image_files:
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        if not lbl_path.exists():
            images_without_label += 1
            continue

        lines = [l.strip() for l in lbl_path.read_text().splitlines() if l.strip()]
        if not lines:
            images_empty_label += 1
            continue

        images_with_label += 1
        for line in lines:
            parts = line.split()
            cls = int(parts[0])
            class_counts[cls] += 1
            instance_counts[cls] += 1
            coords = parts[1:]
            # bbox: 4 values; polygon: >4 values (pairs of x,y)
            polygon_vertex_counts.append(len(coords) // 2)

    total = len(image_files)
    print(f"\n- Total images     : {total}")
    print(f"- With annotations : {images_with_label}")
    print(f"- Empty label file : {images_empty_label}  ← background/negative samples")
    print(f"- No label file    : {images_without_label}")

    if polygon_vertex_counts:
        avg_verts = sum(polygon_vertex_counts) / len(polygon_vertex_counts)
        label_type = "SEGMENTATION polygon" if avg_verts > 4 else "BOUNDING BOX"
        print(f"- Label type       : {label_type} (avg {avg_verts:.1f} coords per instance)")

    print(f"\n- Class breakdown:")
    for cls_id, count in sorted(class_counts.items()):
        name = class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}"
        print(f"- [{cls_id}] {name:30s} : {count} instances")

    if class_counts:
        total_instances = sum(class_counts.values())
        print(f"- Total instances  : {total_instances}")
        imgs_with_objects = images_with_label
        print(f"- Avg instances/img: {total_instances / max(imgs_with_objects, 1):.2f}")




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True, help="Path to YOLOv11 dataset root")
    args = parser.parse_args()

    root = Path(args.data_root)
    class_names = ['none', 'uav']
    
    print(f"Classes from data.yaml: {class_names}")

    for split in ["train", "valid", "test"]:
        split_dir = root / split
        if split_dir.exists():
            print(f"## Split: {split.upper()}")
            analyze_split(split_dir, class_names)

if __name__ == "__main__":
    main()