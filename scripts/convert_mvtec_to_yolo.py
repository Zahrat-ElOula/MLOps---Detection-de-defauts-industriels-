#!/usr/bin/env python
from __future__ import annotations

import random
import shutil
from pathlib import Path

import cv2


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    import shutil

    if Path("data/yolo_v2").exists():
        shutil.rmtree("data/yolo_v2", ignore_errors=True)
    raw_root = project_root / "data" / "raw"
    yolo_root = project_root / "data" / "yolo_v2"
    images_train = yolo_root / "images" / "train"
    images_val = yolo_root / "images" / "val"
    labels_train = yolo_root / "labels" / "train"
    labels_val = yolo_root / "labels" / "val"
    for d in [images_train, images_val, labels_train, labels_val]:
        d.mkdir(parents=True, exist_ok=True)

    samples = []
    if raw_root.exists():
        for category_dir in raw_root.iterdir():
            if not category_dir.is_dir():
                continue
            gt_root = category_dir / "ground_truth"
            test_root = category_dir / "test"
            if not gt_root.exists():
                continue
            for defect_dir in gt_root.iterdir():
                if not defect_dir.is_dir():
                    continue
                for mask_path in defect_dir.glob("*.png"):
                    img_name = mask_path.name.replace("_mask.png", ".png")
                    img_path = test_root / defect_dir.name / img_name
                    if not img_path.exists():
                        continue
                    samples.append((img_path, mask_path))

    random.seed(42)
    random.shuffle(samples)
    split_idx = int(len(samples) * 0.8)
    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:]

    def convert_pair(img_path: Path, mask_path: Path, img_out_dir: Path, lbl_out_dir: Path) -> None:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return
        img = cv2.imread(str(img_path))
        if img is None:
            return
        h, w = mask.shape[:2]
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return
        for c in contours:
            x, y, bw, bh = cv2.boundingRect(c)
            if bw == 0 or bh == 0:
                continue
            xc = (x + bw / 2.0) / w
            yc = (y + bh / 2.0) / h
            nw = bw / w
            nh = bh / h
            base = f"{img_path.parent.parent.name}_{img_path.parent.name}_{img_path.stem}"
            dst = img_out_dir / f"{base}.png"
            shutil.copyfile(img_path, dst)
            label_path = lbl_out_dir / f"{base}.txt"
            with label_path.open("w", encoding="utf-8") as f:
                f.write(f"0 {xc} {yc} {nw} {nh}\n")

    for img_path, mask_path in train_samples:
        convert_pair(img_path, mask_path, images_train, labels_train)
    for img_path, mask_path in val_samples:
        convert_pair(img_path, mask_path, images_val, labels_val)

    data_yaml = yolo_root / "data.yaml"
    data_yaml.write_text(
        "path: ../data/yolo_v2\ntrain: images/train\nval: images/val\nnames:\n  0: 'defect'\n",
        encoding="utf-8",
    )
    print(f"Converted samples: {len(samples)} | train: {len(train_samples)} | val: {len(val_samples)}")

    cat_count = {}
    for img_path, _ in samples:
        cat = img_path.parent.parent.name
        cat_count[cat] = cat_count.get(cat, 0) + 1
    print("Images by category:", cat_count)

    defect_count = {}
    for img_path, _ in samples:
        d = img_path.parent.name
        defect_count[d] = defect_count.get(d, 0) + 1
    print("Images by defect type:", defect_count)


if __name__ == "__main__":
    main()
