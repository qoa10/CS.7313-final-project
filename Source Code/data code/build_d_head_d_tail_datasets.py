#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Build YOLO datasets for D-Head and D-Tail from instance lists.

Inputs (under ROOT):
    - yolo/animal_lt.yaml
    - yolo/d_head_instances.txt
    - yolo/d_tail_instances.txt

Outputs:
    - images/train_d_head/, labels/train_d_head/
    - images/train_d_tail/, labels/train_d_tail/
    - yolo/animal_lt_d_head.yaml
    - yolo/animal_lt_d_tail.yaml
"""

from pathlib import Path
import shutil
import yaml
from collections import defaultdict

# ================== CONFIG ================== #

ROOT = Path(r"F:\Tao\longtail\dataset")

DATA_YAML = ROOT / "yolo" / "animal_lt.yaml"

# 🔴 updated: instance lists are inside yolo/
D_HEAD_TXT = ROOT /  "d_head_instances.txt"
D_TAIL_TXT = ROOT / "d_tail_instances.txt"

# output dirs
IMAGES_D_HEAD = ROOT / "images" / "train_d_head"
LABELS_D_HEAD = ROOT / "labels" / "train_d_head"

IMAGES_D_TAIL = ROOT / "images" / "train_d_tail"
LABELS_D_TAIL = ROOT / "labels" / "train_d_tail"

YAML_D_HEAD = ROOT / "yolo" / "animal_lt_d_head.yaml"
YAML_D_TAIL = ROOT / "yolo" / "animal_lt_d_tail.yaml"


# ================== HELPERS ================== #

def load_yaml(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_instance_file(txt_path: Path):
    """
    Parse d_head_instances.txt or d_tail_instances.txt

    Returns:
        dict: image_rel_path (str) -> list of (cls_id, xc, yc, w, h)
    """
    inst_per_img = defaultdict(list)

    with txt_path.open("r", encoding="utf-8") as f:
        header = f.readline()  # skip header line
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 8:
                raise ValueError(f"Unexpected line format in {txt_path}: {line}")

            image_rel, cls_id_str, cls_name, conf_str, xc_str, yc_str, w_str, h_str = parts

            cls_id = int(cls_id_str)
            xc = float(xc_str)
            yc = float(yc_str)
            w = float(w_str)
            h = float(h_str)

            inst_per_img[image_rel].append((cls_id, xc, yc, w, h))

    return inst_per_img


def build_subset(inst_per_img, images_out: Path, labels_out: Path, subset_name: str):
    """
    Create YOLO images/labels dirs for a subset (D-Head or D-Tail).
    """
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    n_imgs = 0
    n_boxes = 0

    for image_rel, boxes in inst_per_img.items():
        src_img = ROOT / image_rel
        if not src_img.exists():
            print(f"[WARN] Image not found: {src_img}, skip.")
            continue

        # Flatten directory structure: only keep file name.
        dst_img = images_out / src_img.name
        dst_lbl = labels_out / (src_img.stem + ".txt")

        shutil.copy2(src_img, dst_img)

        with dst_lbl.open("w", encoding="utf-8") as f:
            for cls_id, xc, yc, w, h in boxes:
                f.write(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

        n_imgs += 1
        n_boxes += len(boxes)

    print(f"[INFO] Built subset '{subset_name}': {n_imgs} images, {n_boxes} boxes.")
    return n_imgs, n_boxes


def write_subset_yaml(base_cfg, yaml_out: Path, subset_images_dir: Path):
    """
    Write a new data.yaml for subset training.

    We keep:
        path: ROOT
        train: images/<subset_dir_name>
        val:   same as original (base_cfg['val'])
        names: same as original
    """
    cfg = dict(base_cfg)  # shallow copy

    # ensure path is ROOT
    cfg["path"] = str(ROOT)

    # set train to relative path from ROOT
    cfg["train"] = str(subset_images_dir.relative_to(ROOT))

    # val stays the same as in original yaml
    yaml_out.parent.mkdir(parents=True, exist_ok=True)
    with yaml_out.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

    print(f"[INFO] Wrote YAML: {yaml_out}")


# ================== MAIN ================== #

def main():
    if not DATA_YAML.exists():
        raise FileNotFoundError(f"Base yaml not found: {DATA_YAML}")
    if not D_HEAD_TXT.exists():
        raise FileNotFoundError(f"D-Head instances file not found: {D_HEAD_TXT}")
    if not D_TAIL_TXT.exists():
        raise FileNotFoundError(f"D-Tail instances file not found: {D_TAIL_TXT}")

    base_cfg = load_yaml(DATA_YAML)

    # ----- D-Head -----
    print("[INFO] Parsing D-Head instance list...")
    d_head_map = parse_instance_file(D_HEAD_TXT)
    build_subset(d_head_map, IMAGES_D_HEAD, LABELS_D_HEAD, "D-Head")
    write_subset_yaml(base_cfg, YAML_D_HEAD, IMAGES_D_HEAD)

    # ----- D-Tail -----
    print("[INFO] Parsing D-Tail instance list...")
    d_tail_map = parse_instance_file(D_TAIL_TXT)
    build_subset(d_tail_map, IMAGES_D_TAIL, LABELS_D_TAIL, "D-Tail")
    write_subset_yaml(base_cfg, YAML_D_TAIL, IMAGES_D_TAIL)

    print("[DONE] D-Head and D-Tail datasets + YAMLs are ready.")


if __name__ == "__main__":
    main()
