#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Build D-Head instance list for Step-wise Long-tailed Detection (Dong et al. ICCV 2023).

- Uses a trained YOLOv8 model to assign a confidence score to EACH GT instance
  in the TRAIN set.
- For each class:
    - If class is HEAD (long): keep top N_HEAD instances (by conf)
    - If class is TAIL:        keep top N_TAIL instances (by conf)
- Writes a text file 'd_head_instances.txt' under ROOT, listing the selected instances.

You can later use this txt to build the actual D-Head dataset
(e.g., reconstruct labels & copy images).
"""

from pathlib import Path
import yaml
from ultralytics import YOLO
import numpy as np

# ================== CONFIG (EDIT THIS PART) ================== #

# Paths you provided
ROOT = Path(r"F:\Tao\longtail\dataset")
DATA_YAML = ROOT / r"yolo\animal_lt.yaml"
WEIGHTS = ROOT / r"runs\yolov8s_animal_lt5\weights\best.pt"

# D-Head exemplar numbers (from thesis)
N_HEAD = 100  # long/head class: top 200 instances
N_TAIL = 35   # tail class: top 30 instances

IOU_THRESH = 0.5  # IoU threshold to "match" GT to a prediction

# >>>>>>>>>> IMPORTANT: FILL THESE WITH YOUR CLASS NAMES <<<<<<<<<<
# They must match exactly the "names" in animal_lt.yaml
HEAD_CLASS_NAMES = [
    # example:
     "cat", "dog", "cow", "horse"
]

TAIL_CLASS_NAMES = [
    # example:
    "bear", "bird", "sheep", "zebra"
]

# Output txt file
OUT_TXT = ROOT / "d_tail_instances.txt"

# ================== HELPER FUNCTIONS ================== #


def load_yaml(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_train_images_and_labels(data_cfg, root: Path):
    """
    Resolve train images dir and corresponding labels dir.
    Returns: list of (img_path, lbl_path)
    """
    # Resolve train images directory
    train_entry = Path(str(data_cfg["train"]))
    if not train_entry.is_absolute():
        base = Path(data_cfg.get("path", root))
        train_dir = (base / train_entry).resolve()
    else:
        train_dir = train_entry.resolve()

    if "images" in train_dir.parts:
        parts = list(train_dir.parts)
        new_parts = ["labels" if p == "images" else p for p in parts]
        labels_dir = Path(*new_parts)
    else:
        # fallback: images/train -> labels/train pattern
        labels_dir = train_dir.parent.parent / "labels" / train_dir.name

    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    pairs = []
    for img_path in sorted(train_dir.glob("*")):
        if img_path.suffix.lower() not in img_exts:
            continue
        lbl_path = labels_dir / (img_path.stem + ".txt")
        if not lbl_path.exists():
            # no label file, skip this image
            continue
        pairs.append((img_path, lbl_path))

    print(f"[INFO] Found {len(pairs)} train images with labels.")
    return pairs


def yolo_txt_to_boxes(lbl_path: Path):
    """
    Parse YOLO txt:
      cls x_center y_center w h
    All normalized. Returns list of dicts with:
      { "cls": int, "xyxy": np.array([x1,y1,x2,y2]), "xywh": np.array([xc,yc,w,h]) }
    """
    instances = []
    text = lbl_path.read_text().strip()
    if not text:
        return instances

    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls = int(float(parts[0]))
        xc, yc, w, h = map(float, parts[1:])
        x1 = xc - w / 2.0
        y1 = yc - h / 2.0
        x2 = xc + w / 2.0
        y2 = yc + h / 2.0
        instances.append({
            "cls": cls,
            "xyxy": np.array([x1, y1, x2, y2], dtype=float),
            "xywh": np.array([xc, yc, w, h], dtype=float),
        })
    return instances


def box_iou_xyxy(box1, box2):
    """
    Compute IoU between two boxes in [x1,y1,x2,y2] normalized format.
    box1, box2: np.array of shape (4,)
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = max(0.0, (box1[2] - box1[0])) * max(0.0, (box1[3] - box1[1]))
    area2 = max(0.0, (box2[2] - box2[0])) * max(0.0, (box2[3] - box2[1]))

    union = area1 + area2 - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


# ================== MAIN LOGIC ================== #


def main():
    # ---- Load config & model ----
    data_cfg = load_yaml(DATA_YAML)
    names = data_cfg["names"]
    # names is either list or dict {id: name}
    if isinstance(names, dict):
        id2name = {int(k): v for k, v in names.items()}
    else:
        id2name = {i: v for i, v in enumerate(names)}

    name2id = {v: k for k, v in id2name.items()}

    # Resolve head/tail class IDs from names
    head_ids = []
    tail_ids = []

    for cname in HEAD_CLASS_NAMES:
        if cname not in name2id:
            raise ValueError(f"HEAD class name '{cname}' not found in yaml names: {list(name2id.keys())}")
        head_ids.append(name2id[cname])

    for cname in TAIL_CLASS_NAMES:
        if cname not in name2id:
            raise ValueError(f"TAIL class name '{cname}' not found in yaml names: {list(name2id.keys())}")
        tail_ids.append(name2id[cname])

    print("[INFO] Head class IDs:", head_ids)
    print("[INFO] Tail class IDs:", tail_ids)

    print("[INFO] Loading YOLO model...")
    model = YOLO(str(WEIGHTS))

    # ---- Build list of train image/label pairs ----
    pairs = get_train_images_and_labels(data_cfg, ROOT)

    # ---- For each GT instance, assign a confidence score from model ----
    # We'll store: per-class list of records
    # record = {
    #   "img_path": Path,
    #   "cls": int,
    #   "cls_name": str,
    #   "conf": float,
    #   "xywh": np.array([xc,yc,w,h])
    # }
    per_class_records = {cid: [] for cid in id2name.keys()}

    print("[INFO] Running inference on train images and matching GT to predictions...")

    for img_path, lbl_path in pairs:
        # Load GT boxes
        gt_instances = yolo_txt_to_boxes(lbl_path)
        if not gt_instances:
            continue

        # Run YOLO prediction on this image
        # Very low conf threshold to keep as many candidate boxes as possible
        results = model.predict(
            source=str(img_path),
            imgsz=640,
            conf=0.001,
            iou=0.7,
            verbose=False
        )
        if not results:
            preds_boxes = []
        else:
            r = results[0]
            preds_boxes = []
            if r.boxes is not None and len(r.boxes) > 0:
                # boxes.xywhn: normalized; boxes.xyxy: pixels
                # We'll use normalized
                xywhn = r.boxes.xywhn.cpu().numpy()
                cls_pred = r.boxes.cls.cpu().numpy()
                conf_pred = r.boxes.conf.cpu().numpy()
                for b, c, cf in zip(xywhn, cls_pred, conf_pred):
                    xc, yc, w, h = b.tolist()
                    x1 = xc - w / 2.0
                    y1 = yc - h / 2.0
                    x2 = xc + w / 2.0
                    y2 = yc + h / 2.0
                    preds_boxes.append({
                        "cls": int(c),
                        "xyxy": np.array([x1, y1, x2, y2], dtype=float),
                        "xywh": np.array([xc, yc, w, h], dtype=float),
                        "conf": float(cf),
                    })

        # For each GT instance, find best matching prediction of same class
        for gt in gt_instances:
            gt_cls = gt["cls"]
            gt_xyxy = gt["xyxy"]
            gt_xywh = gt["xywh"]

            best_conf = 0.0
            best_iou = 0.0

            for pred in preds_boxes:
                if pred["cls"] != gt_cls:
                    continue
                iou = box_iou_xyxy(gt_xyxy, pred["xyxy"])
                if iou > best_iou:
                    best_iou = iou
                    best_conf = pred["conf"]

            if best_iou < IOU_THRESH:
                best_conf = 0.0  # treat as missed

            record = {
                "img_path": img_path,
                "cls": gt_cls,
                "cls_name": id2name[gt_cls],
                "conf": best_conf,
                "xywh": gt_xywh,
            }
            per_class_records[gt_cls].append(record)

    # ---- For each class, select top N_HEAD or N_TAIL ----
    print("[INFO] Selecting top instances for D-Head...")

    selected_records = []

    for cid, records in per_class_records.items():
        if not records:
            continue

        is_head = cid in head_ids
        is_tail = cid in tail_ids

        if not (is_head or is_tail):
            # class not used (neither head nor tail) -> skip or treat as tail? Here we skip.
            print(f"[WARN] Class id {cid} ('{id2name[cid]}') is neither in HEAD nor TAIL list. Skipping.")
            continue

        # sort by confidence descending
        records_sorted = sorted(records, key=lambda r: r["conf"], reverse=True)

        if is_head:
            n_keep = min(N_HEAD, len(records_sorted))
        else:  # is_tail
            n_keep = min(N_TAIL, len(records_sorted))

        kept = records_sorted[:n_keep]
        selected_records.extend(kept)

        print(f"[INFO] Class {cid} ('{id2name[cid]}'): "
              f"{len(records_sorted)} instances, keep {n_keep} for D-Head.")

    # ---- Write d_head_instances.txt ----
    print(f"[INFO] Writing selected instances to: {OUT_TXT}")
    with OUT_TXT.open("w", encoding="utf-8") as f:
        header = "image_rel_path\tclass_id\tclass_name\tconf\tx_center\ty_center\tw\th\n"
        f.write(header)
        for rec in selected_records:
            img_rel = rec["img_path"].relative_to(ROOT)
            cid = rec["cls"]
            cname = rec["cls_name"]
            conf = rec["conf"]
            xc, yc, w, h = rec["xywh"].tolist()
            line = f"{img_rel}\t{cid}\t{cname}\t{conf:.6f}\t{xc:.6f}\t{yc:.6f}\t{w:.6f}\t{h:.6f}\n"
            f.write(line)

    print("[DONE] D-Head instance list generated.")


if __name__ == "__main__":
    main()
