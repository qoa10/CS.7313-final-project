import os
import json
import shutil
from pathlib import Path

# ===================== CONFIG =====================

ROOT = r"F:\Tao\longtail\dataset"   # project root

DATA_DIR   = os.path.join(ROOT, "data")
IMAGE_ROOT = os.path.join(DATA_DIR, "images")        # where all images live
ANNOT_DIR  = os.path.join(DATA_DIR, "annotations")

MINI_TRAIN_JSON = os.path.join(ANNOT_DIR, "mini_lvis_animal_train.json")
MINI_VAL_JSON   = os.path.join(ANNOT_DIR, "mini_lvis_animal_val.json")

YOLO_ROOT   = os.path.join(ROOT, "yolo")
YOLO_IMG_TR = os.path.join(YOLO_ROOT, "images", "train")
YOLO_IMG_VA = os.path.join(YOLO_ROOT, "images", "val")
YOLO_LAB_TR = os.path.join(YOLO_ROOT, "labels", "train")
YOLO_LAB_VA = os.path.join(YOLO_ROOT, "labels", "val")

# =================================================

IMAGE_INDEX = {}  # basename -> full path


def ensure_dirs():
    for d in [YOLO_IMG_TR, YOLO_IMG_VA, YOLO_LAB_TR, YOLO_LAB_VA]:
        os.makedirs(d, exist_ok=True)


def load_coco(path):
    print(f"[INFO] Load COCO: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def build_image_index():
    """
    Walk IMAGE_ROOT once and index all image files by basename.
    Handles both 000000123456.jpg and COCO_train2017_000000123456.jpg.
    """
    global IMAGE_INDEX
    IMAGE_INDEX = {}
    print(f"[INFO] Building image index under: {IMAGE_ROOT}")
    for root, dirs, files in os.walk(IMAGE_ROOT):
        for fname in files:
            lower = fname.lower()
            if not (lower.endswith(".jpg") or lower.endswith(".jpeg") or lower.endswith(".png")):
                continue
            full = os.path.join(root, fname)
            # exact name
            IMAGE_INDEX.setdefault(fname, full)

            # If LVIS-style name like COCO_train2017_000000123456.jpg,
            # also index by the last part 000000123456.jpg
            if fname.startswith("COCO_train2017_") or fname.startswith("COCO_val2017_"):
                tail = fname.split("_")[-1]
                IMAGE_INDEX.setdefault(tail, full)

    print(f"[INFO] Image index built, {len(IMAGE_INDEX)} files indexed.")


def find_image_file(file_name: str):
    """
    Map LVIS file_name/basename to actual path using IMAGE_INDEX.
    """
    # direct lookup
    path = IMAGE_INDEX.get(file_name)
    if path is not None:
        return path

    # If file_name itself is COCO_train2017_000000123456.jpg but
    # the index only has 000000123456.jpg (or vice versa)
    if "train2017_" in file_name or "val2017_" in file_name:
        tail = file_name.split("_")[-1]
        path = IMAGE_INDEX.get(tail)
        if path is not None:
            return path

    return None


def coco_to_yolo_bbox(bbox, img_w, img_h):
    """COCO bbox [x_min, y_min, w, h] -> YOLO [cx, cy, w, h] normalized"""
    x_min, y_min, w, h = bbox
    cx = x_min + w / 2.0
    cy = y_min + h / 2.0

    cx /= img_w
    cy /= img_h
    w  /= img_w
    h  /= img_h
    return cx, cy, w, h


def convert_split(coco_data, split_name):
    """
    split_name: 'train' or 'val'
    """
    if split_name == "train":
        img_out_dir = YOLO_IMG_TR
        lab_out_dir = YOLO_LAB_TR
    else:
        img_out_dir = YOLO_IMG_VA
        lab_out_dir = YOLO_LAB_VA

    images = coco_data["images"]
    annos  = coco_data["annotations"]
    cats   = coco_data["categories"]

    # build mapping category_id -> 0..(K-1) in the mini order
    cats_sorted = sorted(cats, key=lambda c: c["id"])
    catid2yoloid = {c["id"]: idx for idx, c in enumerate(cats_sorted)}

    print("[INFO] Mini categories in this JSON (id -> name):")
    for c in cats_sorted:
        print(f"    {c['id']} -> {c['name']}")

    # group annos by image
    anns_by_img = {}
    for ann in annos:
        anns_by_img.setdefault(ann["image_id"], []).append(ann)

    skipped_images = 0
    used_images = 0

    for img in images:
        img_id = img["id"]

        # LVIS v1: sometimes no 'file_name', often only 'coco_url'
        file_name = img.get("file_name")
        if file_name is None:
            coco_url = img.get("coco_url")
            if coco_url:
                file_name = os.path.basename(coco_url)
            else:
                print(f"[WARN] image id {img_id} has no file_name and no coco_url, skip.")
                skipped_images += 1
                continue

        img_w = img["width"]
        img_h = img["height"]

        src_path = find_image_file(file_name)
        if src_path is None:
            print(f"[WARN] Cannot find image file for {file_name}, skip.")
            skipped_images += 1
            continue

        used_images += 1

        # Copy image
        dst_img_path = os.path.join(img_out_dir, Path(file_name).name)
        shutil.copy2(src_path, dst_img_path)

        # Prepare label file
        dst_lab_path = os.path.join(lab_out_dir, Path(file_name).stem + ".txt")
        lines = []
        for ann in anns_by_img.get(img_id, []):
            cid = ann["category_id"]
            if cid not in catid2yoloid:
                continue
            yolo_cls = catid2yoloid[cid]  # 0-based

            cx, cy, w, h = coco_to_yolo_bbox(ann["bbox"], img_w, img_h)
            # clip to [0, 1]
            cx = max(min(cx, 1.0), 0.0)
            cy = max(min(cy, 1.0), 0.0)
            w  = max(min(w, 1.0), 0.0)
            h  = max(min(h, 1.0), 0.0)

            line = f"{yolo_cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
            lines.append(line)

        with open(dst_lab_path, "w", encoding="utf-8") as f:
            if lines:
                f.write("\n".join(lines))

    print(f"[INFO] Finished split={split_name}. Used images: {used_images}, skipped images: {skipped_images}")


def write_yolo_yaml(coco_data, yaml_path):
    cats = sorted(coco_data["categories"], key=lambda c: c["id"])
    names = [c["name"] for c in cats]
    nc = len(names)

    yaml_text = "path: ..\n" \
                "train: yolo/images/train\n" \
                "val: yolo/images/val\n\n" \
                f"nc: {nc}\n" \
                "names:\n"

    for n in names:
        yaml_text += f"  - {n}\n"

    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_text)
    print(f"[SAVE] YOLO yaml -> {yaml_path}")


def main():
    ensure_dirs()
    build_image_index()

    coco_train = load_coco(MINI_TRAIN_JSON)
    coco_val   = load_coco(MINI_VAL_JSON)

    convert_split(coco_train, "train")
    convert_split(coco_val, "val")

    yaml_path = os.path.join(YOLO_ROOT, "animal_lt.yaml")
    write_yolo_yaml(coco_train, yaml_path)


if __name__ == "__main__":
    main()
