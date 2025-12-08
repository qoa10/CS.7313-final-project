# build_mini_lvis_animal_lt.py
import os
import json
import random
from collections import defaultdict

# ================== CONFIG (EDIT THESE) ==================

# Root of your dataset **data** folder
ROOT = r"F:\Tao\longtail\dataset\data"

ANNOT_DIR   = os.path.join(ROOT, "annotations")
IMAGE_ROOT  = os.path.join(ROOT, "images")   # where you extracted COCO images

LVIS_TRAIN_JSON = os.path.join(ANNOT_DIR, "lvis_v1_train.json")

MINI_TRAIN_JSON = os.path.join(ANNOT_DIR, "mini_lvis_animal_train.json")
MINI_VAL_JSON   = os.path.join(ANNOT_DIR, "mini_lvis_animal_val.json")

# Choose 8 animal categories (LVIS names)
SELECTED_CATEGORY_NAMES = [
    "cat",
    "dog",
    "horse",
    "cow",
    "sheep",
    "bird",
    "bear",
    "zebra",
]

HEAD_CLASS_NAMES = ["cat", "dog", "horse", "cow"]
TAIL_CLASS_NAMES = ["sheep", "bird", "bear", "zebra"]

HEAD_MAX_INSTANCES = 500   # per head class (but limited by available images)
TAIL_MAX_INSTANCES = 50    # per tail class

VAL_RATIO   = 0.30
RANDOM_SEED = 42

# =========================================================

IMAGE_INDEX = {}  # basename -> full path


def build_image_index():
    """Scan IMAGE_ROOT and index all images by basename."""
    global IMAGE_INDEX
    IMAGE_INDEX = {}
    print(f"[INFO] Building image index under: {IMAGE_ROOT}")
    for root, dirs, files in os.walk(IMAGE_ROOT):
        for fname in files:
            lower = fname.lower()
            if not (lower.endswith(".jpg") or lower.endswith(".jpeg") or lower.endswith(".png")):
                continue
            full = os.path.join(root, fname)
            IMAGE_INDEX.setdefault(fname, full)

            # If LVIS-style name like COCO_train2017_000000123456.jpg,
            # also index by the last part 000000123456.jpg
            if fname.startswith("COCO_train2017_") or fname.startswith("COCO_val2017_"):
                tail = fname.split("_")[-1]
                IMAGE_INDEX.setdefault(tail, full)

    print(f"[INFO] Image index built, {len(IMAGE_INDEX)} files indexed.\n")


def find_image_file(file_name: str):
    """Return full path if this basename exists in IMAGE_INDEX."""
    # exact
    path = IMAGE_INDEX.get(file_name)
    if path is not None:
        return path

    # maybe LVIS/COCO style name mismatch
    if "train2017_" in file_name or "val2017_" in file_name:
        tail = file_name.split("_")[-1]
        path = IMAGE_INDEX.get(tail)
        if path is not None:
            return path

    return None


def load_lvis_annotations(path):
    print(f"[INFO] Loading LVIS annotations from: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def build_category_maps(categories):
    id_to_cat = {c["id"]: c for c in categories}
    name_to_id = {}
    for c in categories:
        name_to_id[c["name"].lower()] = c["id"]
    return id_to_cat, name_to_id


def pick_category_ids(name_to_id):
    selected_ids = {}
    missing = []
    for name in SELECTED_CATEGORY_NAMES:
        key = name.lower()
        if key in name_to_id:
            selected_ids[key] = name_to_id[key]
        else:
            missing.append(name)

    if missing:
        print("[WARN] These category names were NOT found in LVIS:")
        for m in missing:
            print("   -", m)
        print("[HINT] Adjust SELECTED_CATEGORY_NAMES if needed.\n")

    print("[INFO] Selected categories (name -> id):")
    for n, cid in selected_ids.items():
        print(f"   {n} -> {cid}")
    print()
    return selected_ids


def build_image_info(images):
    """
    For each image, compute:
      - file_name (from file_name or coco_url)
      - exists: whether the file actually exists on disk
    """
    imginfo = {}
    existing = 0
    missing  = 0

    for img in images:
        img_id = img["id"]
        file_name = img.get("file_name")
        if file_name is None:
            coco_url = img.get("coco_url", "")
            file_name = os.path.basename(coco_url) if coco_url else None

        if not file_name:
            exists = False
        else:
            exists = find_image_file(file_name) is not None

        if exists:
            existing += 1
        else:
            missing += 1

        imginfo[img_id] = {
            "img": img,
            "file_name": file_name,
            "exists": exists,
        }

    print(f"[INFO] LVIS train images: {len(images)} total, {existing} with local file, {missing} missing.\n")
    return imginfo


def filter_and_sample_annotations(annotations, selected_ids, head_ids, tail_ids, imginfo):
    """
    Only keep annotations:
      - whose category is in selected_ids
      - AND whose image actually exists on disk
    Then sample per class using HEAD_MAX_INSTANCES / TAIL_MAX_INSTANCES.
    """
    anns_by_cat = defaultdict(list)
    dropped_missing_img = 0

    for ann in annotations:
        cid = ann["category_id"]
        if cid not in selected_ids.values():
            continue

        img_id = ann["image_id"]
        info = imginfo.get(img_id)
        if not info or not info["exists"]:
            dropped_missing_img += 1
            continue

        anns_by_cat[cid].append(ann)

    if dropped_missing_img > 0:
        print(f"[INFO] Dropped {dropped_missing_img} annotations whose image file is missing.\n")

    random.seed(RANDOM_SEED)
    selected_annos = []

    for cat_name, cid in selected_ids.items():
        ann_list = anns_by_cat[cid]
        n_total = len(ann_list)
        if cid in head_ids:
            max_n = HEAD_MAX_INSTANCES
        elif cid in tail_ids:
            max_n = TAIL_MAX_INSTANCES
        else:
            max_n = n_total

        if n_total == 0:
            print(f"[WARN] Category '{cat_name}' (id={cid}) has 0 usable instances after filtering.")
            continue

        n_keep = min(n_total, max_n)
        random.shuffle(ann_list)
        keep = ann_list[:n_keep]
        selected_annos.extend(keep)
        print(f"[INFO] Category '{cat_name}' (id={cid}): total_usable={n_total}, keep={n_keep}")

    print(f"[INFO] Total selected annotations (all classes): {len(selected_annos)}\n")
    return selected_annos


def build_image_subset(imginfo, selected_annotations):
    image_ids = {ann["image_id"] for ann in selected_annotations}
    subset_images = []
    for iid in sorted(image_ids):
        info = imginfo.get(iid)
        if not info or not info["exists"]:
            continue
        subset_images.append(info["img"])
    print(f"[INFO] Total subset images: {len(subset_images)}\n")
    return subset_images


def split_train_val(images, annotations):
    random.seed(RANDOM_SEED)
    image_ids = [img["id"] for img in images]
    random.shuffle(image_ids)

    n_total = len(image_ids)
    n_val = int(round(VAL_RATIO * n_total))
    val_ids = set(image_ids[:n_val])
    train_ids = set(image_ids[n_val:])

    print(f"[INFO] Split images: train={len(train_ids)}, val={len(val_ids)}")

    train_anns = []
    val_anns = []
    for ann in annotations:
        if ann["image_id"] in train_ids:
            train_anns.append(ann)
        elif ann["image_id"] in val_ids:
            val_anns.append(ann)

    print(f"[INFO] Split annotations: train={len(train_anns)}, val={len(val_anns)}\n")
    return train_ids, val_ids, train_anns, val_anns


def save_coco_json(path, images, annotations, categories):
    new_anns = []
    for i, ann in enumerate(annotations, start=1):
        new_ann = dict(ann)
        new_ann["id"] = i
        new_anns.append(new_ann)

    data = {
        "images": images,
        "annotations": new_anns,
        "categories": categories,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    print(f"[SAVE] {path}  (images={len(images)}, annotations={len(new_anns)})")


def main():
    build_image_index()

    lvis = load_lvis_annotations(LVIS_TRAIN_JSON)
    images = lvis["images"]
    annotations = lvis["annotations"]
    categories = lvis["categories"]

    id_to_cat, name_to_id = build_category_maps(categories)
    selected_ids = pick_category_ids(name_to_id)

    if len(selected_ids) == 0:
        print("[ERROR] No selected categories found. Edit SELECTED_CATEGORY_NAMES and run again.")
        return

    # new category IDs 1..K
    mini_categories = []
    oldid_to_newid = {}
    for new_id, (name, old_id) in enumerate(sorted(selected_ids.items()), start=1):
        oldid_to_newid[old_id] = new_id
        mini_categories.append({
            "id": new_id,
            "name": name,
            "supercategory": id_to_cat[old_id].get("supercategory", "animal")
        })

    head_ids_old = {name_to_id[n.lower()] for n in HEAD_CLASS_NAMES if n.lower() in name_to_id}
    tail_ids_old = {name_to_id[n.lower()] for n in TAIL_CLASS_NAMES if n.lower() in name_to_id}

    imginfo = build_image_info(images)

    selected_annos_old = filter_and_sample_annotations(
        annotations, selected_ids, head_ids_old, tail_ids_old, imginfo
    )

    # remap category_id
    selected_annos = []
    for ann in selected_annos_old:
        cid_old = ann["category_id"]
        if cid_old not in oldid_to_newid:
            continue
        ann_new = dict(ann)
        ann_new["category_id"] = oldid_to_newid[cid_old]
        selected_annos.append(ann_new)

    subset_images = build_image_subset(imginfo, selected_annos)

    train_ids, val_ids, train_anns, val_anns = split_train_val(subset_images, selected_annos)

    id_to_image = {img["id"]: img for img in subset_images}
    train_images = [id_to_image[iid] for iid in train_ids]
    val_images   = [id_to_image[iid] for iid in val_ids]

    save_coco_json(MINI_TRAIN_JSON, train_images, train_anns, mini_categories)
    save_coco_json(MINI_VAL_JSON,   val_images,   val_anns,   mini_categories)


if __name__ == "__main__":
    main()
