import os
import json
import random
from collections import defaultdict

# ====== paths ======
DATA_ROOT = r"F:\Tao\longtail\dataset\data"
ANN_DIR   = os.path.join(DATA_ROOT, "annotations")

TRAIN_JSON = os.path.join(ANN_DIR, "mini_lvis_animal_train.json")
VAL_JSON   = os.path.join(ANN_DIR, "mini_lvis_animal_val.json")

VAL_RATIO   = 0.30
RANDOM_SEED = 42
# ===================


def load_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(p, data):
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f)
    print(f"[SAVE] {p} (images={len(data['images'])}, annos={len(data['annotations'])})")


def main():
    random.seed(RANDOM_SEED)

    train = load_json(TRAIN_JSON)
    val   = load_json(VAL_JSON)

    # categories should be identical in both
    categories = train["categories"]

    # merge images
    img_by_id = {}
    for img in train["images"] + val["images"]:
        img_by_id[img["id"]] = img

    # merge annotations
    all_anns = train["annotations"] + val["annotations"]

    # group annos by category
    anns_by_cat = defaultdict(list)
    for ann in all_anns:
        anns_by_cat[ann["category_id"]].append(ann)

    new_train_anns = []
    new_val_anns = []

    print("[INFO] Per-class 7:3 splitting:")
    for cid, ann_list in sorted(anns_by_cat.items()):
        total = len(ann_list)
        random.shuffle(ann_list)

        n_train = int(round((1.0 - VAL_RATIO) * total))  # ≈ 70%
        # ensure at least 1 in each split when possible
        if n_train == 0 and total > 0:
            n_train = 1
        if n_train == total and total > 1:
            n_train = total - 1

        t_anns = ann_list[:n_train]
        v_anns = ann_list[n_train:]

        new_train_anns.extend(t_anns)
        new_val_anns.extend(v_anns)

        print(f"  class {cid}: total={total}, train={len(t_anns)}, val={len(v_anns)}")

    # collect images used by each split
    train_img_ids = {a["image_id"] for a in new_train_anns}
    val_img_ids   = {a["image_id"] for a in new_val_anns}

    new_train_imgs = [img_by_id[iid] for iid in sorted(train_img_ids)]
    new_val_imgs   = [img_by_id[iid] for iid in sorted(val_img_ids)]

    # reassign annotation ids inside each split
    for i, ann in enumerate(new_train_anns, start=1):
        ann["id"] = i
    for i, ann in enumerate(new_val_anns, start=1):
        ann["id"] = i

    new_train_data = {
        "images": new_train_imgs,
        "annotations": new_train_anns,
        "categories": categories,
    }
    new_val_data = {
        "images": new_val_imgs,
        "annotations": new_val_anns,
        "categories": categories,
    }

    save_json(TRAIN_JSON, new_train_data)
    save_json(VAL_JSON, new_val_data)


if __name__ == "__main__":
    main()
