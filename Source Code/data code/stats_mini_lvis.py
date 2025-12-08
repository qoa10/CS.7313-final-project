import os
import json
from collections import Counter

# ====== paths (change if needed) ======
DATA_ROOT = r"F:\Tao\longtail\dataset\data"
ANN_DIR   = os.path.join(DATA_ROOT, "annotations")

TRAIN_JSON = os.path.join(ANN_DIR, "mini_lvis_animal_train.json")
VAL_JSON   = os.path.join(ANN_DIR, "mini_lvis_animal_val.json")
# ======================================


def load_coco(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def count_split(coco):
    # cat_id -> name
    id2name = {c["id"]: c["name"] for c in coco["categories"]}
    counter = Counter(ann["category_id"] for ann in coco["annotations"])

    return id2name, counter


def main():
    print("[INFO] Train stats:")
    train = load_coco(TRAIN_JSON)
    id2name_train, cnt_train = count_split(train)
    for cid in sorted(id2name_train.keys()):
        name = id2name_train[cid]
        n = cnt_train.get(cid, 0)
        print(f"  {cid:2d} ({name:6s}): {n:4d}")

    print("\n[INFO] Val stats:")
    val = load_coco(VAL_JSON)
    id2name_val, cnt_val = count_split(val)
    for cid in sorted(id2name_val.keys()):
        name = id2name_val[cid]
        n = cnt_val.get(cid, 0)
        print(f"  {cid:2d} ({name:6s}): {n:4d}")


if __name__ == "__main__":
    main()
