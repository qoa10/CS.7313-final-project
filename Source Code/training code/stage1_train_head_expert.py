#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
from ultralytics import YOLO

ROOT = Path(r"F:\Tao\longtail\dataset")

BASELINE_WEIGHTS = ROOT / r"F:\Tao\longtail\code\runs\detect\yolov8s_animal_lt_head_expert_ep20\weights\best.pt"
DATA_D_HEAD      = ROOT / r"yolo\animal_lt_d_tail.yaml"
RUN_NAME         = "yolov8s_animal_lt_tail_expert_ep20"

IMG_SIZE   = 640
EPOCHS     = 20          # try 5 or 10
BATCH_SIZE = 64
LR0        = 1e-4       # small LR for fine-tuning


def main():
    print("========== STAGE 2: Train Head Expert on D-Head ==========")
    print(f"[INFO] Baseline weights: {BASELINE_WEIGHTS}")
    print(f"[INFO] D-Head yaml:      {DATA_D_HEAD}")
    print(f"[INFO] Run name:         {RUN_NAME}")

    if not BASELINE_WEIGHTS.exists():
        raise FileNotFoundError(f"Baseline weights not found: {BASELINE_WEIGHTS}")
    if not DATA_D_HEAD.exists():
        raise FileNotFoundError(f"D-Head yaml not found: {DATA_D_HEAD}")

    # Load baseline model
    model = YOLO(str(BASELINE_WEIGHTS))

    # Figure out how many top-level layers there are
    n_layers = len(model.model.model)  # this is the ModuleList (0..22)
    freeze_layers = n_layers - 1       # freeze 0..(n_layers-2), keep last (Detect)

    print(f"[INFO] Model has {n_layers} modules, freezing first {freeze_layers} "
          f"(0..{freeze_layers-1}), Detect head remains trainable.")

    # Train with built-in freeze
    model.train(
        data=str(DATA_D_HEAD),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        lr0=LR0,
        name=RUN_NAME,
        deterministic=True,
        freeze=freeze_layers,   # <--- IMPORTANT
    )

    run_dir     = ROOT / "runs" / "detect" / RUN_NAME
    best_weights = run_dir / "weights" / "best.pt"
    last_weights = run_dir / "weights" / "last.pt"

    print("\n========== STAGE 2 DONE ==========")
    print(f"[INFO] Run directory: {run_dir}")
    print(f"[INFO] Best weights:  {best_weights}")
    print(f"[INFO] Last weights:  {last_weights}")


if __name__ == "__main__":
    main()
