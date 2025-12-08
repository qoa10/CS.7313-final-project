#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Stage 3: Tail KD training for YOLOv8 (detection only)

- Teacher = head-expert model (after D-Head training), fully frozen
- Student = same architecture, initialized from the same weights
- Train on D-Tail images WITHOUT labels (KD-only)
- Freeze backbone + neck (modules 0..21), only Detect head (module 22) is trainable
- KD loss = KL between teacher & student feature maps (3 FPN maps, per-location softmax)

After KD, evaluate the new student on the original Animal-LT val set.
"""

import os
from pathlib import Path

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from ultralytics import YOLO
from tqdm import tqdm

# ===================== 1. CONFIG ===================== #

# ⚠️ Your Stage-2 head-expert checkpoint (20-epoch D-Head best.pt)
HEAD_EXPERT_WEIGHTS = r"F:\Tao\longtail\code\runs\detect\yolov8s_animal_lt_head_expert_ep20\weights\best.pt"

# We use the same weights for teacher and initial student
TEACHER_WEIGHTS       = HEAD_EXPERT_WEIGHTS
STUDENT_INIT_WEIGHTS  = HEAD_EXPERT_WEIGHTS

# D-Tail image root (KD-only, no labels)
DTAIL_IMG_DIR = r"F:\Tao\longtail\dataset\images\train_d_tail"

# Original Animal-LT data yaml for evaluation
ORIG_DATA_YAML = r"F:\Tao\longtail\dataset\yolo\animal_lt.yaml"

# Where to save the KD student (state_dict)
OUT_STATE = r"F:\Tao\longtail\dataset\yolo\stage3_tail_kd_student_state.pth"

# Training hyper-parameters
EPOCHS       = 20
BATCH_SIZE   = 16
IMG_SIZE     = 640
LR           = 1e-4
WEIGHT_DECAY = 5e-4
MOMENTUM     = 0.9
LAMBDA_KD    = 1.0
TEMP         = 2.0
NUM_WORKERS  = 4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ===================== 2. D-Tail dataset (unlabeled) ===================== #

class DTailImageDataset(Dataset):
    """
    Simple image dataset for D-Tail.
    We only need images (no labels) for KD-only training.
    """

    def __init__(self, img_root: str, img_size: int = 640):
        self.img_root = Path(img_root)
        assert self.img_root.exists(), f"D-Tail dir does not exist: {img_root}"

        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        self.files = [p for p in self.img_root.rglob("*") if p.suffix.lower() in exts]
        if len(self.files) == 0:
            raise RuntimeError(f"No images found under {img_root}")

        self.img_size = img_size
        print(f"[INFO] D-Tail dataset: {len(self.files)} images")

    def __len__(self):
        return len(self.files)

    @staticmethod
    def letterbox(img, new_size=640, color=(114, 114, 114)):
        """
        YOLO-style letterbox:
        - keep aspect ratio
        - pad to (new_size, new_size)
        """
        h0, w0 = img.shape[:2]
        r = min(new_size / h0, new_size / w0)
        new_unpad = (int(round(w0 * r)), int(round(h0 * r)))
        dw, dh = new_size - new_unpad[0], new_size - new_unpad[1]
        dw /= 2
        dh /= 2

        if (w0, h0) != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(
            img, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=color
        )
        return img

    def __getitem__(self, idx):
        img_path = self.files[idx]
        img = cv2.imread(str(img_path))
        if img is None:
            raise RuntimeError(f"Failed to read image: {img_path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.letterbox(img, self.img_size)
        img = img.astype("float32") / 255.0
        # HWC -> CHW
        img = torch.from_numpy(img).permute(2, 0, 1).contiguous()
        return img, str(img_path)


# ===================== 3. KD loss on feature maps ===================== #

def kd_loss_from_features(student_feats, teacher_feats, temp: float = 1.0) -> torch.Tensor:
    """
    KD on feature maps from Detect head (3 scales).

    student_feats / teacher_feats: list[Tensor], each Tensor: (B, C, H, W)

    For each layer:
      - reshape (B, C, H, W) -> (B*H*W, C)
      - per-location softmax over C channels
      - KL(student || teacher)
    """
    if not isinstance(student_feats, (list, tuple)) or not isinstance(teacher_feats, (list, tuple)):
        raise ValueError(f"[KD] Expected list/tuple, got student {type(student_feats)}, teacher {type(teacher_feats)}")

    if len(student_feats) != len(teacher_feats):
        raise ValueError(f"[KD] Different number of feature maps: student {len(student_feats)} vs teacher {len(teacher_feats)}")

    total_kd = 0.0
    n_layers = len(student_feats)

    for i, (s, t) in enumerate(zip(student_feats, teacher_feats)):
        if s.shape != t.shape:
            raise ValueError(f"[KD] Layer {i} shape mismatch: student {s.shape} vs teacher {t.shape}")

        B, C, H, W = s.shape
        # (B, C, H, W) -> (B*H*W, C)
        s = s.view(B, C, -1).permute(0, 2, 1).contiguous().view(-1, C)
        t = t.view(B, C, -1).permute(0, 2, 1).contiguous().view(-1, C)

        s_log_prob = F.log_softmax(s / temp, dim=1)
        t_prob     = F.softmax(t / temp, dim=1)

        kd = F.kl_div(s_log_prob, t_prob, reduction="batchmean") * (temp * temp)
        total_kd = total_kd + kd

    return total_kd / n_layers


# ===================== 4. Helpers ===================== #
from typing import Iterable, Literal, Optional

FreezeMode = Literal["detect_only", "last_blocks", "all"]

def freeze_student_backbone(
    student_det_model: nn.Module,
    mode: FreezeMode = "last_blocks",
    extra_trainable_idx: Optional[Iterable[int]] = None,
):
    """
    student_det_model: ultralytics.nn.tasks.DetectionModel

    mode:
      - "detect_only": only last Detect head trainable  (old behavior)
      - "last_blocks": Detect + a few last blocks trainable (for Exp A / D)
      - "all":         full finetune, everything trainable (for Exp C)

    extra_trainable_idx:
      - optional explicit indices to mark as trainable in addition to the default
        (useful if you inspect the model and pick exact layers).
    """
    modules = list(student_det_model.model)
    n_layers = len(modules)  # typically 23 for YOLOv8s: 0..22

    assert n_layers >= 2, f"Unexpected number of modules: {n_layers}"
    detect_idx = n_layers - 1  # last one is Detect

    # ----- choose which indices are trainable ----- #
    if mode == "detect_only":
        trainable_idx = {detect_idx}

    elif mode == "last_blocks":
        # Example: unfreeze last 3 blocks + Detect
        # You can adjust this range if you want more/less:
        # e.g. range(n_layers-4, n_layers) -> last 4 modules incl. Detect
        start = max(0, n_layers - 4)
        trainable_idx = set(range(start, n_layers))  # e.g. {19, 20, 21, 22}

    elif mode == "all":
        trainable_idx = set(range(n_layers))

    else:
        raise ValueError(f"Unknown mode: {mode}")

    # add extra indices if user provides
    if extra_trainable_idx is not None:
        trainable_idx |= set(extra_trainable_idx)

    # ----- actually set requires_grad ----- #
    for i, m in enumerate(modules):
        requires_grad = (i in trainable_idx)
        for p in m.parameters():
            p.requires_grad = requires_grad

    total = sum(p.numel() for p in student_det_model.parameters())
    trainable = sum(p.numel() for p in student_det_model.parameters()
                    if p.requires_grad)
    print(
        f"[INFO] Student DetectionModel has {n_layers} modules. "
        f"Trainable params: {trainable} / {total} ({trainable/total:.4%}). "
        f"Trainable module indices: {sorted(trainable_idx)}"
    )


def extract_yolo_feats(model_out):
    """
    Extract the 3 feature maps [P3,P4,P5] from YOLO DetectionModel forward output.

    - In training mode, DetectionModel usually returns a list[Tensor].
    - In eval mode, some versions return a tuple(preds, [feats]).
    This helper normalizes both to a list[Tensor].
    """
    # case 1: already a list/tuple of tensors
    if isinstance(model_out, list):
        return model_out

    if isinstance(model_out, tuple):
        # find the first element that is a list/tuple of tensors
        for item in model_out:
            if isinstance(item, (list, tuple)):
                return list(item)

    raise TypeError(f"Unsupported YOLO model output type for feats extraction: {type(model_out)}")


# ===================== 5. Main KD training loop ===================== #

def main():
    device = torch.device(DEVICE)
    print(f"[INFO] Using device: {device}")

    # ---- D-Tail dataset ----
    dtail_dataset = DTailImageDataset(DTAIL_IMG_DIR, img_size=IMG_SIZE)
    loader = DataLoader(
        dtail_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    # ---- Load YOLO teacher & student ----
    print("[INFO] Loading YOLO teacher & student ...")

    yolo_teacher = YOLO(TEACHER_WEIGHTS)
    yolo_student = YOLO(STUDENT_INIT_WEIGHTS)

    teacher_model: nn.Module = yolo_teacher.model.to(device)
    student_model: nn.Module = yolo_student.model.to(device)

    # Freeze ALL teacher params
    for p in teacher_model.parameters():
        p.requires_grad = False

    teacher_model.eval()   # fixed teacher
    student_model.train()  # student will be trained

    # Freeze backbone + neck of student; only Detect head trainable
    freeze_student_backbone(student_model)

    # ---- Warmup forward for shape check ----
    print("\n=== Warmup forward for shape check ===")
    imgs0, _ = next(iter(loader))
    imgs0 = imgs0.to(device)

    with torch.no_grad():
        t_raw = teacher_model(imgs0)
    s_raw = student_model(imgs0)

    t_feats = extract_yolo_feats(t_raw)
    s_feats = extract_yolo_feats(s_raw)

    print("Student feats type:", type(s_feats))
    for i, f in enumerate(s_feats):
        print(f"  S[{i}] shape:", f.shape)
    print("Teacher feats type:", type(t_feats))
    for i, f in enumerate(t_feats):
        print(f"  T[{i}] shape:", f.shape)
    print("=== End shape check ===\n")

    # ---- Build optimizer on student trainable params ----
    trainable_params = [p for p in student_model.parameters() if p.requires_grad]
    num_trainable = sum(p.numel() for p in trainable_params)
    print(f"[DEBUG] #trainable params in student: {num_trainable}, tensors: {len(trainable_params)}")

    if len(trainable_params) == 0:
        raise RuntimeError("[BUG] No trainable parameters found for student. Check freeze_student_backbone().")

    optimizer = torch.optim.SGD(
        trainable_params,
        lr=LR,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
    )

    # ---- KD training loop ----
    for epoch in range(1, EPOCHS + 1):
        student_model.train()
        teacher_model.eval()  # keep eval; no BN stat updates

        epoch_kd = 0.0
        n_batches = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch}/{EPOCHS}", ncols=100)
        for imgs, _paths in pbar:
            imgs = imgs.to(device, non_blocking=True)

            # teacher forward (no grad)
            with torch.no_grad():
                teacher_raw = teacher_model(imgs)
                teacher_feats = extract_yolo_feats(teacher_raw)

            # student forward
            student_raw = student_model(imgs)
            student_feats = extract_yolo_feats(student_raw)

            # KD loss on Detect feature maps
            kd = kd_loss_from_features(student_feats, teacher_feats, temp=TEMP)
            loss = LAMBDA_KD * kd

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            epoch_kd += kd.item()
            n_batches += 1

            pbar.set_postfix({"L_kd": f"{kd.item():.3f}"})

        avg_kd = epoch_kd / max(n_batches, 1)
        print(f"[EPOCH {epoch}] kd_loss={avg_kd:.4f}")

    # ---- Save student state_dict ----
    os.makedirs(os.path.dirname(OUT_STATE), exist_ok=True)
    torch.save(student_model.state_dict(), OUT_STATE)
    print(f"[INFO] Saved KD-only student weights (state_dict) to: {OUT_STATE}")

    # ===================== 6. Evaluation on original val set ===================== #
    print("\n[INFO] Running evaluation on original val set (animal_lt.yaml)...")

    # Re-create YOLO model from the same architecture, then load KD student weights
    eval_yolo = YOLO(STUDENT_INIT_WEIGHTS)  # just to get architecture
    eval_yolo.model.load_state_dict(torch.load(OUT_STATE, map_location=device))

    eval_device = 0 if device.type == "cuda" else "cpu"
    metrics = eval_yolo.val(
        data=ORIG_DATA_YAML,
        split="val",
        imgsz=IMG_SIZE,
        device=eval_device,
        verbose=True,
    )

    print("[INFO] Val metrics:", metrics)
    if hasattr(metrics, "results_dict"):
        print("[INFO] results_dict:", metrics.results_dict)


if __name__ == "__main__":
    main()
