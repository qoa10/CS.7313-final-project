#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Stage 3: KD + GT joint training for YOLOv8 (Animal-LT)

- Teacher: head-expert model (stage 2), frozen
- Student: same arch, initialized from teacher weights
- Train on D-Tail images with labels (d_tail.yaml)
- Loss = detection loss (GT) + KD loss on feature maps
"""

import os
from pathlib import Path
import yaml
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from ultralytics import YOLO
from ultralytics.utils.loss import v8DetectionLoss
from tqdm import tqdm
from types import SimpleNamespace

# --------------------- CONFIG --------------------- #
TEACHER_WEIGHTS = r"F:\Tao\longtail\code\runs\detect\yolov8s_animal_lt_head_expert_ep20\weights\best.pt"
STUDENT_INIT_WEIGHTS = TEACHER_WEIGHTS

DTAIL_YAML = r"F:\Tao\longtail\dataset\labels\d_tail.yaml"
VAL_YAML   = r"F:\Tao\longtail\dataset\yolo\animal_lt.yaml"

OUT_STATE  = r"F:\Tao\longtail\dataset\yolo\stage3_tail_kd_gt1.pth"

IMG_SIZE   = 640
EPOCHS     = 30
BATCH      = 16
LR         = 1e-4
TEMP       = 2.0          # KD temperature
ALPHA_KD   = 1.0          # weight of KD loss
NUM_WORKERS = 4
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

def freeze_all_but_detect(student: nn.Module):
    """
    student: ultralytics.nn.tasks.DetectionModel
    Freeze backbone+neck, only Detect head trainable.
    """
    modules = list(student.model)       # ModuleList of backbone+neck+Detect
    n_layers = len(modules)
    detect_idx = n_layers - 1           # last layer is Detect

    for i, m in enumerate(modules):
        trainable = (i == detect_idx)
        for p in m.parameters():
            p.requires_grad = trainable

    total = sum(p.numel() for p in student.parameters())
    trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
    print(f"[INFO] freeze_all_but_detect -> trainable params: "
          f"{trainable} / {total} ({trainable/total:.2%}), "
          f"only layer idx {detect_idx} is trainable.")


# --------------------- KD LOSS --------------------- #
def kd_loss_from_features(student_feats, teacher_feats, temp: float = 1.0) -> torch.Tensor:
    """
    KD on feature maps from Detect head (3 scales).

    student_feats / teacher_feats: list[Tensor], each (B, C, H, W)
    """
    if not isinstance(student_feats, (list, tuple)) or not isinstance(teacher_feats, (list, tuple)):
        raise TypeError(f"[KD] expected list/tuple, got {type(student_feats)} and {type(teacher_feats)}")

    assert len(student_feats) == len(teacher_feats), "[KD] student/teacher layer count mismatch"

    total_kd = 0.0
    n_layers = len(student_feats)

    for i, (s, t) in enumerate(zip(student_feats, teacher_feats)):
        if s.shape != t.shape:
            raise ValueError(f"[KD] Layer {i} shape mismatch: {s.shape} vs {t.shape} ")

        B, C, H, W = s.shape
        # flatten spatial, keep channels last: (B*H*W, C)
        s = s.view(B, C, -1).permute(0, 2, 1).contiguous().view(-1, C)
        t = t.view(B, C, -1).permute(0, 2, 1).contiguous().view(-1, C)

        s_log_prob = F.log_softmax(s / temp, dim=1)
        t_prob     = F.softmax(t / temp, dim=1)

        kd = F.kl_div(s_log_prob, t_prob, reduction="batchmean") * (temp * temp)
        total_kd = total_kd + kd

    return total_kd / n_layers


# --------------------- Feature extraction --------------------- #
def extract_feats(model_out):
    """
    Extract the 3 feature maps [P3,P4,P5] from YOLO DetectionModel forward output.
    For ultralytics v8 DetectionModel, forward(img) already returns list[Tensor].
    """
    if isinstance(model_out, list):
        return model_out
    if isinstance(model_out, tuple):
        # sometimes it might be (preds, aux), etc.
        for item in model_out:
            if isinstance(item, (list, tuple)):
                return list(item)
    raise TypeError(f"Unsupported model output type for feats: {type(model_out)}")


# --------------------- Dataset --------------------- #
class YoloDetectDataset(Dataset):
    """Simple YOLO detection dataset reading from a data.yaml like d_tail.yaml"""

    def __init__(self, data_yaml: str, img_size: int = 640):
        with open(data_yaml, "r") as f:
            data = yaml.safe_load(f)

        self.img_dir = Path(data["train"])
        self.label_dir = Path(str(self.img_dir).replace("images", "labels"))
        self.img_size = img_size

        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        self.img_files = sorted(
            [p for p in self.img_dir.rglob("*") if p.suffix.lower() in exts]
        )
        if len(self.img_files) == 0:
            raise RuntimeError(f"No images found under {self.img_dir}")

        print(f"[INFO] D-Tail dataset: {len(self.img_files)} images")

    @staticmethod
    def letterbox_with_labels(img, labels, new=640):
        """
        YOLO-style letterbox, and adjust labels (xywh normalized) to the new image.

        labels: Tensor [N, 5]  (cls, x, y, w, h) normalized to original w,h.
        """
        h0, w0 = img.shape[:2]
        if labels.numel() > 0:
            assert labels.shape[1] == 5, f"labels shape should be [N,5], got {labels.shape}"

        # scale ratio
        r = min(new / h0, new / w0)
        new_unpad = (int(round(w0 * r)), int(round(h0 * r)))
        dw, dh = new - new_unpad[0], new - new_unpad[1]  # padding
        dw /= 2
        dh /= 2

        # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(
            img, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=(114, 114, 114),
        )

        # adjust labels from original (w0,h0) → new (new,new)
        if labels.numel() > 0:
            cls = labels[:, 0:1]
            xywh = labels[:, 1:5]

            x = xywh[:, 0] * w0
            y = xywh[:, 1] * h0
            w = xywh[:, 2] * w0
            h = xywh[:, 3] * h0

            # scale + pad
            x = x * r + dw
            y = y * r + dh
            w = w * r
            h = h * r

            # back to normalized xywh in new image
            x /= new
            y /= new
            w /= new
            h /= new

            labels = torch.cat([cls, x.unsqueeze(1), y.unsqueeze(1),
                                w.unsqueeze(1), h.unsqueeze(1)], dim=1)

        # safety: force exact (new,new)
        if img.shape[0] != new or img.shape[1] != new:
            img = cv2.resize(img, (new, new), interpolation=cv2.INTER_LINEAR)

        return img, labels

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        label_path = self.label_dir / (img_path.stem + ".txt")

        img = cv2.imread(str(img_path))
        if img is None:
            raise RuntimeError(f"Failed to read image: {img_path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # --- read labels (normalized to original image) ---
        labels_list = []
        if label_path.exists():
            with open(label_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    cls, x, y, w, h = map(float, line.split())
                    labels_list.append([cls, x, y, w, h])

        if labels_list:
            labels = torch.tensor(labels_list, dtype=torch.float32)
        else:
            labels = torch.zeros((0, 5), dtype=torch.float32)

        # --- letterbox image AND adjust labels to new size ---
        img, labels = self.letterbox_with_labels(img, labels, self.img_size)

        # to tensor
        img_tensor = torch.from_numpy(img.astype("float32") / 255.0).permute(2, 0, 1)

        return img_tensor, labels


# --------------------- Custom collate --------------------- #
def yolo_collate(batch):
    """
    batch: list of (img, labels) where labels are [N,5] per image.

    Returns:
        imgs:   (B,3,H,W)
        labels: (M,6) with columns [batch_idx, cls, x,y,w,h]
    """
    imgs = []
    labels_out = []

    for i, (img, labels) in enumerate(batch):
        imgs.append(img)
        if labels.numel() > 0:
            bi = torch.full((labels.shape[0], 1), i, dtype=torch.float32)
            labels_out.append(torch.cat([bi, labels], dim=1))  # [N,6]

    imgs = torch.stack(imgs, dim=0)

    if len(labels_out):
        labels_out = torch.cat(labels_out, dim=0)
    else:
        labels_out = torch.zeros((0, 6), dtype=torch.float32)

    return imgs, labels_out


# ================================================================
#  MAIN
# ================================================================
def main():
    device = torch.device(DEVICE)
    print(f"[INFO] Device = {device}")

    # ---- Load teacher & student ----
    print("[INFO] Loading teacher and student weights...")
    y_teacher = YOLO(TEACHER_WEIGHTS)
    y_student = YOLO(STUDENT_INIT_WEIGHTS)

    teacher = y_teacher.model.to(device)
    student = y_student.model.to(device)

    # Freeze teacher
    for p in teacher.parameters():
        p.requires_grad = False
    teacher.eval()

    # Full finetune student in Stage 3
    freeze_all_but_detect(student)

    total = sum(p.numel() for p in student.parameters())
    trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
    print(f"[INFO] Student DetectionModel params: {trainable} / {total} trainable ({trainable/total:.2%})")

    # --------- Build proper student.args for v8DetectionLoss --------- #
    # Start from YOLO object overrides if available (usually contains box/cls/dfl)
    base_cfg = {}

    if hasattr(y_student, "overrides") and isinstance(y_student.overrides, dict):
        base_cfg.update(y_student.overrides)

    # Merge any existing student.args contents
    if hasattr(student, "args"):
        if isinstance(student.args, dict):
            base_cfg.update(student.args)
        elif isinstance(student.args, SimpleNamespace):
            base_cfg.update(vars(student.args))

    # Loss-gain defaults if still missing
    base_cfg.setdefault("box", 7.5)  # default YOLOv8 box gain
    base_cfg.setdefault("cls", 0.5)  # default YOLOv8 cls gain
    base_cfg.setdefault("dfl", 1.5)  # default YOLOv8 DFL gain

    # (optional) some extra fields that loss may touch in some versions
    base_cfg.setdefault("fl_gamma", 0.0)
    base_cfg.setdefault("label_smoothing", 0.0)

    student.args = SimpleNamespace(**base_cfg)

    print("[DEBUG] student.args.box/cls/dfl =",
          student.args.box, student.args.cls, student.args.dfl)

    # --------- Build detection criterion explicitly --------- #
    student.criterion = v8DetectionLoss(student)

    # Optimizer (joint KD + GT)
    optimizer = torch.optim.AdamW(
        [p for p in student.parameters() if p.requires_grad],
        lr=LR,
        weight_decay=5e-4,
    )

    # ---- DataLoader ----
    train_dataset = YoloDetectDataset(DTAIL_YAML, IMG_SIZE)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
        collate_fn=yolo_collate,
    )

    print("[INFO] Starting joint KD + GT training ...")

    # ----------------- Training loop ----------------- #
    for epoch in range(1, EPOCHS + 1):
        student.train()
        epoch_kd = 0.0
        epoch_det = 0.0
        epoch_total = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}", ncols=100)
        for imgs, labels in pbar:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # ---- build batch dict for v8DetectionLoss ----
            if labels.numel() > 0:
                batch_idx = labels[:, 0].long()
                cls = labels[:, 1:2]         # [M,1]
                bboxes = labels[:, 2:6]      # [M,4] (xywh normalized)
            else:
                batch_idx = torch.zeros((0,), dtype=torch.long, device=device)
                cls = torch.zeros((0, 1), dtype=torch.float32, device=device)
                bboxes = torch.zeros((0, 4), dtype=torch.float32, device=device)

            batch = {
                "img": imgs,
                "cls": cls,
                "bboxes": bboxes,
                "batch_idx": batch_idx,
            }

            # ---- forward teacher (no grad) ----
            with torch.no_grad():
                t_out = teacher(imgs)
            teacher_feats = extract_feats(t_out)

            # ---- forward student ----
            s_out = student(imgs)
            student_feats = extract_feats(s_out)

            # ---- KD loss ----
            kd_loss = kd_loss_from_features(student_feats, teacher_feats, TEMP)

            # ---- detection loss (GT) ----
            det_loss_vec, _ = student.criterion(s_out, batch)  # shape [3] = [box, cls, dfl]
            det_loss = det_loss_vec.sum()  # scalar

            # ---- total loss ----
            total_loss = ALPHA_KD * kd_loss + det_loss

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            optimizer.step()

            # ---- logging ----
            bs = imgs.shape[0]
            epoch_kd += kd_loss.item() * bs
            epoch_det += det_loss.item() * bs
            epoch_total += total_loss.item() * bs
            n_batches += bs

            pbar.set_postfix({
                "loss": f"{total_loss.item():.4f}",
                "kd":   f"{kd_loss.item():.4f}",
                "det":  f"{det_loss.item():.4f}",
            })

        avg_kd = epoch_kd / max(1, n_batches)
        avg_det = epoch_det / max(1, n_batches)
        avg_total = epoch_total / max(1, n_batches)
        print(f"[INFO] Epoch {epoch} done. avg_total={avg_total:.4f}, avg_kd={avg_kd:.4f}, avg_det={avg_det:.4f}")

    # ----------------- Save final student weights ----------------- #
    torch.save(student.state_dict(), OUT_STATE)
    print(f"[INFO] Training done. Saved student state_dict to: {OUT_STATE}")
    print("You can now run a separate val to get mAP on animal_lt.yaml, e.g.:")
    print(f"  y = YOLO('{OUT_STATE}')")
    print(f"  y.val(data=r'{VAL_YAML}', imgsz={IMG_SIZE}, batch={BATCH})")


if __name__ == "__main__":
    main()
