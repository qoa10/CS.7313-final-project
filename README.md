# CS.7313 Final Project — Long-Tail Object Detection

This repository contains the implementation of a three-stage long-tail object detection pipeline using a mini-LVIS dataset. The goal is to improve tail-class performance using a head-expert model and knowledge distillation.

---

## 📦 Dataset (10 GB)

The full dataset is hosted on Google Drive:

**🔗 Dataset Download:**
[https://drive.google.com/drive/folders/1o0CEVT2s0Pl-W_71onLo3iuziuupjb1T?usp=drive_link](https://drive.google.com/drive/folders/1o0CEVT2s0Pl-W_71onLo3iuziuupjb1T?usp=drive_link)

After downloading, place the dataset anywhere on your machine and update the dataset path variables inside the scripts (e.g., `DATA_ROOT`).

### Dataset Structure

```
dataset_root/
├── full data/              # Original mini-LVIS dataset (Stage 0)
├── images/                 # YOLO-format images for D-Head / D-Tail
├── labels/                 # YOLO-format labels
├── results from kd/        # Teacher predictions used for KD
├── d_head_instances.txt    # Head-class instances predicted by teacher
└── d_tail_instances.txt    # Tail-class instances predicted by teacher
```

---

## ⚙️ Quick Environment Setup (PyCharm)

1. Clone this repository:

   ```
   git clone https://github.com/<your-username>/CS.7313-final-project.git
   ```
2. Open the folder in **PyCharm**.
3. Create a new virtual environment:
   *File → Settings → Project → Python Interpreter → Add → Virtualenv*
4. Install required packages:

   ```
   pip install torch torchvision ultralytics opencv-python numpy pandas tqdm pycocotools
   ```
5. Update dataset paths in all scripts to match your local directory.

---

## 📁 Repository Structure

```
CS.7313-final-project/
├── data code/          # Dataset building and preprocessing scripts
├── training code/      # Training scripts for Stage 0 / Stage 1 / Stage 2
└── result/             # Saved weights and experiment outputs
```

---

## 🧩 Short Explanation of Each Folder & Script

### **data code/** – Dataset Construction & Utilities

* **build Mini-LVIS-Animal-LT.py**
  Creates the mini-LVIS animal-only long-tail dataset from the original LVIS annotation.

* **build D-Head.py**
  Selects high-frequency (head) classes and generates the D-Head subset.

* **build_d_head_d_tail_datasets.py**
  Creates D-Head and D-Tail subsets from instance lists and writes corresponding data splits.

* **build_mini_to_yolo.py**
  Converts LVIS/COCO annotation format into YOLO `.txt` files for detection training.

* **resplit_mini_70_30.py**
  Produces a 70/30 train–val split for consistent evaluation.

* **stats_mini_lvis.py**
  Computes class frequencies and shows the long-tail distribution of the dataset.

---

### **training code/** – Full Training Pipeline

* **stage0_pretrian_fulldata.py**
  Baseline training on the full mini-LVIS dataset (teacher model for KD).

* **stage1_train_head_expert.py**
  Trains a head-class expert model on **D-Head**.

* **stage2_train_tail_kd.py**
  Tail-class student trained with **pure Knowledge Distillation** using teacher predictions.

* **stage2_train_tail_kd+gt.py**
  Tail-class student trained with **KD + Ground Truth loss**.

---

### **result/** – Saved Outputs

Contains:

* Stage 0 baseline results
* Stage 1 head-expert results
* Stage 2: GT-only, KD-only, KD+GT experiments
* Model weights, logs, and evaluation metrics

You can use these to inspect results without re-running training.

---

## ▶️ Running Experiments

After setting dataset paths, run training scripts in this order:

```
# Stage 0 — Baseline model
python "training code/stage0_pretrian_fulldata.py"

# Stage 1 — Head-class expert
python "training code/stage1_train_head_expert.py"

# Stage 2 — Tail-class student (KD only)
python "training code/stage2_train_tail_kd.py"

# Stage 2 — Tail-class student (KD + GT)
python "training code/stage2_train_tail_kd+gt.py"
```

---

## ✅ Notes

* All scripts are standalone and can be run directly inside PyCharm.
* CUDA/GPU is recommended for training speed but not required for inference.
* If you only want to inspect results, use the pretrained models inside `result/`.


# # 📘 **End-to-End Workflow Summary (English + Code Steps)**

This workflow converts **YOLO bounding boxes → SAM2 segmentation masks → YOLO-seg polygons → COCO JSON → CVAT editable annotations**.

---

# ## 1. **Activate the project environment**

Activate your Python virtual environment before running any scripts.

```bash
cd F:\Tao\cvat
.\.venv310\Scripts\activate
```

---

# ## 2. **Start CVAT + Serverless backend (Nuclio)**

This brings up CVAT with all required containers.

```bash
docker compose -f docker-compose.yml -f components/serverless/docker-compose.serverless.yml up -d
```

---

# ## 3. **Verify that CVAT is running**

Use Docker to confirm that CVAT containers are alive.

```bash
docker ps
```

---

# ## 4. **Run SAM2 on a single image + YOLO box file**

This generates a segmentation mask + YOLO-seg polygons + visualization.

```bash
python box2mask_sam2.py --image "path/to/img.png" --yolo-txt "path/to/label.txt" --out "output_mask.png" --model-id "facebook/sam2-hiera-large"
```

---

# ## 5. **Run SAM2 on an entire folder (batch processing)**

This converts all YOLO bounding-box labels into YOLO-seg polygon labels.

```bash
python box2mask_sam2.py --images-dir "F:\Tao\dataset\images" --labels-dir "F:\Tao\dataset\labels" --out-dir "F:\Tao\dataset\labels_seg" --model-id "facebook/sam2-hiera-large"
```

---

# ## 6. **Convert YOLO-seg polygon labels → COCO format for CVAT**

This creates a CVAT-compatible JSON file.

```bash
python yoloseg_to_coco_cvat.py --images-dir "F:\Tao\cvat\test\images" --seg-labels-dir "F:\Tao\cvat\test\labels_seg" --output-json "F:\Tao\cvat\test\coco_sam2_for_cvat.json"
```

---

# ## 7. **Open CVAT in the browser**

Access the CVAT interface to create tasks and import annotations.

```text
http://localhost:8080
```

---

# ## 8. **Create a new CVAT task**

Add labels matching your dataset classes (e.g., class_0 … class_n).

```text
CVAT → Tasks → Create Task → Add labels manually
```

---

# ## 9. **Upload images into the CVAT task**

Select the same images used for JSON generation.

```text
CVAT → Task → Data → Select "test/images" folder → Submit
```

---

# ## 10. **Import COCO JSON annotations into CVAT**

Load all polygon segmentation results into the task.

```text
CVAT → Task → Actions → Upload annotations → Format: COCO 1.0 → Select coco_sam2_for_cvat.json
```

---

# ## 11. **Visually inspect and refine polygons in CVAT**

Use the polygon editing tool to adjust vertices and correct segmentation errors.

```text
CVAT editor → Polygon tool → Edit → Save
```

---

# ## 12. **Export refined annotations (optional)**

Export the cleaned segmentation dataset for YOLOv8-seg training.

```text
CVAT → Task → Actions → Export annotations → YOLO Segmentation 1.0
```

---

# ## 13. **Train YOLOv8-seg with your refined segmentation dataset**

Use the exported YOLO-seg labels to train the segmentation model.

```bash
yolo train model=yolov8s-seg.pt data=your_data.yaml imgsz=640 epochs=100
```

---

# ## ✔️ Final Summary

You have now built a **full industrial-grade annotation pipeline**:

1. Activate environment
2. Start CVAT
3. Run SAM2 segmentation on YOLO boxes
4. Generate YOLO-seg polygon labels
5. Convert to COCO JSON
6. Import into CVAT
7. Refine masks
8. Export final dataset
9. Train YOLOv8-seg





