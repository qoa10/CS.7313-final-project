from ultralytics import YOLO
from pathlib import Path

ROOT = Path(r"F:\Tao\longtail\dataset")
DATA_YAML = r"F:\Tao\longtail\dataset\yolo\animal_lt.yaml"

def main():
    model = YOLO("yolov8m.pt")  # COCO-pretrained

    model.train(
        data=str(DATA_YAML),
        project=str(ROOT / "runs"),
        name="yolov8s_animal_lt",
        epochs=100,
        imgsz=640,
        batch=64,
        device=0,   # use "cpu" if no GPU
        workers=0,
        patience=30,
        mosaic=1,  # turn off heavy aug on tiny dataset
        mixup=0.0,
        copy_paste=0.0,
        freeze=10,  # freeze backbone, train head first
    )

if __name__ == "__main__":
    main()
