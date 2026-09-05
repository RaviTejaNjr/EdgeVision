"""Export YOLOv5n to TorchScript so the Nano can load it without Ultralytics."""

import yaml
from ultralytics import YOLO

with open("configs/params.yaml") as f:
    cfg = yaml.safe_load(f)

model = YOLO(cfg["model"]["weights"])
model.export(format="torchscript", imgsz=cfg["model"]["input_res"])