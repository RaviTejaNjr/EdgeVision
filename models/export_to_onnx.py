from ultralytics import YOLO

model = YOLO("yolov5nu.pt")
model.export(format="onnx", opset=13, imgsz=640, simplify=True, dynamic=False)