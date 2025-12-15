from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.info(verbose=True)
model.save("yolov8n_arch.yaml")  # hoặc copy từ ultralytics/models/v8/yolov8n.yaml
