from ultralytics import YOLO
import torch

def main():
    # dùng kiến trúc detect custom
    model = YOLO("yolov8n-seg.yaml")
    print(torch.cuda.is_available())
    print(torch.__version__)

    # nạp weight pretrained detect COCO (không phải -seg)
    model.load("yolov8n-seg.pt")
    

    model.train(
        data="data2/data.yaml",   # file YAML roboflow (train/val/test/nc/names)      
        epochs=30,
        imgsz=640,
        batch=8,
        device="cuda", 
        freeze=10,          
        workers=4,
        project="runs_rice",
        name="detector_from_custom_yaml"
    )

if __name__ == "__main__":
    main()
