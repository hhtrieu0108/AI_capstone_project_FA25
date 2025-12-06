from ultralytics import YOLO
import torch

def main():
    # dùng kiến trúc detect custom
    model = YOLO("yolo11n-seg.pt")
    print(torch.cuda.is_available())
    print(torch.__version__)

    # nạp weight pretrained detect COCO (không phải -seg)
    # model.load("yolo11n-seg.pt")
    

    model.train(
        data="dataset2/data.yaml",   # file YAML roboflow (train/val/test/nc/names)      
        epochs=200, # weight pretrained COCO detect
        imgsz=640,
        batch=8,
        device="cuda", 
        # freeze=10,          
        workers=4,
        project="runs_rice",
        name="yolov11-captsone-transfer-learning",
        augment=True,
    )

if __name__ == "__main__":
    main()
