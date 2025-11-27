# modify.py
from ultralytics import YOLO

def train_modify():
    # 1) Khởi tạo từ YAML bạn đã sửa (yolov8n-seg-p2.yaml có nc=4)
    # model = YOLO("yolov8n-seg-p2.yaml")

    # 2) Train và nạp pretrain COCO seg đúng cách
    # model.train(
    #     data="data2/data.yaml",      # names phải có 4 class
    #     pretrained="yolo11n-seg.pt", # <- KHÔNG dùng .reset()
    #     epochs=80,
    #     imgsz=832,
    #     batch=8,
    #     device="cuda",
    #     freeze=10,
    #     lr0=4e-3, lrf=4e-4, cos_lr=True,
    #     mosaic=0.4, copy_paste=0.35,
    #     project="runs_rice", name="modify_p2_stage1"
    # )

    best_stage1 = "runs_rice/modify_p2_stage22/weights/best.pt"

    # Stage-2
    model2 = YOLO(best_stage1)
    model2.train(
        data="data2/data.yaml",
        epochs=150,
        imgsz=832,
        batch=8,
        device="cuda",
        freeze=0,
        lr0=2.5e-3, lrf=2e-4, cos_lr=True,
        mosaic=0.25, copy_paste=0.2, mixup=0.0,
        project="runs_rice", name="modify_p2_stage2"
    )

if __name__ == "__main__":
    train_modify()
