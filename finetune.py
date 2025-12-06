# finetune.py
from ultralytics import YOLO

def train_finetune():
    # --- STAGE 1: Freeze backbone, khớp head 4 lớp ---
    best_transfer = r"C:\Users\CoreUltra7\Desktop\ultralytics\runs_rice\yolov11-captsone-transfer-learning\weights\best.pt"
    model1 = YOLO(best_transfer)  # pretrained seg COCO
    model1.train(
        data="dataset2/data.yaml",      # names: [Chay_bia_la, Dao_on, Dom_nau, Dom_soc_vi_khuan]
        epochs=80,
        imgsz=640,
        batch=8,
        device="cuda",
        freeze=10,
        augment=True,
        lr0=5e-3, 
        lrf=5e-4, 
        cos_lr=True,
        mosaic=0.5, 
        copy_paste=0.3,
        hsv_h=0.015, 
        hsv_s=0.7, 
        hsv_v=0.4,
        project="runs_rice", 
        name="yolo11_finetune(from_transfer)_stage1"
    )

    # đường dẫn best của stage-1
    best_stage1 = "runs_rice/yolo11_finetune(from_transfer)_stage1/weights/best.pt"
    # --- STAGE 2: Unfreeze, tinh chỉnh toàn bộ ---
    # Tạo object mới và nạp best stage-1 làm pretrained init
    model2 = YOLO(best_stage1)
    model2.train(
        data="dataset2/data.yaml",
        epochs=120,
        imgsz=640,
        batch=8,
        device="cuda",
        freeze=0,
        augment=True,
        lr0=2e-3, 
        lrf=2e-4, 
        cos_lr=True,
        mosaic=0.3, 
        copy_paste=0.2,
        mixup=0.0,
        project="runs_rice", 
        name="yolo11_finetune_stage2"
    )

if __name__ == "__main__":
    train_finetune()