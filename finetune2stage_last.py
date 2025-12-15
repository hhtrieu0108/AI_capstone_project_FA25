# yolo11_2stage.py - YOLO11n Segmentation 2-stage Training
from ultralytics import YOLO

DATA_YAML = "dataset3/data.yaml"
MODEL_CFG = "yolo11-seg-modify.yaml"
PRETRAINED = "yolo11n-seg.pt"


def train_2stage():
    # ==========================
    # STAGE 1: Freeze backbone, train head
    # ==========================
    model1 = YOLO(MODEL_CFG).load(PRETRAINED)

    model1.train(
        data=DATA_YAML,
        epochs=150,          # Stage 1: 150 epoch
        imgsz=640,
        batch=16,
        device="cuda",
        workers=4,

        # ---- Freeze backbone (10 block đầu) ----
        freeze=10,

        # ---- Optimizer & LR (dựa trên config gốc nhưng điều chỉnh nhẹ) ----
        optimizer="SGD",
        lr0=0.008,           # nhỏ hơn 0.01 một chút cho stage head-only
        lrf=0.1,             # LR cuối ~0.0008 cho 150 epoch
        momentum=0.937,
        weight_decay=0.0005,

        warmup_epochs=3.0,
        warmup_momentum=0.8,

        # ---- Augmentation (giữ tinh thần gốc nhưng giảm bớt độ "nặng") ----
        augment=True,
        mosaic=0.8,          # gốc 1.0
        mixup=0.05,          # gốc 0.1
        flipud=0.0,          # bỏ vertical flip
        fliplr=0.5,
        degrees=7.0,         # gốc 10
        translate=0.1,
        scale=0.4,           # gốc 0.5

        # ---- Validation & save ----
        val=True,
        patience=50,
        project="runs_rice",
        name="yolo11_seg_stage1_freeze_backbone",
        save=True,
        verbose=True,
    )

    # ==========================
    # STAGE 2: Fine-tune full model
    # ==========================
    stage1_best = "runs_rice/yolo11_seg_stage1_freeze_backbone/weights/best.pt"
    model2 = YOLO(stage1_best)
    model2.train(
        data=DATA_YAML,
        epochs=300,          # Stage 2: 300 epoch  -> tổng 450
        imgsz=640,
        batch=16,
        device="cuda",
        workers=4,

        # ---- Không freeze nữa ----
        freeze=0,

        # ---- Optimizer & LR: fine-tune nhẹ nhàng hơn ----
        optimizer="SGD",
        lr0=0.003,           # nhỏ hơn stage 1
        lrf=0.01,            # LR cuối ~3e-5
        momentum=0.937,
        weight_decay=0.0005,

        warmup_epochs=3.0,
        warmup_momentum=0.8,

        # ---- Augmentation: "hiền" hơn để tập trung mask lesion ----
        augment=True,
        mosaic=0.5,          # giảm mosaic xuống
        # mixup=0.0,           # tắt mixup ở giai đoạn fine-tune cuối
        # flipud=0.0,
        fliplr=0.5,
        # degrees=5.0,
        translate=0.1,
        scale=0.3,
        mixup=0.1,              # Mixup augmentation
        flipud=0.5,             # Vertical flip         
        degrees=10, 

        val=True,
        patience=80,
        project="runs_rice",
        name="yolo11_seg_stage2_full_finetune(2)",
        save=True,
        verbose=True,
    )


if __name__ == "__main__":
    train_2stage()
