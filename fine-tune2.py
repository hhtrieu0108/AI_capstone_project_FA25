from ultralytics import YOLO

DATA_YAML = "dataset3/data.yaml"       # dataset 4 bệnh lúa
MODEL_CFG = "yolo11-seg-modify.yaml"   # cấu trúc model
PRETRAINED = "yolo11s-seg.pt"          # weight COCO seg


def freeze_backbone_only(model, backbone_last_idx=9):
    """
    Khóa toàn bộ backbone (model.0 .. model.backbone_last_idx),
    giữ nguyên toàn bộ neck + head (model.backbone_last_idx+1 .. end).

    Ở đây backbone_last_idx=9 dựa trên log freeze=10 của bạn:
    freeze=10 -> Ultralytics đã freeze model.0 .. model.9.
    """
    for name, param in model.model.named_parameters():
        # Tên dạng 'model.<idx>....'
        parts = name.split(".")
        if len(parts) < 3:
            continue
        if parts[0] != "model":
            continue

        try:
            idx = int(parts[1])
        except ValueError:
            continue

        if idx <= backbone_last_idx:
            param.requires_grad = False  # backbone: khóa
        else:
            param.requires_grad = True   # neck + head: cho phép train


def train_2stages():
    # ==========================
    # STAGE 1: chỉ train head, khóa backbone
    # ==========================
    model1 = YOLO(MODEL_CFG).load(PRETRAINED)

    # Khóa toàn bộ backbone, giữ nguyên head (detect + seg)
    freeze_backbone_only(model1, backbone_last_idx=9)

    model1.train(
        data=DATA_YAML,
        epochs=200,          # 200 epoch đầu
        imgsz=640,
        batch=16,
        device="cuda",
        workers=4,
        augment=True,
        # KHÔNG dùng freeze của Ultralytics nữa
        # freeze=0 hoặc bỏ hẳn
        freeze=0,

        # Optimizer & LR: giữ cấu hình bạn đang dùng gần đây
        optimizer="AdamW",
        lr0=0.001,           # LR chuẩn cho AdamW (docs Ultralytics)
        lrf=0.1,             # LR cuối ≈ 0.0001
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,

        # Không set box/cls/dfl, mosaic, hsv... -> dùng default
        project="runs_rice",
        name="yolov11_stage1_backbone_frozen_head_train",
        save=True,
        val=True,
        patience=80,
        verbose=True,
    )

    # ==========================
    # STAGE 2: fine-tune full model (mở backbone)
    # ==========================
    model_best_stage1 = "runs_rice/yolov11_stage1_backbone_frozen_head_train/weights/best.pt"
    model2 = YOLO(model_best_stage1)

    # Mở lại toàn bộ để fine-tune (hoặc bạn có thể vẫn gọi freeze_backbone_only
    # nếu muốn tiếp tục chỉ train head)
    for _, param in model2.model.named_parameters():
        param.requires_grad = True

    model2.train(
        data=DATA_YAML,
        epochs=300,          # thêm 300 epoch -> tổng 500
        imgsz=640,
        batch=16,
        device="cuda",
        workers=4,
        augment=True,
        freeze=0,

        optimizer="AdamW",
        lr0=0.0007,          # nhỏ hơn stage 1 một chút
        lrf=0.1,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,

        project="runs_rice",
        name="yolov11_stage2_full_finetune_adamw",
        save=True,
        val=True,
        patience=100,
        verbose=True,
    )


if __name__ == "__main__":
    train_2stages()
