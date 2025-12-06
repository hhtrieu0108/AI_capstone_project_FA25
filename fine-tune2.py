from ultralytics import YOLO

DATA_YAML = "dataset2/data.yaml"
PROJECT = "runs_rice"

STAGE1_NAME = "fine-tune_stage1_freeze10"
STAGE2_NAME = "fine-tune_stage2_unfreeze"

# Anh có thể chỉnh trong khoảng anh muốn:
STAGE1_EPOCHS = 80   # khoảng 60–100
STAGE2_EPOCHS = 100  # khoảng 100–120


def main():
    # -----------------------
    # STAGE 0: load checkpoint 200 epochs hiện tại
    # -----------------------
    start_ckpt = r"C:\Users\CoreUltra7\Downloads\transfer_200epochs_augment-20251205T034455Z-1-001\transfer_200epochs_augment\weights\best.pt"
    model = YOLO(start_ckpt)

    # -----------------------
    # STAGE 1: freeze 10 layer đầu,
    # tinh chỉnh nhẹ để cải thiện recall và ổn định hơn
    # -----------------------
    model.train(
        data=DATA_YAML,
        epochs=STAGE1_EPOCHS,
        imgsz=640,
        batch=8,
        device="cuda",

        # Freeze backbone dưới, chỉ học phần trên + head
        freeze=10,

        # Học chậm hơn một chút so với default lr0=0.01 để tránh “giật”
        lr0=0.005,
        lrf=0.1,
        cos_lr=True,

        # Augmentation “vừa phải”, giữ màu & hình dạng vết bệnh
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=5.0,
        translate=0.1,
        scale=0.5,
        shear=2.0,
        perspective=0.0005,
        mosaic=0.8,   # vẫn dùng mosaic nhưng không quá mạnh
        mixup=0.15,
        copy_paste=0.3,  # rất hữu ích cho segmentation bệnh lá

        # Loss giữ gần default
        box=7.5,
        cls=0.5,
        dfl=1.5,

        project=PROJECT,
        name=STAGE1_NAME,
        workers=4,
    )

    # -----------------------
    # STAGE 2: unfreeze toàn bộ backbone + tăng imgsz + cải thiện mask
    # -----------------------
    stage1_best = fr"{PROJECT}/{STAGE1_NAME}/weights/best.pt"
    model2 = YOLO(stage1_best)

    model2.train(
        data=DATA_YAML,
        epochs=STAGE2_EPOCHS,

        # Tăng resolution để mask biên bệnh sắc nét hơn
        imgsz=768,
        multi_scale=True,   # random quanh 768, giúp generalize tốt hơn

        batch=4,            # giảm nếu VRAM hạn chế
        device="cuda",

        # Unfreeze hết
        freeze=0,

        # Giảm LR để fine-tune mượt hơn, tránh phá vỡ feature đã học
        lr0=0.002,
        lrf=0.1,
        cos_lr=True,

        # Giảm một chút cường độ augment để stage 2 học chuẩn hơn
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=5.0,
        translate=0.1,
        scale=0.5,
        shear=2.0,
        perspective=0.0005,
        mosaic=0.5,
        mixup=0.10,
        copy_paste=0.35,

        # Tập trung hơn cho mask & IoU cao
        overlap_mask=True,  # default cho segment
        mask_ratio=2,       # default là 4 → dùng mask phân giải cao hơn
        box=8.0,            # tăng nhẹ trọng số box IoU
        cls=0.4,            # giảm chút trọng số cls (4 class khá dễ phân biệt)
        dfl=1.5,

        project=PROJECT,
        name=STAGE2_NAME,
        workers=4,
    )

    # (tuỳ chọn) validate lại sau stage 2
    metrics = model2.val()
    print("mAP50-95:", metrics.box.map)
    print("mAP50:", metrics.box.map50)
    print("mAP75:", metrics.box.map75)
    print("Per-class mAP:", metrics.box.maps)


if __name__ == "__main__":
    main()
