from ultralytics import YOLO
import torch

def main():
    model = YOLO("yolov11-seg.yaml").load("yolo11n-seg.pt")   # pretrained YOLO11n-seg on COCO

    print("CUDA available:", torch.cuda.is_available())
    print("Torch version:", torch.__version__)

    model.train(
        # ====== DỮ LIỆU & I/O ======
        data="dataset2/data.yaml",          # 4 lớp: Dao_on, Dom_nau, Chay_bia_la, Dom_soc_vi_khuan
        project="runs_rice",
        name="yolo11n-trans_all_sett",
        exist_ok=False,                     # không overwrite run cũ
        save=True,
        # save_period=25,                     # 5 lần/lần train 400 epoch
        cache=False,                        # nếu RAM dư có thể chuyển thành 'ram' theo docs

        # ====== LỊCH TRAIN & TỐI ƯU ======
        epochs=400,                         # nhiều hơn default 100 để fine-tune kỹ dataset nhỏ
        time=None,                          # không giới hạn theo giờ
        patience=50,                        # early stop nếu 50 epoch không cải thiện
        batch=16,                            # giống cấu hình bạn đã dùng, phù hợp GPU 8–12 GB
        imgsz=640,                          # lớn hơn 640 để thấy rõ đốm nhỏ trên lá
        device="cuda",
        workers=8,
        pretrained=True,                    # tiếp tục từ weight COCO
        optimizer="SGD",                   # để Ultralytics tự chọn (SGD/AdamW) theo doc 
        seed=42,
        deterministic=True,

        # ====== TRANSFER LEARNING (FREEZE) ======
        freeze=10,                          # freeze các layer backbone thấp, fine-tune head + neck
                                            # theo phong cách “feature reuse” trong doc hyperparam 

        # ====== LEARNING RATE & WARMUP ======
        lr0=5e-3,                           # nhỏ hơn default 1e-2 → an toàn cho fine-tune từ COCO 
        lrf=5e-4,                           # LR cuối rất nhỏ, giúp hội tụ mượt ở cuối train
        momentum=0.937,                     # giữ default do Ultralytics đã tối ưu trên COCO 
        weight_decay=5e-4,                  # L2 regularization chuẩn YOLO
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        cos_lr=True,                        # dùng cosine LR scheduler, docs khuyến nghị cho hội tụ ổn định 

        # ====== KIỂM SOÁT BÀI TOÁN ======
        single_cls=False,                   # 4 lớp riêng, không gộp
        classes=None,                       # train tất cả lớp
        rect=False,
        multi_scale=False,
        amp=True,                           # mixed precision theo mặc định của Ultralytics 
        fraction=1.0,
        profile=False,

        # ====== LOSS & SEGMENTATION ======
        box=7.5,                            # giữ gain box từ default YOLO11
        cls=0.5,                            # class loss gain
        dfl=1.5,                            # distribution focal loss gain
        nbs=64,                             # nominal batch size cho việc scale loss
        overlap_mask=True,                  # nên bật với instance seg để mask không bị mất phần overlap
        mask_ratio=4,                       # downsample mask x4 (default cho YOLO-seg)
        dropout=0.0,
        val=True,
        plots=True,

        # ====== DATA AUGMENTATION ======
        # HSV: giữ gần default vì dataset ngoài đồng ánh sáng thay đổi mạnh
        hsv_h=0.015,                        # thay đổi hue nhẹ, theo cấu hình gốc YOLO 
        hsv_s=0.7,                          # tăng giảm saturation mạnh để chịu được khác biệt màu lá 
        hsv_v=0.4,                          # biến thiên độ sáng (nắng gắt, bóng tay, mây che)

        # Hình học: phù hợp lá lúa (dài, hẹp)
        degrees=10.0,                       # xoay nhẹ ±10° để mô hình quen nhiều hướng lá, theo range 0–180 trong doc 
        translate=0.10,                     # dịch vị trí vật thể tối đa 10% ảnh 
        scale=0.50,                         # zoom 0.5 ± → mô phỏng khoảng cách máy ảnh khác nhau
        shear=0.0,                          # không shear để không làm méo shape lesion
        perspective=0.0,                    # ảnh thực tế chụp gần orthographic, không cần perspective mạnh

        mixup=0.1,              # Mixup augmentation
        flipud=0.5,             # Vertical flip
        fliplr=0.5,             # Horizontal flip           # Rotation        
        bgr=0.0,                            # không đảo kênh màu

        # Mosaics & Mixings
        mosaic=0.5,                         # giảm từ 1.0 → 0.5: vẫn giúp small object nhưng đỡ “ảo” cho lesion               
        cutmix=0.0,                         # không cutmix vì lesion cần giữ shape & context lá

        # Instance-seg specific
        copy_paste=0.30,                    # 30% copy-paste tăng số lesion, đặc biệt class hiếm 
        copy_paste_mode="flip",             # cách mặc định trong implementation CopyPaste 

        # Albumentations custom (không dùng thêm ở đây)
        augment=True
    )

if __name__ == "__main__":
    main()
