from ultralytics import YOLO
import cv2
import torch
# ============================
# 1️ Load model
# ============================
model_path = r"runs_rice/yolo11_seg_stage2_full_finetune(2)/weights/best.pt"
#model_path = r"C:\Users\CoreUltra7\Desktop\ultralytics\runs_rice\fine-tune_stage2_unfreeze\weights\best.pt"
#model_path = r"C:\Users\CoreUltra7\Desktop\ultralytics\runs_rice\Yolo11_seg_dataset2\weights\best.pt"
#model_path = r"C:\Users\CoreUltra7\Desktop\ultralytics\runs_rice\yolo11n-trans_all_sett\weights\best.pt"
#model_path = r"C:\Users\CoreUltra7\Desktop\ultralytics\runs_rice\Yolo11_seg_dataset_450_epochs\weights\best.pt"  # hoặc đường dẫn bạn lưu model
#model_path = r"C:\Users\CoreUltra7\Desktop\ultralytics\runs_rice\yolo11_finetune_stage2\weights\best.pt"
model = YOLO(model_path)

# ============================
# 2️ Dự đoán trên 1 ảnh
# ============================
image_path = r"C:\Users\CoreUltra7\Desktop\ultralytics\Dau-hieu-benh-dom-nau-tren-la.png"
#image_path = r"C:\Users\CoreUltra7\Desktop\ultralytics\z7322666108425_8511e38334cd17c44899aa03ce2f4ae4.jpg"  # thay bằng ảnh bạn muốn test
results = model.predict(
    task="segment",
    source=image_path,
    conf=0.4,      # ngưỡng tin cậy
    iou=0.6,      # ngưỡng NMSp
    save=False,     # lưu kết quả (mask overlay)
    show=False,     # hiển thị cửa sổ OpenCV nếu True
    retina_masks=True
)

torch.cuda.empty_cache()  # giải phóng bộ nhớ GPU nếu cần
# ============================
# 3️ Xem thông tin dự đoán
# ============================
for result in results:
    boxes = result.boxes.xyxy  # toạ độ bounding boxes (x1, y1, x2, y2)
    masks = result.masks.data if result.masks is not None else None
    names = result.names

    print("Số đối tượng phát hiện:", len(boxes))
    for i, box in enumerate(boxes):
        cls_id = int(result.boxes.cls[i])
        conf = float(result.boxes.conf[i])
        label = names[cls_id]
        print(f"{i+1}. Lớp: {label} - Độ tin cậy: {conf:.2f} - Box: {box.tolist()}")

# ============================
# 4 Hiển thị kết quả (OpenCV)
# ============================
result_img = results[0].plot()  # ảnh có overlay mask + bbox
cv2.imshow("Segmentation Result", result_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
