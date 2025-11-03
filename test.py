from ultralytics import YOLO
import cv2

# ============================
# 1️⃣ Load model
# ============================
model_path = r""  # hoặc đường dẫn bạn lưu model
model = YOLO(model_path)

# ============================
# 2️⃣ Dự đoán trên 1 ảnh
# ============================
image_path = "DOMSOC2.jpg"  # thay bằng ảnh bạn muốn test
results = model.predict(
    source=image_path,
    conf=0.05,      # ngưỡng tin cậy
    iou=0.45,      # ngưỡng NMS
    save=True,     # lưu kết quả (mask overlay)
    show=False     # hiển thị cửa sổ OpenCV nếu True
)

# ============================
# 3️⃣ Xem thông tin dự đoán
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
# 4️⃣ Hiển thị kết quả (OpenCV)
# ============================
result_img = results[0].plot()  # ảnh có overlay mask + bbox
cv2.imshow("Segmentation Result", result_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
