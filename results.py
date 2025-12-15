from ultralytics import YOLO

def main():
    # 1) Đường dẫn tới model đã train xong
    model_path = r"C:\Users\CoreUltra7\Desktop\ultralytics\runs_rice\Yolo11_seg_dataset2\weights\best.pt"

    # 2) Load model
    model = YOLO(model_path)

    # 3) Val lại trên dataset mà model đã được train (data, imgsz... được lưu trong model)
    metrics = model.val()  # theo docs: không cần truyền thêm args nếu không đổi dataset

    # 4) In các chỉ số mAP cho BOUNDING BOX (Box) và MASK (Segmentation)
    print("=== BOX metrics (bounding boxes) ===")
    print(f"Box mAP50-95 (metrics.box.map):   {metrics.box.map:.4f}")
    print(f"Box mAP50    (metrics.box.map50): {metrics.box.map50:.4f}")
    print(f"Box mAP75    (metrics.box.map75): {metrics.box.map75:.4f}")
    print(f"Box mAP50-95 per class (metrics.box.maps):")
    print(metrics.box.maps)  # list mAP50-95 cho từng class

    print("\n=== MASK metrics (segmentation masks) ===")
    print(f"Mask mAP50-95 (metrics.seg.map):   {metrics.seg.map:.4f}")
    print(f"Mask mAP50    (metrics.seg.map50): {metrics.seg.map50:.4f}")
    print(f"Mask mAP75    (metrics.seg.map75): {metrics.seg.map75:.4f}")
    print(f"Mask mAP50-95 per class (metrics.seg.maps):")
    print(metrics.seg.maps)  # list mAP50-95 cho từng class

if __name__ == "__main__":
    main()
