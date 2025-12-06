# modify.py - YOLO11n Segmentation Training
# Dataset: 1504 training images, 4 rice disease classes
from ultralytics import YOLO

def train_modify():
    # Train từ scratch với YAML custom
    model = YOLO("yolo11-seg-modify.yaml").load("yolo11n-seg.pt")  # Nạp weight pretrained COCO segmentation
    
    model.train(
        data="dataset2/data.yaml",
        epochs=450,
        imgsz=640,              # Image size (tối ưu cho lesion nhỏ)
        batch=16,               # Batch size (1504 samples -> 16 batch)
        device="cuda",
        workers=4,
        
        # Augmentation cho bệnh lá lúa
        augment=True,
        mosaic=1.0,             # Mosaic augmentation
        mixup=0.1,              # Mixup augmentation
        flipud=0.5,             # Vertical flip
        fliplr=0.5,             # Horizontal flip
        degrees=10,             # Rotation
        translate=0.1,          # Translation
        scale=0.5,              # Scale
        
        # Learning rate
        lr0=0.01,               # Initial learning rate
        lrf=0.01,               # Final learning rate ratio
        
        # Optimizer
        optimizer="SGD",        # SGD hoặc Adam
        momentum=0.937,
        weight_decay=0.0005,
        
        # Validation
        val=True,
        patience=50,            # Early stopping patience
        
        # Output
        project="runs_rice",
        name="Yolo11_seg_dataset_450_epochs",
        save=True,
        # save_period=10,
        verbose=True
    )

if __name__ == "__main__":
    train_modify()
