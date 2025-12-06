# finetune_last.py - Two-Stage Fine-tuning for YOLO11n Segmentation
# Stage 1: Freeze backbone (10 layers), fine-tune head
# Stage 2: Unfreeze all layers, full fine-tuning
# Total epochs: 450 (Stage 1: 250 + Stage 2: 200)
# Dataset: 1504 training images, 4 rice disease classes

from ultralytics import YOLO
import torch

def finetune_last():
    print("=" * 60)
    print("YOLO11n Segmentation - Two-Stage Fine-tuning")
    print("=" * 60)
    
    # ==========================================
    # STAGE 1: Freeze backbone (10 layers)
    # ==========================================
    print("\n[STAGE 1] Starting: Freeze backbone training...")
    print("Freezing first 10 layers of backbone")
    
    # Load model từ best.pt (model đã train trước đó)
    model1 = YOLO("runs_rice/Yolo11_seg_dataset2/weights/best.pt")
    
    # Freeze first 10 layers (backbone layers 0-9)
    # Ultralytics: model.model.model là backbone
    for i, layer in enumerate(model1.model.model):
        if i < 10:  # Freeze first 10 layers
            for param in layer.parameters():
                param.requires_grad = False
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model1.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model1.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,}")
    
    # Stage 1 Training
    model1 = model1.train(
        data="dataset2/data.yaml",
        epochs=250,             # Stage 1: 250 epochs
        imgsz=640,
        batch=16,
        device="cuda",
        workers=4,
        
        # Augmentation strategy for segmentation
        augment=True,
        mosaic=1.0,             # Mosaic augmentation (strong)
        mixup=0.15,             # Mixup augmentation
        flipud=0.5,             # Vertical flip
        fliplr=0.5,             # Horizontal flip
        degrees=15,             # Rotation range
        translate=0.15,         # Translation
        scale=0.6,              # Scale variation
        hsv_h=0.015,            # HSV Hue
        hsv_s=0.7,              # HSV Saturation
        hsv_v=0.4,              # HSV Value
        
        # Learning rate (higher for stage 1 - head training)
        lr0=0.001,              # Initial LR
        lrf=0.0001,             # Final LR ratio
        warmup_epochs=3,        # Warmup epochs
        warmup_momentum=0.8,
        
        # Optimizer
        optimizer="SGD",        # SGD for stable training
        momentum=0.937,
        weight_decay=0.0005,
        
        # Loss weights for segmentation
        box=7.5,                # Box loss weight
        cls=0.5,                # Class loss weight
        dfl=1.5,                # DFL loss weight
        
        # Validation
        val=True,
        patience=100,            # Early stopping
        
        # Output
        project="runs_rice",
        name="Yolo11_finetune_stage1_freeze_last",
        save=True,
        # save_period=10,
        verbose=True
    )
    
    # print("\n[STAGE 1] Completed!")
    # print(f"Best mAP50: {results_stage1.results_dict.get('metrics/segmentation/mAP50', 'N/A'):.4f}")
    
    # # ==========================================
    # # STAGE 2: Unfreeze all layers
    # # ==========================================
    # print("\n" + "=" * 60)
    # print("[STAGE 2] Starting: Full fine-tuning (all layers)...")
    # print("=" * 60)
    
    # Load best model from stage 1
    model2 = YOLO("runs_rice/Yolo11_finetune_stage1_freeze_last/weights/best.pt")
    
    # Unfreeze all layers
    for layer in model2.model.model:
        for param in layer.parameters():
            param.requires_grad = True
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model2.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model2.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,}")
    
    # Stage 2 Training
    model2 = model2.train(
        data="dataset2/data.yaml",
        epochs=200,             # Stage 2: 200 epochs (total: 250+200=450)
        imgsz=640,
        batch=16,
        device="cuda",
        workers=4,
        
        # Moderate augmentation for stage 2
        augment=True,
        mosaic=0.8,             # Reduce mosaic slightly
        mixup=0.1,              # Reduce mixup
        flipud=0.5,
        fliplr=0.5,
        degrees=10,             # Less aggressive rotation
        translate=0.1,
        scale=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        
        # Lower learning rate for fine-tuning
        lr0=0.0001,             # Much lower LR
        lrf=0.00001,            # Very low final LR
        warmup_epochs=1,        # Short warmup
        
        # Optimizer
        optimizer="SGD",
        momentum=0.937,
        weight_decay=0.0005,
        
        # Loss weights (fine-tune)
        box=7.5,
        cls=0.5,
        dfl=1.5,
        
        # Validation
        val=True,
        patience=100,
        
        # Output
        project="runs_rice",
        name="Yolo11_finetune_stage2_unfreeze_last",
        save=True,
        # save_period=10,
        verbose=True
    )
    
    # print("\n[STAGE 2] Completed!")
    # print(f"Best mAP50: {results_stage2.results_dict.get('metrics/segmentation/mAP50', 'N/A'):.4f}")
    
    # # ==========================================
    # # Summary
    # # ==========================================
    # print("\n" + "=" * 60)
    # print("TRAINING SUMMARY")
    # print("=" * 60)
    # print(f"Stage 1 (Freeze): 250 epochs")
    # print(f"Stage 2 (Unfreeze): 200 epochs")
    # print(f"Total: 450 epochs")
    # print(f"\nBest model: runs_rice/Yolo11_finetune_stage2_unfreeze/weights/best.pt")
    # print("=" * 60)

if __name__ == "__main__":
    finetune_last()
