from ultralytics import YOLO

# Define a YOLO model
model = YOLO("yolo11-seg-modify.yaml").load("yolo11s-seg.pt")

# Define search space
# search_space = {
#     "lr0": (1e-5, 1e-1),
#     "degrees": (0.0, 45.0),
# }

# Tune hyperparameters on COCO8 for 30 epochs
model.tune(
    data="dataset2/data.yaml",
    epochs=40,
    iterations=50,
    device="cuda",
    workers=4,
    batch=64,
    optimizer="auto",
    # space=search_space,
    plots=True,
    save=True,
    val=False,
    resume=True
)