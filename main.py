from typing import List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ultralytics import YOLO


# ---------- Load model 1 lần khi khởi động ----------
# best-seg.pt có thể là yolov8-seg hoặc yolo11-seg, API như nhau (task=segment):contentReference[oaicite:16]{index=16}
MODEL_PATH = r"ultralytics\runs_rice\best_model\weights\best.pt"
model = YOLO(MODEL_PATH)

app = FastAPI(
    title="Rice Disease Segmentation API",
    description="FastAPI wrapper cho YOLO segmentation (Ultralytics)",
    version="1.0.0",
)

# Cho phép CORS cho web client
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tuỳ bạn siết lại domain sau
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- Pydantic schemas ----------
class Box(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class MaskPolygon(BaseModel):
    # Một segment polygon: list điểm [x, y]
    points: List[List[float]]


class Detection(BaseModel):
    id: int
    class_id: int
    class_name: str
    confidence: float
    box: Box
    mask: Optional[List[MaskPolygon]] = None


class PredictionResponse(BaseModel):
    image_width: int
    image_height: int
    detections: List[Detection]


# ---------- Utils ----------
def read_image_from_upload(file: UploadFile) -> np.ndarray:
    """Đọc UploadFile thành ảnh BGR (numpy) cho YOLO."""
    data = np.frombuffer(file.file.read(), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Cannot decode image")
    return img


# ---------- Inference endpoint ----------
@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)) -> PredictionResponse:
    """
    Nhận 1 ảnh, chạy YOLO segmentation, trả về bbox + class + mask polygon.
    """

    # 1) Parse input image
    img = read_image_from_upload(file)

    # 2) Run inference (không cần stream)
    # results là list[Results]; mỗi Results = 1 ảnh:contentReference[oaicite:17]{index=17}
    results_list = model.predict(
        img,
        verbose=False,   # tắt log
        conf=0.25,       # ngưỡng confidence tuỳ chỉnh
    )
    result = results_list[0]

    # Lấy shape gốc của ảnh
    h, w = result.orig_shape  # (H, W):contentReference[oaicite:18]{index=18}
    names = result.names      # dict id -> class_name:contentReference[oaicite:19]{index=19}

    boxes = result.boxes      # Boxes object
    masks = result.masks      # Masks object

    detections: List[Detection] = []

    if boxes is not None:
        # Chuyển về numpy/CPU cho dễ serialize:contentReference[oaicite:20]{index=20}
        xyxy = boxes.xyxy.cpu().numpy()    # shape (N, 4)
        conf = boxes.conf.cpu().numpy()    # shape (N,)
        cls = boxes.cls.cpu().numpy().astype(int)  # shape (N,)

        # Chuẩn bị polygon nếu có mask
        mask_polygons = None
        if masks is not None:
            # masks.xy: list length = N, mỗi phần tử là:
            # - ndarray shape (num_points, 2), hoặc
            # - list[ndarray] nếu có nhiều segment
            mask_polygons = masks.xy

        num_det = len(xyxy)
        for i in range(num_det):
            x1, y1, x2, y2 = xyxy[i].tolist()
            c_id = int(cls[i])
            c_name = names.get(c_id, str(c_id))
            c_conf = float(conf[i])

            polygons: Optional[List[MaskPolygon]] = None
            if mask_polygons is not None:
                seg_i = mask_polygons[i]

                polys_for_det: List[MaskPolygon] = []

                # Trường hợp seg_i là list nhiều segment
                if isinstance(seg_i, list):
                    for arr in seg_i:
                        # arr: (num_points, 2) → arr.tolist(): [[x, y], [x, y], ...]
                        polys_for_det.append(
                            MaskPolygon(points=arr.tolist())
                        )
                else:
                    # Trường hợp phổ biến: seg_i là ndarray (num_points, 2)
                    polys_for_det.append(
                        MaskPolygon(points=seg_i.tolist())
                    )

                polygons = polys_for_det

            det = Detection(
                id=i,
                class_id=c_id,
                class_name=c_name,
                confidence=c_conf,
                box=Box(x1=x1, y1=y1, x2=x2, y2=y2),
                mask=polygons,
            )
            detections.append(det)

    return PredictionResponse(
        image_width=int(w),
        image_height=int(h),
        detections=detections,
    )


# ---------- Health check ----------
@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL_PATH}


# ---------- Run local ----------
# uvicorn main:app --reload --host 0.0.0.0 --port 8000
