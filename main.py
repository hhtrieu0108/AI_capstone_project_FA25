from typing import List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ultralytics import YOLO


# ---------- Load model 1 lần khi khởi động ----------
MODEL_PATH = r"ultralytics\runs_rice\best_model\weights\best.pt"
#MODEL_PATH = r"C:\Users\CoreUltra7\Desktop\ultralytics\runs_rice\yolo11n-trans_all_sett\weights\best.pt"
#MODEL_PATH = r"C:\Users\CoreUltra7\Desktop\ultralytics\runs_rice\Yolo11_seg_dataset2\weights\best.pt"
model = YOLO(MODEL_PATH)

app = FastAPI(
    title="Rice Disease Segmentation API",
    description="FastAPI wrapper cho YOLO segmentation (Ultralytics)",
    version="1.0.0",
)

# Cho phép CORS cho web client
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tùy bạn siết lại domain sau
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


class ImagePrediction(BaseModel):
    filename: str
    image_width: int
    image_height: int
    detections: List[Detection]


class MultiPredictionResponse(BaseModel):
    results: List[ImagePrediction]


# ---------- Utils ----------
def read_image_from_upload(file: UploadFile) -> np.ndarray:
    """Đọc UploadFile thành ảnh BGR (numpy) cho YOLO."""
    data = np.frombuffer(file.file.read(), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Cannot decode image: {file.filename}")
    return img


# ---------- Inference endpoint: nhiều ảnh ----------
@app.post("/predict", response_model=MultiPredictionResponse)
async def predict(files: List[UploadFile] = File(...)) -> MultiPredictionResponse:
    """
    Nhận N ảnh, resize vào model 640x640, trả về bbox + class + mask polygon theo KÍCH THƯỚC ẢNH GỐC.
    """

    all_results: List[ImagePrediction] = []

    for file in files:
    # 1) Đọc ảnh gốc
        img = read_image_from_upload(file)
        h0, w0 = img.shape[:2]  # kích thước gốc để trả về cho frontend

        # 2) Cho model predict TRỰC TIẾP trên ảnh gốc
        # Ultralytics sẽ tự resize/letterbox nội bộ với imgsz=640 và
        # scale kết quả về hệ toạ độ (h0, w0) của ảnh gốc.
        results_list = model.predict(
            img,
            verbose=False,
            conf=0.4,
            imgsz=640,  # giữ kích thước input mạng như bạn muốn
        )
        result = results_list[0]

        boxes = result.boxes
        masks = result.masks
        names = result.names  # dict id -> class_name

        detections: List[Detection] = []

        if boxes is not None:
            # boxes.xyxy đã ở hệ toạ độ ảnh gốc (h0, w0)
            xyxy = boxes.xyxy.cpu().numpy()           # (N, 4)
            conf = boxes.conf.cpu().numpy()           # (N,)
            cls = boxes.cls.cpu().numpy().astype(int) # (N,)

            mask_polygons = None
            if masks is not None:
                # masks.xy cũng đã ở hệ toạ độ ảnh gốc:contentReference[oaicite:3]{index=3}
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

                    # Trường hợp có nhiều segment
                    if isinstance(seg_i, list):
                        for arr in seg_i:
                            points = arr.tolist()  # [[x, y], ...] trên hệ toạ độ ảnh gốc
                            polys_for_det.append(MaskPolygon(points=points))
                    else:
                        # Trường hợp 1 segment ndarray (num_points, 2)
                        points = seg_i.tolist()
                        polys_for_det.append(MaskPolygon(points=points))

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

        img_pred = ImagePrediction(
            filename=file.filename,
            image_width=int(w0),   # KÍCH THƯỚC ẢNH GỐC
            image_height=int(h0),
            detections=detections,
        )
        all_results.append(img_pred)


    return MultiPredictionResponse(results=all_results)


# ---------- Health check ----------
@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL_PATH}
