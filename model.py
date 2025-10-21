#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
yolo_segmentation.py

Mô tả:
    File này cung cấp lớp YoloSegModel để:
      - Khởi tạo model YOLOv8 segmentation (pretrained hoặc từ scratch)
      - Huấn luyện trên dataset dạng YOLO (images + labels in COCO/YOLO format)
      - Dự đoán segmentation trên ảnh/ folder
      - Export model (onnx, torchscript, etc.)

Hướng dẫn nhanh:
    1) pip install ultralytics opencv-python tqdm
    2) Chuẩn bị dataset theo cấu trúc YOLOv8/Ultralytics (yaml config hoặc COCO)
    3) Chạy ví dụ trong phần if __name__ == "__main__"
"""

from __future__ import annotations
import os
import sys
import shutil
import argparse
from pathlib import Path
from typing import Optional, List, Union, Dict, Any

# Thư viện chính cho YOLOv8
try:
    from ultralytics import YOLO
except Exception as e:
    raise ImportError(
        "Package 'ultralytics' not found. Cài đặt bằng: pip install ultralytics"
    ) from e

import cv2
import json
import tempfile


class YoloSegModel:
    """
    Lớp bao trùm cho YOLOv8 Segmentation.
    Tham số:
        model_name_or_path: 'yolov8n-seg.pt' hoặc đường dẫn tới checkpoint
        work_dir: thư mục lưu checkpoint và kết quả
    """

    def __init__(self, model_name_or_path: str = "yolov8n-seg.pt", work_dir: str = "runs"):
        self.model_name_or_path = model_name_or_path
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        # Tải model (nếu có) — giữ instance để gọi train/predict
        self.model = YOLO(self.model_name_or_path)

    def train(
        self,
        data: Union[str, Dict[str, Any]],
        epochs: int = 50,
        batch: int = 16,
        imgsz: int = 640,
        save_period: int = 1,
        lr: Optional[float] = None,
        project: Optional[str] = None,
        name: Optional[str] = None,
        exist_ok: bool = True,
        resume: bool = False,
        **kwargs,
    ) -> dict:
        """
        Huấn luyện model segmentation.
        Tham số:
            data: đường dẫn tới file config dataset yaml (hoặc dict) theo format ultralytics (train/val/nc/names)
            epochs, batch, imgsz: hyper-params
            save_period: lưu checkpoint mỗi n ep (ultralytics arg: save_period)
            lr: learning rate (optional)
            project/name: nơi lưu
            exist_ok: nếu True, ghi đè project/name nếu có
            resume: nếu True resume từ checkpoint trong project/name/latest
            kwargs: gửi thêm tới self.model.train
        Trả về:
            dict kết quả huấn luyện (như ultralytics trả về)
        """
        # Chuẩn bị args
        train_args = {
            "data": data,
            "epochs": epochs,
            "batch": batch,
            "imgsz": imgsz,
            "save_period": save_period,
            "project": project or str(self.work_dir),
            "name": name or "yolov8_seg_run",
            "exist_ok": exist_ok,
            "resume": resume,
        }
        if lr is not None:
            train_args["lr"] = lr
        train_args.update(kwargs)

        # Gọi API ultralytics
        result = self.model.train(**train_args)
        return result

    def val(self, data: Union[str, Dict[str, Any]], imgsz: int = 640, batch: int = 8, **kwargs) -> dict:
        """
        Đánh giá model trên tập validation.
        """
        val_args = {"data": data, "imgsz": imgsz, "batch": batch}
        val_args.update(kwargs)
        result = self.model.val(**val_args)
        return result

    def predict(
        self,
        source: Union[str, Path],
        save: bool = False,
        save_dir: Optional[Union[str, Path]] = None,
        imgsz: int = 640,
        conf: float = 0.25,
        iou: float = 0.45,
        max_det: int = 100,
        device: Optional[str] = None,
        show: bool = False,
        classes: Optional[Union[int, List[int]]] = None,
        augment: bool = False,
        verbose: bool = False,
        **kwargs,
    ) -> List[Any]:
        """
        Dự đoán segmentation cho image hoặc folder hoặc video.
        Tham số:
            source: đường dẫn tới file/dir hoặc camera (0)
        Trả về:
            results list (objects returned by ultralytics)
        """
        predict_args = {
            "source": str(source),
            "save": save,
            "imgsz": imgsz,
            "conf": conf,
            "iou": iou,
            "max_det": max_det,
            "device": device,
            "show": show,
            "classes": classes,
            "augment": augment,
        }
        predict_args.update(kwargs)

        # Nếu người dùng muốn lưu ảnh đã vẽ, đảm bảo thư mục
        if save:
            if save_dir is None:
                save_dir = self.work_dir / "predict"
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            predict_args["save_txt"] = False
            predict_args["save_dir"] = str(save_dir)

        results = self.model.predict(**predict_args)
        if verbose:
            print(f"[INFO] Predicted {len(results)} items for source={source}")
        return results

    def export(self, format: str = "onnx", imgsz: int = 640, device: Optional[str] = None, simplify: bool = False, **kwargs) -> dict:
        """
        Export model sang format mong muon (onnx, torchscript, coreml, etc.)
        Tham số:
            format: str, ví dụ 'onnx', 'torchscript'
            imgsz: kích thước input
            simplify: nếu True, ultralytics sẽ cố gắng đơn giản hóa onnx
        Trả về:
            dict thông tin export
        """
        export_args = {
            "format": format,
            "imgsz": imgsz,
            "device": device,
            "simplify": simplify,
        }
        export_args.update(kwargs)
        result = self.model.export(**export_args)
        return result

    def load_weights(self, path: str):
        """
        Tải weights mới cho instance model.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Weight file not found: {path}")
        self.model = YOLO(path)
        return self.model

    def save_config(self, config: dict, filename: Union[str, Path]):
        """
        Lưu config (ví dụ dataset yaml) ra file.
        """
        filename = Path(filename)
        filename.parent.mkdir(parents=True, exist_ok=True)
        with open(filename, "w", encoding="utf-8") as f:
            # Nếu dict, lưu bằng yaml-like minimal (json ok)
            json.dump(config, f, ensure_ascii=False, indent=2)
        return str(filename)


# --- Ví dụ sử dụng CLI ---
def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv8 Segmentation wrapper (class-based).")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "val", "predict", "export"], help="Chế độ chạy")
    parser.add_argument("--model", type=str, default="yolov8n-seg.pt", help="Model pretrained hoặc checkpoint")
    parser.add_argument("--data", type=str, default="data.yaml", help="Đường dẫn tới data yaml hoặc COCO json")
    parser.add_argument("--epochs", type=int, default=50, help="Số epoch")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--source", type=str, default="inference/images", help="Source cho predict")
    parser.add_argument("--save", action="store_true", help="Lưu kết quả predict")
    parser.add_argument("--save-dir", type=str, default=None, help="Thư mục lưu predict")
    parser.add_argument("--export-format", type=str, default="onnx", help="Format để export")
    parser.add_argument("--work-dir", type=str, default="runs", help="Thư mục làm việc")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for predict")
    parser.add_argument("--resume", action="store_true", help="Resume training")
    return parser.parse_args()


def _print_header():
    print("=" * 60)
    print("YOLOv8 Segmentation - Wrapper")
    print("=" * 60)


if __name__ == "__main__":
    _print_header()
    args = parse_args()
    model_wrapper = YoloSegModel(model_name_or_path=args.model, work_dir=args.work_dir)

    if args.mode == "train":
        print("Lưu ý: Đảm bảo file data yaml hợp lệ (train/val/nc/names).")
        res = model_wrapper.train(
            data=args.data,
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            resume=args.resume,
            project=args.work_dir,
            name="seg_run",
        )
        print("Training finished. Result summary:", res)

    elif args.mode == "val":
        print("Lưu ý: Đảm bảo model đã được load đúng checkpoint nếu không dùng pretrained.")
        res = model_wrapper.val(data=args.data, imgsz=args.imgsz, batch=max(1, args.batch // 2))
        print("Validation finished. Result summary:", res)

    elif args.mode == "predict":
        print(f"Predict: source={args.source}, save={args.save}")
        results = model_wrapper.predict(
            source=args.source,
            save=args.save,
            save_dir=args.save_dir,
            imgsz=args.imgsz,
            conf=args.conf,
            verbose=True,
        )
        print("Predict finished. Items returned:", len(results))

    elif args.mode == "export":
        print(f"Exporting model to format {args.export_format}")
        res = model_wrapper.export(format=args.export_format, imgsz=args.imgsz)
        print("Export result:", res)

    else:
        print("Chế độ không hợp lệ.")
