"""SharedModelRegistry — Rule 1 & 2 & 11 của AI Performance Contract.

1 model = 1 instance/process (singleton). Camera KHÔNG được tự khởi tạo model;
mọi camera đều đi qua registry này để lấy chung 1 instance.

Giới hạn CPU threads (Rule 11): PyTorch/OpenMP/MKL bị giới hạn để model
KHÔNG tự chiếm 12 logical thread của máy.
"""
from __future__ import annotations

import os
import threading

import config

_lock = threading.Lock()
_detector: YOLODetector | None = None
_vehicle_cls: VehicleTypeClassifier | None = None
_infer_lock = threading.Lock()   # serial hoá predict trên model SHARED (thread-safe)


def get_inference_lock() -> threading.Lock:
    """Khóa chung cho mọi cuộc gọi predict — 1 model = 1 predict tại 1 thời điểm.

    Ultralytics model KHÔNG thread-safe; dùng chung 1 instance giữa các camera
    bắt buộc phải serial hoá (Rule 1). CPU giới hạn nên throughput vẫn ổn.
    """
    return _infer_lock


def limit_cpu_threads() -> None:
    """Rule 11: giới hạn PyTorch/OpenMP/MKL bằng AI_MAX_THREADS.

    Phải gọi TRƯỚC khi load bất kỳ model nào (thread pools của torch
    được tạo lúc khởi tạo).
    """
    n = int(getattr(config, "AI_MAX_THREADS", 4))
    interop = int(getattr(config, "AI_MAX_INTEROP_THREADS", 1))
    os.environ["OMP_NUM_THREADS"] = str(n)
    os.environ["OPENBLAS_NUM_THREADS"] = str(n)
    os.environ["MKL_NUM_THREADS"] = str(n)
    os.environ["NUMEXPR_NUM_THREADS"] = str(n)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(n)
    try:
        import torch

        torch.set_num_threads(n)
        try:
            torch.set_num_interop_threads(interop)
        except RuntimeError:
            pass  # đã set rồi -> bỏ qua
    except Exception:
        pass
    print(f"[AI] Giới hạn CPU threads: torch/OpenMP/MKL = {n} thread "
          f"(total logical = {getattr(config, 'CPU_LOGICAL_THREADS', 12)})")


def get_detector() -> YOLODetector:
    """Trả DUY NHẤT 1 YOLODetector dùng chung cho mọi camera (Rule 1)."""
    global _detector
    if _detector is None:
        with _lock:
            if _detector is None:
                from detector import YOLODetector

                _detector = YOLODetector()
                print("[AI] YOLO detector: 1 instance dùng chung cho mọi camera")
    return _detector


def get_vehicle_classifier() -> VehicleTypeClassifier:
    """Trả DUY NHẤT 1 VehicleTypeClassifier (YOLO-cls) dùng chung (Rule 1)."""
    global _vehicle_cls
    if _vehicle_cls is None:
        with _lock:
            if _vehicle_cls is None:
                from vehicle_classifier import VehicleTypeClassifier

                _vehicle_cls = VehicleTypeClassifier()
                if _vehicle_cls.available:
                    print("[AI] Vehicle classifier: 1 instance dùng chung cho mọi camera")
                else:
                    print("[AI] Vehicle classifier: model chưa có weights, vô hiệu hoá")
    return _vehicle_cls