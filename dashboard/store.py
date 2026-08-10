"""AlertStore — giữ ảnh + danh sách cảnh báo trong BỘ NHỚ cho dashboard.

Mỗi khi 1 alert được xác nhận lần đầu, CameraWorker gọi record():
  - Mã hoá frame có event (snapshot toàn frame + snapshot CẮT vùng bbox)
    thành JPEG ngay trong bộ nhớ (KHÔNG ghi file ra đĩa).
  - Ghi 1 dòng log cảnh báo vào file alerts.log (kèm vị trí bbox).
  - Thêm metadata alert (kèm key ảnh trong bộ nhớ) vào hàng đợi để
    dashboard phục vụ ảnh TRỰC TIẾP qua API.
Thread-safe: nhiều CameraWorker chạy song song ghi đồng thời.
"""
import os
import time
import threading
from collections import deque, OrderedDict

import cv2
import numpy as np

import config


class AlertStore:
    def __init__(self, snapshot_dir, max_alerts=200):
        self.snapshot_dir = snapshot_dir
        os.makedirs(snapshot_dir, exist_ok=True)
        self.alerts = deque(maxlen=max_alerts)
        self._images = OrderedDict()  # key ảnh -> JPEG bytes (trong bộ nhớ)
        self._lock = threading.Lock()
        self._seq = 0
        self.log_file = os.path.join(snapshot_dir, config.ALERT_LOG_FILE)

    def record(self, camera_id, event, frame_bgr):
        """Mã hoá ảnh vào bộ nhớ + thêm alert cho dashboard.

        Args:
            camera_id: ID camera.
            event: dict event (event_type, description, confidence, bbox...).
            frame_bgr: frame BGR hiện tại (đã annotate) để làm bằng chứng.

        Returns:
            dict alert đã lưu (kèm key ảnh full + crop trong bộ nhớ).
        """
        if frame_bgr is None:
            return None

        ts = time.time()
        ev_type = event.get("event_type", "UNKNOWN")
        bbox = event.get("bbox")

        with self._lock:
            self._seq += 1
            snap_key = f"alert_{self._seq}.jpg"
            crop_key = f"alert_{self._seq}_crop.jpg"

        try:
            ok, buf = cv2.imencode(
                ".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            full_bytes = buf.tobytes()
            crop_img = self._crop_bbox(frame_bgr, bbox)
            if crop_img is not None:
                ok_c, buf_c = cv2.imencode(
                    ".jpg", crop_img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                crop_bytes = buf_c.tobytes() if ok_c else None
            else:
                crop_bytes = None
        except Exception:
            ok = False
            full_bytes = b""
            crop_bytes = None
        if not ok or not full_bytes:
            return None

        alert = {
            "id": self._seq,
            "timestamp": ts,
            "time_str": time.strftime("%H:%M:%S", time.localtime(ts)),
            "camera_id": camera_id,
            "event_type": ev_type,
            "description": event.get("description", ""),
            "confidence": round(float(event.get("confidence", 0.0)), 3),
            "track_ids": list(event.get("track_ids", [])),
            "zone_name": event.get("zone_name", ""),
            "bbox": bbox,
            "evidence_objects": list(event.get("evidence_objects", [])),
            "snapshot": snap_key,
            "crop": crop_key if crop_bytes is not None else None,
        }

        with self._lock:
            self.alerts.append(alert)
            self._images[snap_key] = full_bytes
            if crop_bytes is not None:
                self._images[crop_key] = crop_bytes
            # Giới hạn bộ nhớ: ảnh cũ nhất bị loại khi vượt 2 ảnh/alert
            while len(self._images) > self.alerts.maxlen * 2:
                self._images.popitem(last=False)

        self._write_log_line(alert)
        return alert

    def get_image(self, key):
        """Trả JPEG bytes của ảnh (full/crop) theo key; None nếu không có."""
        with self._lock:
            return self._images.get(key)

    def _crop_bbox(self, frame_bgr, bbox):
        """Cắt vùng quanh bbox cảnh báo (mở rộng theo margin) để làm bằng chứng."""
        if bbox is None:
            return None
        try:
            x1, y1, x2, y2 = [int(round(float(v))) for v in np.ravel(bbox)]
        except (TypeError, ValueError):
            return None
        if x2 <= x1 or y2 <= y1:
            return None

        h, w = frame_bgr.shape[:2]
        bw, bh = x2 - x1, y2 - y1
        margin = config.SNAPSHOT_CROP_MARGIN
        pad_x = int(bw * margin)
        pad_y = int(bh * margin)

        cx1 = max(0, x1 - pad_x)
        cy1 = max(0, y1 - pad_y)
        cx2 = min(w, x2 + pad_x)
        cy2 = min(h, y2 + pad_y)
        if cx2 <= cx1 or cy2 <= cy1:
            return None
        return frame_bgr[cy1:cy2, cx1:cx2].copy()

    def _write_log_line(self, alert):
        """Ghi 1 dòng log cảnh báo (có vị trí bbox) vào alerts.log."""
        bbox = alert["bbox"]
        bbox_str = "none" if bbox is None else ",".join(str(int(round(float(v)))) for v in np.ravel(bbox))
        evidence = alert.get("evidence_objects") or []
        evidence_str = "none"
        if evidence:
            evidence_str = ";".join(
                f"id={obj.get('track_id')} cls={obj.get('cls_id')} speed={obj.get('speed')}"
                for obj in evidence
            )
        line = (
            f"[{alert['time_str']}] [{alert['camera_id']}] "
            f"{alert['event_type']} | conf={alert['confidence']} | "
            f"bbox=[{bbox_str}] | {alert['description']} | "
            f"evidence={evidence_str} | "
            f"snapshot={alert['snapshot']} crop={alert['crop']}\n"
        )
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(line)
        except Exception:
            pass

    def recent(self, limit=100):
        """Danh sách alert gần nhất (mới nhất trước) cho dashboard."""
        with self._lock:
            items = list(self.alerts)[-limit:]
        return list(reversed(items))

    def count(self):
        with self._lock:
            return len(self.alerts)
