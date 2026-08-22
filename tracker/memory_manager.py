from typing import Any

import numpy as np

import config
import geometry
from tracker.object_state import TrackedObjectState


class ObjectMemoryManager:
    """Quản lý danh sách các object đang active của 1 CAMERA và gán track_id.

    Rule 1/2: YOLO shared KHÔNG giữ tracker nội bộ; mọi ID track được gán
    ở đây theo per-camera (IoU association giữa các detection frame).
    """

    def __init__(self, maxlen: int = 30, ttl_frames: int = 30):
        self.objects: dict[int, TrackedObjectState] = {}
        self.maxlen = maxlen
        self.ttl_frames = ttl_frames
        self.current_frame = 0
        self._next_id = 1

    def _assign_id(self, det: dict[str, Any]) -> int:
        """Gán track_id cho 1 detection (nếu trùng IoU với track cũ).

        Dùng IoU KHÔNG ngưỡng quá cao (AI 5 FPS — bbox có thể dịch nhiều giữa
        2 frame detect). Kèm fallback theo khoảng cách tâm hợp lệ để giữ
        track ID ổn định khi bbox dịch nhanh (đánh nhau/va chạm).
        """
        bbox = det["bbox"]
        center = geometry.bbox_center(bbox)
        best_id: int | None = None
        best_score = 0.0
        for t_id, obj in self.objects.items():
            if obj.missed_frames >= config.DETECTION_INTERVAL:
                continue  # track quá cũ -> không tái sử dụng id
            ref = obj.predicted_bbox if obj.predicted_bbox is not None else obj.bbox_history[-1]
            if ref is None:
                continue
            iou = geometry.iou(ref, bbox)
            ref_center = geometry.bbox_center(ref)
            diag = geometry.bbox_diagonal(ref)
            d = float(np.linalg.norm(center - ref_center)) / max(diag, 1e-5)
            center_score = 1.0 - min(1.0, d / 0.6)
            # IoU mạnh (>0.05) thắng; ngược lại dựa vào khoảng cách tâm.
            score = iou if iou >= 0.05 else 0.25 * center_score
            if score > best_score:
                best_score = score
                best_id = t_id

        if best_id is not None and best_score >= config.TRACK_ASSOC_IOU:
            return best_id

        tid = self._next_id
        self._next_id += 1
        return tid

    def update_detections(
        self,
        detections: list[dict[str, Any]],
        frame_idx: int,
        timestamp: float,
    ) -> list[dict[str, Any]]:
        """Gán track_id (Không cần detector giữ tracker) + cập nhật history.

        Args:
            detections: list dict {cls_id, bbox, conf, pose} (KHÔNG có track_id).
        """
        self.current_frame = frame_idx
        updated_ids: set[int] = set()

        for det in detections:
            t_id = self._assign_id(det)
            det["track_id"] = t_id
            bbox = det["bbox"]
            cls_id = det["cls_id"]
            pose = det.get("pose", None)

            if t_id not in self.objects:
                self.objects[t_id] = TrackedObjectState(t_id, cls_id, maxlen=self.maxlen)
            obj = self.objects[t_id]
            obj.update(bbox, timestamp, pose)
            obj.conf = det.get("conf", 0.0)   # lưu confidence để gate collision pipeline
            obj.missed_frames = 0
            obj.last_updated_frame = frame_idx
            obj.last_update_predicted = False
            obj.predicted_bbox = None
            updated_ids.add(t_id)

        # Tăng missed_frames cho object không xuất hiện trong frame này
        for t_id, obj in self.objects.items():
            if t_id not in updated_ids:
                obj.missed_frames += 1

        # Cleanup lost objects
        dead_ids = [
            t_id for t_id, obj in self.objects.items()
            if self.current_frame - obj.last_updated_frame > self.ttl_frames
        ]
        for t_id in dead_ids:
            del self.objects[t_id]

        return detections

    def visible_objects(self) -> dict[int, TrackedObjectState]:
        """Chỉ trả về object còn bám dấu (detect thật gần đây); loại object ma."""
        return {
            t_id: obj for t_id, obj in self.objects.items()
            if len(obj.bbox_history) > 0 and obj.missed_frames < config.DETECTION_INTERVAL
        }
