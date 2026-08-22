import numpy as np
from ultralytics import YOLO
import config
import geometry


class YOLODetector:
    """YOLO detector STATELESS — 1 instance dùng chung cho MỌI camera (Rule 1).

    Khác bản cũ:
      - predict() KHÔNG track/persist (ultralytics `track` giữ tracker nội bộ
        trong model → không an toàn khi dùng chung model giữa các camera).
      - Track ID được gán theo CAMERA bởi ObjectMemoryManager (IoU association),
        hoàn toàn tách khỏi model.
      - Pose & Vehicle-Cls là Level-3 (specialized): KHÔNG chạy trong detect(),
        pipeline gọi riêng khi có candidate phù hợp (Rule 4).
    """

    def __init__(self, device: str | None = None):
        self.device = device or config.DEVICE
        self.det_model = YOLO(config.DET_MODEL_PATH)
        self.pose_model = YOLO(config.POSE_MODEL_PATH) if config.ENABLE_POSE else None
        self.classes = config.DETECT_CLASSES
        self.conf = config.CONF_THRESH
        # ultralytics imgsz nhận (height, width)
        self.imgsz = (config.MODEL_INPUT_SIZE[1], config.MODEL_INPUT_SIZE[0])

    def predict(self, frame_bgr: np.ndarray) -> list[dict]:
        """STATELESS detect: trả list detection dict {cls_id, bbox, conf}.

        KHÔNG gán track_id. KHÔNG chạy pose / vehicle-cls (Level 1 chỉ).
        """
        results = self.det_model.predict(
            frame_bgr,
            imgsz=self.imgsz,
            classes=self.classes,
            conf=self.conf,
            verbose=False,
        )
        if not results or results[0].boxes is None:
            return []

        boxes = results[0].boxes.xyxy.cpu().numpy()
        cls_ids = results[0].boxes.cls.cpu().numpy().astype(int)
        confs = results[0].boxes.conf.cpu().numpy()

        dets: list[dict] = []
        for i in range(len(boxes)):
            dets.append({
                "cls_id": int(cls_ids[i]),
                "bbox": [float(v) for v in boxes[i]],
                "conf": float(confs[i]),
                "pose": None,
            })

        # Loại bỏ nhiễu tin cậy thấp theo cặp người/xe trong tracking (giữ logic cũ)
        dets = self._remove_persons_inside_vehicles(dets)
        dets = self._remove_vehicles_inside_persons(dets)
        return dets

    def predict_pose(self, frame_bgr: np.ndarray, dets: list[dict]) -> bool:
        """Level-3: gắn pose keypoints cho track NGƯỜI đã có (chỉ gọi khi cần).

        Rule 4: pose là SPECIALIZED model — chỉ chạy khi có candidate người
        tư thế NGÃ (aspect ngang). Người đứng bình thường KHÔNG chạy pose
        (tiết kiệm ~100ms video). Trả True nếu đã chạy.
        """
        if self.pose_model is None:
            return False
        persons = [d for d in dets if d["cls_id"] == 0]
        # Gate L3: chỉ có tư thế ngã (aspect>ngưỡng) mới cần pose chi tiết
        suspects = [
            p for p in persons
            if (p["bbox"][2] - p["bbox"][0]) / max(p["bbox"][3] - p["bbox"][1], 1e-5)
            > config.FALL_ASPECT_RATIO_THRESH * 0.8
        ]
        if not suspects:
            return False

        try:
            pose_results = self.pose_model(
                frame_bgr, imgsz=self.imgsz, conf=self.conf, verbose=False)
        except Exception:
            return False
        if not pose_results or pose_results[0].keypoints is None:
            return False
        if pose_results[0].boxes is None:
            return False

        kp_all = pose_results[0].keypoints.data.cpu().numpy()
        pb_all = pose_results[0].boxes.xyxy.cpu().numpy()
        if len(pb_all) == 0:
            return False

        for person in suspects:
            ious = [geometry.iou(person["bbox"], b) for b in pb_all]
            best = int(np.argmax(ious))
            if ious[best] > 0.5:
                person["pose"] = kp_all[best]
        return True

    def _remove_persons_inside_vehicles(self, dets: list[dict]) -> list[dict]:
        """Bỏ detection NGƯỜI nằm >= ngưỡng diện tích trong bbox xe."""
        vehicles = [d for d in dets if d["cls_id"] in config.VEHICLE_CLASSES]
        persons = [d for d in dets if d["cls_id"] == 0]
        if not vehicles or not persons:
            return dets

        keep = []
        for p in persons:
            drop = False
            for v in vehicles:
                if (geometry.containment_ratio(p["bbox"], v["bbox"])
                        > config.VEHICLE_CLS_IN_VEHICLE_REMOVE_RATIO
                        and geometry.box_area(p["bbox"]) < geometry.box_area(v["bbox"]) * 0.5):
                    drop = True
                    break
            if not drop:
                keep.append(p)
        return [d for d in dets if d["cls_id"] != 0] + keep

    def _remove_vehicles_inside_persons(self, dets: list[dict]) -> list[dict]:
        """Bỏ detection XE có bbox gần như trọn bên trong bbox NGƯỜI."""
        persons = [d for d in dets if d["cls_id"] == 0]
        vehicles = [d for d in dets if d["cls_id"] in config.VEHICLE_CLASSES]
        if not persons or not vehicles:
            return dets

        keep = []
        for v in vehicles:
            drop = False
            for p in persons:
                if (geometry.containment_ratio(v["bbox"], p["bbox"]) > 0.85
                        and geometry.box_area(v["bbox"]) < geometry.box_area(p["bbox"]) * 0.3):
                    drop = True
                    break
            if not drop:
                keep.append(v)
        return [d for d in dets if d["cls_id"] == 0] + keep
