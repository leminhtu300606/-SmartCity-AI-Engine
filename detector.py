import numpy as np
from ultralytics import YOLO
import config


class YOLODetector:
    """YOLO detector + ByteTrack (ultralytics) + pose keypoints.

    Trả về list track dict: {track_id, cls_id, bbox [x1,y1,x2,y2], conf, pose}.
    Pose keypoints (17,3) chỉ gắn cho Person (cls 0) khi ENABLE_POSE=True.
    """

    def __init__(self, device=None):
        self.device = device or config.DEVICE
        self.det_model = YOLO(config.DET_MODEL_PATH)
        self.pose_model = YOLO(config.POSE_MODEL_PATH) if config.ENABLE_POSE else None
        self.classes = config.DETECT_CLASSES
        self.conf = config.CONF_THRESH
        # ultralytics imgsz nhận (height, width)
        self.imgsz = (config.MODEL_INPUT_SIZE[1], config.MODEL_INPUT_SIZE[0])

    def detect(self, frame_bgr):
        """Chạy detector + tracker trên frame -> danh sách track."""
        results = self.det_model.track(
            frame_bgr,
            persist=True,
            imgsz=self.imgsz,
            classes=self.classes,
            conf=self.conf,
            tracker="bytetrack.yaml",
            verbose=False,
        )

        if not results or results[0].boxes is None or results[0].boxes.id is None:
            return []

        boxes = results[0].boxes.xyxy.cpu().numpy()
        cls_ids = results[0].boxes.cls.cpu().numpy().astype(int)
        confs = results[0].boxes.conf.cpu().numpy()
        track_ids = results[0].boxes.id.cpu().numpy().astype(int)

        poses = None
        if self.pose_model is not None:
            poses = self._match_pose(frame_bgr, boxes, cls_ids)

        tracks = []
        for i, t_id in enumerate(track_ids):
            tracks.append({
                "track_id": int(t_id),
                "cls_id": int(cls_ids[i]),
                "bbox": [float(v) for v in boxes[i]],
                "conf": float(confs[i]),
                "pose": None if poses is None else poses[i],
            })
        return tracks

    def _match_pose(self, frame_bgr, det_boxes, cls_ids):
        """Chạy pose model và khớp keypoint vào box person của det model qua IoU."""
        pose_results = self.pose_model(frame_bgr, imgsz=self.imgsz, conf=self.conf, verbose=False)
        poses = [None] * len(det_boxes)
        if not pose_results or pose_results[0].keypoints is None:
            return poses
        if pose_results[0].boxes is None:
            return poses

        kp_all = pose_results[0].keypoints.data.cpu().numpy()
        pb_all = pose_results[0].boxes.xyxy.cpu().numpy()
        if len(pb_all) == 0:
            return poses

        for i, box in enumerate(det_boxes):
            if cls_ids[i] != 0:
                continue
            ious = [self._iou(box, b) for b in pb_all]
            best = int(np.argmax(ious))
            if ious[best] > 0.5:
                poses[i] = kp_all[best]
        return poses

    @staticmethod
    def _iou(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        inter = max(0, xB - xA) * max(0, yB - yA)
        areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        return inter / float(areaA + areaB - inter + 1e-5)
