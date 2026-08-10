import numpy as np
from ultralytics import YOLO
from vehicle_classifier import VehicleTypeClassifier
import config


class YOLODetector:
    """YOLO detector + ByteTrack (ultralytics) + pose keypoints.

    Trả về list track dict: {track_id, cls_id, bbox [x1,y1,x2,y2], conf, pose}.
    Pose keypoints (17,3) chỉ gắn cho Person (cls 0) khi ENABLE_POSE=True.
    Track là PHƯƠNG TIỆN (cls_id thuộc VEHICLE_CLASSES) còn được gán thêm
    "vehicle_type" (tên tiếng Việt tinh: xe máy, xe tải, xe chở dầu...) nếu
    model phân loại tinh khả dụng.
    """

    def __init__(self, device=None):
        self.device = device or config.DEVICE
        self.det_model = YOLO(config.DET_MODEL_PATH)
        self.pose_model = YOLO(config.POSE_MODEL_PATH) if config.ENABLE_POSE else None
        self.vehicle_cls = VehicleTypeClassifier()
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

        # Phân biệt RÕ người / xe: bỏ track NGƯỜI nằm gần như trọn trong bbox
        # xe (tài xế/hành khách trong cabin) để không nhầm lẫn với người đi bộ.
        tracks = self._remove_persons_inside_vehicles(tracks)
        tracks = self._remove_vehicles_inside_persons(tracks)

        # Phân loại tinh loại xe (xe máy/xe tải/xe chở dầu...) cho phương tiện.
        # Truyền track_id để bộ đếm đa số theo thời gian chống che khuất.
        if self.vehicle_cls.available:
            for tr in tracks:
                if tr["cls_id"] not in config.VEHICLE_CLASSES:
                    continue
                vtype, vconf = self.vehicle_cls.classify(
                    frame_bgr, tr["bbox"], track_id=tr["track_id"]
                )
                if vtype is not None and vconf >= config.VEHICLE_CLS_MIN_CONF:
                    tr["vehicle_type"] = vtype
            self.vehicle_cls.prune_votes([t["track_id"] for t in tracks])
        return tracks

    def _remove_persons_inside_vehicles(self, tracks):
        """Bỏ track NGƯỜI (cls 0) nằm >= ngưỡng diện tích trong bbox của xe.

        Người đang ngồi/lái xe bên trong ô tô/xe tải bị detect như person
        riêng → gây trùng/hiểu nhầm là người đi bộ khi xe lưu thông. Bỏ để
        phân biệt RÕ người (đi bộ, đứng ngoài đường) với xe.
        """
        vehicles = [t for t in tracks if t["cls_id"] in config.VEHICLE_CLASSES]
        persons = [t for t in tracks if t["cls_id"] == 0]
        if not vehicles or not persons:
            return tracks

        def area(box):
            return max(box[2] - box[0], 0) * max(box[3] - box[1], 0)

        def contained_ratio(person_box, vehicle_box):
            a = max(person_box[0], vehicle_box[0])
            b = max(person_box[1], vehicle_box[1])
            c = min(person_box[2], vehicle_box[2])
            d = min(person_box[3], vehicle_box[3])
            inter = max(0, c - a) * max(0, d - b)
            p_area = area(person_box)
            return inter / max(p_area, 1e-5)

        keep_ids = set(t["track_id"] for t in tracks)
        for p in persons:
            for v in vehicles:
                v_area = area(v["bbox"])
                p_area = area(p["bbox"])
                # Người bên trong xe: bị phủ phần lớn + nhỏ hơn hẳn vehicle
                if (contained_ratio(p["bbox"], v["bbox"]) > config.VEHICLE_CLS_IN_VEHICLE_REMOVE_RATIO
                        and p_area < v_area * 0.5):
                    keep_ids.discard(p["track_id"])
                    break
        return [t for t in tracks if t["track_id"] in keep_ids]

    @staticmethod
    def _remove_vehicles_inside_persons(tracks):
        """Bỏ track XE có bbox gần như trọn bên trong bbox NGƯỜI.

        Phòng trường hợp ngược lại (tín hiệu tracking nhiễu): xe nhỏ bị gộp
        vào người đang dắt qua → giữ nhãn người cho rõ.
        """
        persons = [t for t in tracks if t["cls_id"] == 0]
        vehicles = [t for t in tracks if t["cls_id"] in config.VEHICLE_CLASSES]
        if not persons or not vehicles:
            return tracks

        def area(box):
            return max(box[2] - box[0], 0) * max(box[3] - box[1], 0)

        def contained_ratio(small_box, big_box):
            a = max(small_box[0], big_box[0])
            b = max(small_box[1], big_box[1])
            c = min(small_box[2], big_box[2])
            d = min(small_box[3], big_box[3])
            inter = max(0, c - a) * max(0, d - b)
            s_area = area(small_box)
            return inter / max(s_area, 1e-5)

        keep_ids = set(t["track_id"] for t in tracks)
        for v in vehicles:
            for p in persons:
                p_area = area(p["bbox"])
                v_area = area(v["bbox"])
                if (contained_ratio(v["bbox"], p["bbox"]) > 0.85
                        and v_area < p_area * 0.3):
                    keep_ids.discard(v["track_id"])
                    break
        return [t for t in tracks if t["track_id"] in keep_ids]

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
