import cv2

# Pipeline Settings
MODEL_INPUT_SIZE = (640, 384)
DETECTION_INTERVAL = 3
TEMPORAL_BUFFER_MAXLEN = 45          # Tăng để phân tích temporal tốt hơn (was 30)

# Model / Inference
DEVICE = "cpu"                       # "cpu" hoặc "0" (GPU)
DET_MODEL_PATH = "weights/yolov8n.pt"
POSE_MODEL_PATH = "weights/yolov8n-pose.pt"
DETECT_CLASSES = [0, 2, 3, 5, 7]     # person, car, motorbike, bus, truck
CONF_THRESH = 0.30                   # Giảm nhẹ để recall tốt hơn (was 0.35)
ENABLE_POSE = True

# Event Type Constants
EVENT_TYPE_HUMAN_ACTION = "HUMAN_ACTION"
EVENT_TYPE_VEHICLE_ACCIDENT = "VEHICLE_ACCIDENT"
EVENT_TYPE_SMOKE_FIRE = "SMOKE_FIRE"
EVENT_TYPE_INTRUSION = "INTRUSION"

# Stream URLs
CAMERA_STREAMS = {
    "cam09": "https://dathoc.net/induxvid/cam09/index.m3u8?cookieCheck=1",
    "cam10": "https://dathoc.net/induxvid/cam10/index.m3u8?cookieCheck=1",
}

# MQTT Sensor Config
MQTT_BROKER = "mqtt.dathoc.net"
MQTT_PORT = 1883
MQTT_USER = "test1"
MQTT_PASS = "123456"
MQTT_TOPIC = "smartcity/#"

# ============================================================
# Spatial ROI Rules cho từng Camera.
# MỖI CAMERA CÓ THỂ KHAI BÁO NHIỀU VÙNG (nhiều vùng cấm INTRUSION,
# nhiều vùng rủi ro SMOKE_FIRE, hoặc trộn lẫn).
# "polygon" có thể là list tọa độ hoặc "full_frame".
# ============================================================
def _resolve_polygon(polygon):
    if polygon == "full_frame":
        w, h = MODEL_INPUT_SIZE
        return [[0, 0], [w, 0], [w, h], [0, h]]
    return polygon


CAMERA_ROIS = {
    "cam09": [
        {
            "name": "Restricted_Zone_09_A",
            "event_type": EVENT_TYPE_INTRUSION,
            "polygon": [[50, 50], [300, 50], [300, 300], [50, 300]],
        },
        {
            "name": "Restricted_Zone_09_B",
            "event_type": EVENT_TYPE_INTRUSION,
            "polygon": [[360, 40], [620, 40], [620, 220], [360, 220]],
        },
        {
            "name": "Fire_Risk_Zone_09",
            "event_type": EVENT_TYPE_SMOKE_FIRE,
            "polygon": "full_frame",
        },
    ],
    "cam10": [
        {
            "name": "Fire_Risk_Zone_10",
            "event_type": EVENT_TYPE_SMOKE_FIRE,
            "polygon": [[0, 0], [640, 0], [640, 384], [0, 384]],
        },
        {
            "name": "Restricted_Zone_10",
            "event_type": EVENT_TYPE_INTRUSION,
            "polygon": "full_frame",
        },
    ],
}

for _zones in CAMERA_ROIS.values():
    for _z in _zones:
        _z["polygon"] = _resolve_polygon(_z["polygon"])

# ============================================================
# EVENT CONFIRMATION — Per-Event-Type
# Mỗi loại event cần số frame xác nhận khác nhau.
# Smoke/Fire = 0 vì chúng có persistence logic riêng bên trong.
# ============================================================
EVENT_CONFIRM_FRAMES_DEFAULT = 4
EVENT_CONFIRM_MAP = {
    "HUMAN_FALL": 5,
    "HUMAN_CONFLICT": 4,
    "PERSON_COLLISION": 3,           # Closing speed tính trung bình 3 frame; cửa sổ candidate ngắn khi vật gặp nhau
    "VEHICLE_COLLISION": 1,          # Rule collision đã tự sustained (VEHICLE_COLLISION_SUSTAINED=3); confirm=1 để khỏi đếm kép
    "VEHICLE_STOP_ANOMALY": 3,          # Rule hard_stop đã tự sustained (VEHICLE_HARD_STOP_SUSTAINED=3); confirm thấp để không bỏ sót
    "FIRE_DETECTED": 0,              # Smoke/fire dùng persistence riêng
    "SMOKE_DETECTED": 0,
    "RESTRICTED_INTRUSION": 3,
}

# ============================================================
# FALL DETECTION — Normalized by Person Height
# ============================================================
FALL_ASPECT_RATIO_THRESH = 1.2       # Width/Height > threshold = tư thế nằm ngang
FALL_TORSO_ANGLE_THRESH = 45.0       # Góc thân trên so với trục đứng (degrees)
FALL_VEL_NORM_THRESH = 1.5           # Vận tốc rơi / chiều cao người (normalized)
FALL_ACCEL_NORM_THRESH = 2.0         # Gia tốc rơi / chiều cao người (normalized)
FALL_PERSIST_FRAMES = 4              # Phải nằm ngang >= N detection frames mới xác nhận

# ============================================================
# CONFLICT / FIGHT DETECTION
# ============================================================
CONFLICT_DIST_THRESH = 1.0           # Khoảng cách tương đối tối đa (by avg height)
CONFLICT_KINETIC_THRESH = 12.0       # Kinetic score threshold
CONFLICT_SUSTAINED_FRAMES = 3        # Phải duy trì tín hiệu xung đột >= N frames
CONFLICT_BBOX_JITTER_THRESH = 8.0    # BBox center jitter fallback (khi không có pose)

# ============================================================
# PERSON COLLISION / APPROACH
# ============================================================
PERSON_APPROACH_DIST_THRESH = 0.5    # Khoảng cách tương đối
PERSON_APPROACH_SPEED_THRESH = 35.0  # Tốc độ tiến gần nhau (px/s)

# ============================================================
# VEHICLE COLLISION — Normalized + Multi-Signal
# ============================================================
VEHICLE_PROXIMITY_DIST_RATIO = 0.4   # Dist / avg_diagonal < threshold = gần nhau
VEHICLE_IOU_THRESH = 0.08            # IoU overlap threshold
VEHICLE_DECEL_THRESH = 80.0          # Giảm tốc đột ngột (px/s²)
VEHICLE_DIR_CHANGE_THRESH = 0.8      # Đổi hướng đột ngột (radians)
VEHICLE_CLOSING_SPEED_THRESH = 25.0  # Tốc độ tiến lại gần nhau
VEHICLE_COLLISION_SUSTAINED = 3      # Sustained proximity + anomaly frames

# ============================================================
# VEHICLE HARD STOP
# ============================================================
VEHICLE_HARD_STOP_SPEED_HIGH = 25.0  # Tốc độ trước đó phải >= threshold
VEHICLE_HARD_STOP_SPEED_LOW = 1.5    # Tốc độ hiện tại phải <= threshold
VEHICLE_HARD_STOP_HISTORY = 8        # Cửa sổ lịch sử kiểm tra
VEHICLE_HARD_STOP_SUSTAINED = 3      # Sustained confirmation

# ============================================================
# INTRUSION — Spatial Rules
# ============================================================
INTRUSION_DWELL_FRAMES = 10          # Thời gian lưu lại vùng cấm (was 5)
INTRUSION_MIN_MOVEMENT_PX = 12.0     # Dịch chuyển tối thiểu từ lúc vào vùng (was 8)
INTRUSION_DEPTH_RATIO = 0.15         # Phải vào sâu >= 15% chiều cao người vào trong polygon

# ============================================================
# SMOKE / FIRE — Morphological + Contour + Flicker + Frame Diff
# ============================================================
SMOKE_FIRE_MIN_CONTOUR_AREA = 500    # Diện tích contour tối thiểu (px²)
SMOKE_FIRE_FIRE_PIXEL_THRESH = 400   # Pixel lửa tối thiểu (was 150)
SMOKE_FIRE_SMOKE_PIXEL_THRESH = 800  # Pixel khói tối thiểu (was 300)
SMOKE_FIRE_FIRE_PERSIST_THRESH = 7   # Persistence frames cho lửa (was 5)
SMOKE_FIRE_SMOKE_PERSIST_THRESH = 12 # Persistence frames cho khói (was 8)
SMOKE_FIRE_FLICKER_THRESH = 0.12     # Tỷ lệ thay đổi frame-to-frame cho flicker
SMOKE_FIRE_GROWTH_WINDOW = 5         # Cửa sổ phân tích region growth (was 4)