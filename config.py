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
        # cam09 tập trung nhận diện HÀNH VI CON NGƯỜI (ngã, xô xát, va chạm)
        # thay vì vùng cấm không gian (restricted zone đã bỏ).
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
# VEHICLE COLLISION — Accident Classifier (Multi-Feature)
# Không chỉ dựa vào bbox overlap: xe chạy sát nhau / bị che khuất
# cũng tạo overlap giả. Kết hợp nhiều đặc trưng quỹ đạo.
# ============================================================
VEHICLE_PROXIMITY_DIST_RATIO = 0.55  # Dist / avg_diagonal < threshold = gần nhau (nhạy hơn)
VEHICLE_IOU_THRESH = 0.05            # IoU overlap threshold (nhạy hơn)
VEHICLE_DECEL_THRESH = 60.0          # Giảm tốc đột ngột (px/s²)
VEHICLE_DIR_CHANGE_THRESH = 0.6      # Đổi hướng đột ngột (radians)
VEHICLE_CLOSING_SPEED_THRESH = 18.0  # Tốc độ tiến lại gần nhau
VEHICLE_COLLISION_SUSTAINED = 2      # Sustained proximity + anomaly frames

# Accident Classifier: điểm tổng hợp từ nhiều đặc trưng
VEHICLE_DIST_DROP_THRESH = 7.0       # Khoảng cách giảm nhanh giữa 2 frame (px)
VEHICLE_COLLISION_SCORE_THRESH = 0.42  # Điểm tai nạn tối thiểu (0..1)
VEHICLE_POST_CONTACT_WINDOW = 8      # Số frame sau tiếp xúc để check hậu va chạm
VEHICLE_COLLISION_WEIGHTS = {
    "proximity": 0.30,   # 2 xe tiến rất gần / bbox giao nhau (gate bắt buộc)
    "closing": 0.15,     # Khoảng cách giảm nhanh (closing velocity)
    "dist_drop": 0.15,   # Khoảng cách tụt nhanh frame-to-frame
    "direction": 0.15,   # Đổi hướng / chuyển động đột ngột
    "decel": 0.10,       # Giảm tốc đột ngột
    "post_contact": 0.15 # Dừng/đổi hướng bất thường ngay sau tiếp xúc
}

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
# Độ nhạy tăng: ngưỡng thấp hơn → phát hiện sớm hơn (có thể nhạy hơn với nhiễu)
# ============================================================
SMOKE_FIRE_MIN_CONTOUR_AREA = 300    # Diện tích contour tối thiểu (px²)
SMOKE_FIRE_FIRE_PIXEL_THRESH = 250   # Pixel lửa tối thiểu (was 400)
SMOKE_FIRE_SMOKE_PIXEL_THRESH = 500  # Pixel khói tối thiểu (was 800)
SMOKE_FIRE_FIRE_PERSIST_THRESH = 5   # Persistence frames cho lửa (was 7)
SMOKE_FIRE_SMOKE_PERSIST_THRESH = 8  # Persistence frames cho khói (was 12)
SMOKE_FIRE_FLICKER_THRESH = 0.08     # Tỷ lệ thay đổi frame-to-frame cho flicker
SMOKE_FIRE_GROWTH_WINDOW = 5         # Cửa sổ phân tích region growth (was 4)

# Fire: độ sáng tương phản cao so với nền tối + tần số nhấp nháy đặc thù lửa thật
SMOKE_FIRE_CONTRAST_THRESH = 40.0          # Chênh lệch mean(V) giữa lửa và nền xung quanh
SMOKE_FIRE_FLICKER_FREQ_THRESH = 0.25      # Tần số dao động nhấp nháy lửa (0..1, lửa thật cao)
SMOKE_FIRE_FLICKER_FREQ_WINDOW = 10        # Cửa sổ mẫu để tính tần số flicker

# Fire: phân loại đối tượng (lửa thật vs vật thể màu tương tự: đèn, mây...)
SMOKE_FIRE_FIRE_EDGE_IRREGULARITY_THRESH = 1.08  # Cạnh sắc tối thiểu; đèn tròn mượt ≈1.0
SMOKE_FIRE_FIRE_JAGGED_THRESH = 1.20            # Cạnh răng cưa rõ rệt (ngọn lửa thật)
SMOKE_FIRE_FIRE_CLASS_SCORE_THRESH = 0.40        # Điểm phân loại lửa tối thiểu (0..1)

# Smoke: dạng mây mờ lan tỏa, đổi hình dạng + độ trong suốt theo thời gian
SMOKE_FIRE_SMOKE_SOFT_GRAD_THRESH = 50.0   # Gradient nội tại < ngưỡng = pixel khói mờ (px)
SMOKE_FIRE_SMOKE_SOFTNESS_THRESH = 0.40    # Tỷ lệ pixel mờ tối thiểu trong vùng khói
SMOKE_FIRE_SMOKE_SHAPE_CHANGE_THRESH = 0.18  # Tỷ lệ hình dạng khói thay đổi giữa 2 frame
SMOKE_FIRE_SMOKE_CHANGE_THRESH = 0.015     # Frame diff tối thiểu cho tín hiệu khói

# Khói giai đoạn đầu: phát hiện sớm trước khi thấy rõ ngọn lửa -> tăng thời gian phản ứng
SMOKE_FIRE_EARLY_SMOKE_PIXEL_THRESH = 250  # Pixel tối thiểu cho khói giai đoạn đầu
SMOKE_FIRE_EARLY_SMOKE_PERSIST_THRESH = 3  # Persistence khói sớm (thấp hơn -> confirm nhanh hơn)