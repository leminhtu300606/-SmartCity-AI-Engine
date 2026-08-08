import os

# 1. Model Paths (weights folder)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DET_PATH = os.path.join(BASE_DIR, "weights", "yolov8n.pt")
MODEL_POSE_PATH = os.path.join(BASE_DIR, "weights", "yolov8n-pose.pt")

# 2. HLS Stream URLs
CAMERA_STREAMS = {
    "cam09": "https://dathoc.net/induxvid/cam09/index.m3u8?cookieCheck=1",
    "cam10": "https://dathoc.net/induxvid/cam10/index.m3u8?cookieCheck=1",
}

# 3. Restricted ROIs (Danh sách vùng cấm cho từng camera, mỗi vùng có tên + polygon tọa độ)
# Dạng hợp lệ: [{"name": "ZONE_1", "polygon": [[x,y], ...]}, ...]
# Vẫn chấp nhận dạng cũ (polygon trần) — helper get_roi_zones sẽ tự chuẩn hóa.
RESTRICTED_ROIS = {
    "cam09": [
        {"name": "ZONE_1", "polygon": [[100, 100], [400, 100], [400, 400], [100, 400]]},
        {"name": "ZONE_2", "polygon": [[450, 50], [600, 50], [600, 300], [450, 300]]},
    ],
    "cam10": [
        {"name": "ZONE_1", "polygon": [[200, 150], [500, 150], [500, 450], [200, 450]]},
    ],
}


def get_roi_zones(camera_id):
    """Chuẩn hóa cấu hình RESTRICTED_ROIS thành danh sách zone {"name", "polygon"}.
    Tương thích ngược với config cũ dạng polygon trần (1 vùng không tên)."""
    raw = RESTRICTED_ROIS.get(camera_id)
    if not raw:
        return []
    if isinstance(raw[0], dict):
        return list(raw)
    return [{"name": "ROI", "polygon": raw}]

# 4. Optimization Settings
SKIP_FRAMES = 4           # Chạy YOLO mỗi 4 frames (Skip 4 frames giữa)
QUEUE_SIZE = 2            # Giữ buffer ngắn nhất để luôn xử lý realtime
INFERENCE_SIZE = (480, 288)  # (Width, Height) resize tối ưu tốc độ

# 5. MQTT Broker Configuration
MQTT_HOST = "mqtt.dathoc.net"
MQTT_PORT = 1883
MQTT_USER = "test1"
MQTT_PASS = "123456"
MQTT_TOPIC_PREFIX = "smartcity/vision/alerts"
MQTT_COOLDOWN_SEC = 10    # Cooldown 10s cho mỗi loại sự kiện của cùng 1 track ID

# 6. Event Thresholds
INTRUSION_DWELL_FRAMES = 15     # Phải xuất hiện trong vùng cấm >= 15 frames (~0.6s)
FALL_V_Y_THRESH = 1.8           # Vận tốc rơi dọc
FALL_ASPECT_RATIO_THRESH = 1.2  # Tỷ lệ W/H khi nằm
FALL_TORSO_ANGLE_THRESH = 60.0  # Góc nghiêng thân người (độ)
CONFLICT_GATING_DIST = 1.8      # Khoảng cách tối đa để xét xô xát (theo chiều cao)
CONFLICT_SCORE_THRESH = 0.70    # Ngưỡng điểm xô xát
ACCIDENT_DECEL_THRESH = -3.5    # Gia tốc âm (giảm tốc) đột ngột của xe