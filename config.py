import os

# 1. HLS Stream URLs
CAMERA_STREAMS = {
    "cam09": "https://dathoc.net/induxvid/cam09/index.m3u8?cookieCheck=1",
    "cam10": "https://dathoc.net/induxvid/cam10/index.m3u8?cookieCheck=1"
}
RESTRICTED_ROIS = {}
# 2. Optimization Settings
SKIP_FRAMES = 4           # Chạy YOLO mỗi 4 frames (Skip 4 frames giữa)
QUEUE_SIZE = 2              # Giữ buffer ngắn nhất để luôn xử lý realtime
INFERENCE_SIZE = (480,288) # (Width, Height) resize tối ưu tốc độ

# 3. Model Paths
MODEL_DET_PATH = "yolov8n.pt"        # Có thể thay bằng "yolov8n.engine" hoặc "yolov8n.onnx"
MODEL_POSE_PATH = "yolov8n-pose.pt"   # Model Pose keypoints

# 4. MQTT Broker Configuration
MQTT_HOST = "mqtt.dathoc.net"
MQTT_PORT = 1883
MQTT_USER = "test1"
MQTT_PASS = "123456"
MQTT_TOPIC_PREFIX = "smartcity/vision/alerts"
MQTT_COOLDOWN_SEC = 10      # Cooldown 10s cho mỗi loại sự kiện của cùng 1 track ID

# 5. Restricted ROIs (Tọa độ Polygon vùng cấm cho từng camera)
RESTRICTED_ROIS = {
    "cam09": [[100, 100], [400, 100], [400, 400], [100, 400]],
    "cam10": [[200, 150], [500, 150], [500, 450], [200, 450]]
}

# 6. Event Thresholds
INTRUSION_DWELL_FRAMES = 15     # Phải xuất hiện trong vùng cấm >= 15 frames (~0.6s)
FALL_V_Y_THRESH = 1.8           # Vận tốc rơi dọc
FALL_ASPECT_RATIO_THRESH = 1.2  # Tỷ lệ W/H khi nằm
FALL_TORSO_ANGLE_THRESH = 60.0  # Góc nghiêng thân người (độ)
CONFLICT_GATING_DIST = 1.8      # Khoảng cách tối đa để xét xô xát (theo chiều cao)
CONFLICT_SCORE_THRESH = 0.70    # Ngưỡng điểm xô xát (đã nâng lên 0.70 để bớt tin giả)
ACCIDENT_DECEL_THRESH = -3.5   # Gia tốc âm (giảm tốc) đột ngột của xe