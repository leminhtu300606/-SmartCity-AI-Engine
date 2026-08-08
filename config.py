import cv2

# Pipeline Settings
MODEL_INPUT_SIZE = (640, 384)
DETECTION_INTERVAL = 3
TEMPORAL_BUFFER_MAXLEN = 30

# Model / Inference
DEVICE = "cpu"                 # "cpu" hoặc "0" (GPU)
DET_MODEL_PATH = "weights/yolov8n.pt"
POSE_MODEL_PATH = "weights/yolov8n-pose.pt"
DETECT_CLASSES = [0, 2, 3, 5, 7]  # person, car, motorbike, bus, truck
CONF_THRESH = 0.35
ENABLE_POSE = True

# Event Type Constants (Khắc phục lỗi AttributeError)
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

# Spatial ROI Rules cho Cam09 và Cam10
CAMERA_ROIS = {
    "cam09": [
        {
            "name": "Restricted_Zone_09",
            "event_type": EVENT_TYPE_INTRUSION,
            "polygon": [[50, 50], [300, 50], [300, 300], [50, 300]],
        }
    ],
    "cam10": [
        {
            "name": "Fire_Risk_Zone_10",
            "event_type": EVENT_TYPE_SMOKE_FIRE,
            "polygon": [[0, 0], [640, 0], [640, 384], [0, 384]],
        }
    ],
}

# Thresholds
FALL_ASPECT_RATIO_THRESH = 1.1
FALL_TORSO_ANGLE_THRESH = 50.0
CONFLICT_DIST_THRESH = 1.2
CONFLICT_KINETIC_THRESH = 15.0
INTRUSION_DWELL_FRAMES = 5
EVENT_CONFIRM_FRAMES = 3
GESTURE_WRIST_SPEED_THRESH = 180.0      # px/s vung tay mạnh (pose wrist speed)
PERSON_APPROACH_DIST_THRESH = 0.6       # Khoảng cách tương đối (so chiều cao) để coi là va chạm/tiếp cận
PERSON_APPROACH_SPEED_THRESH = 40.0     # Tốc độ tiến lại gần nhau (px/s)