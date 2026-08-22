import cv2

# Pipeline Settings
MODEL_INPUT_SIZE = (640, 384)
DETECTION_INTERVAL = 3
TEMPORAL_BUFFER_MAXLEN = 45          # Tăng để phân tích temporal tốt hơn (was 30)

# ============================================================
# AI PERFORMANCE CONTRACT (Task)
# Rule 1: 1 model = 1 instance/process — SHARED, không load theo camera.
# Rule 2: Camera chỉ capture + giữ latest frame; KHÔNG tự chạy inference.
# Rule 3: Camera FPS != AI inference FPS (detect 5 FPS, fire 1-2 FPS, accident 1-2 FPS).
# Rule 4: Cascade — không chạy mọi model trên mọi frame.
# Rule 7: CANDIDATE != EVENT (nomenclature _DETECTED/_CANDIDATE/_CONFIRMED).
# Rule 9: Embedding async, không chặn detection loop.
# Rule 11: Giới hạn PyTorch/OpenMP threads để không chiếm toàn bộ CPU.
# ============================================================

# --- Rule 11: CPU threads limits ---
CPU_LOGICAL_THREADS = 12
AI_MAX_THREADS = 4                # PyTorch/OpenMP dùng tối đa N thread (không chiếm 12)
AI_MAX_INTEROP_THREADS = 1
AI_WORKER_POOL_SIZE = 2           # AI Worker pool dùng chung cho mọi camera

# --- Rule 3: Time-based inference FPS ---
AI_DETECT_FPS = 5.0               # Object detection <= 5 FPS/camera
AI_FIRE_FPS = 1.5                 # Fire/Smoke 1-2 FPS
AI_ACCIDENT_FPS = 1.5             # Accident - tai nạn xe 1-2 FPS

# --- Rule 2/4: latest-frame queue ---
# Camera capture chỉ giữ 1 frame mới nhất; scheduler lấy frame mới -> bỏ frame cũ.
# Job per camera tối đa 1 đang chạy (KHÔNG queue vô hạn).

# --- Rule 12: RAM budget per camera ---
PER_CAMERA_RAM_TARGET_MB = 400
PER_CAMERA_RAM_HARD_MB = 500
# Shared model memory KHÔNG tính lặp lại cho từng camera.

# --- Rule 13: Chỉ số "đã tối ưu" — mục tiêu theo số camera ---
# CPU avg (tỷ lệ trên toàn máy): 1 cam <=25%, 2 cam <=45%, 4 cam <=80%.
BENCH_CPU_TARGET = {1: 0.25, 2: 0.45, 4: 0.80, 8: None}
# RAM toàn process (GB) — model shared đã được trải đều: 1/2/4 cam.
BENCH_RAM_TARGET_GB = {1: 1.4, 2: 1.7, 4: 2.3, 8: None}
# Detection FPS tối thiểu/camera (không được dưới 5).
BENCH_DETECT_FPS_MIN = 5.0
# Frame drop (cadence-miss) tối đa: 1/2 cam <5%, 4 cam <10%, 8 cam <15%.
BENCH_FRAME_DROP_MAX = {1: 0.05, 2: 0.05, 4: 0.10, 8: 0.15}

# Debug: in CANDIDATE mỗi detection pass (mặc định tắt).
AI_DEBUG_CANDIDATES = False

# --- Rule 6C: Vehicle confidence gate trước khi vào collision pipeline ---
VEHICLE_COLLISION_MIN_CONF = 0.65  # conf vehicle >= 0.60-0.70 mới được vào collision

# --- Rule 7: Event nomenclature ---
STAGE_CANDIDATE = "CANDIDATE"
STAGE_CONFIRMED = "CONFIRMED"

EVENT_TYPE_SMOKE = "SMOKE_DETECTED"

# ============================================================
# SCORE FORMULA WEIGHTS + THRESHOLDS — Rule 6
# Mọi sự kiện phải đi qua candidate -> confirmation.
# ============================================================

# ---- A. FALL: FallScore = 0.25*posture + 0.25*vertical_motion
#                 + 0.20*bbox_aspect_change + 0.15*center_velocity
#                 + 0.15*temporal_score ; >= 0.80 & >= 3/5 frames => FALL_CONFIRMED
FALL_W_POSTURE = 0.25
FALL_W_VERTICAL_MOTION = 0.25
FALL_W_ASPECT_CHANGE = 0.20
FALL_W_CENTER_VELOCITY = 0.15
FALL_W_TEMPORAL = 0.15
FALL_CANDIDATE_THRESH = 0.50
FALL_CONFIRM_THRESH = 0.80
FALL_CONFIRM_FRAMES = 3            # >= 3 frames

# ---- B. FIGHT: FightScore = 0.20*person_pair + 0.20*contact
#                 + 0.25*relative_motion + 0.20*motion_intensity
#                 + 0.15*temporal_score
FIGHT_W_PERSON_PAIR = 0.20
FIGHT_W_CONTACT = 0.20
FIGHT_W_RELATIVE_MOTION = 0.25
FIGHT_W_MOTION_INTENSITY = 0.20
FIGHT_W_TEMPORAL = 0.15
FIGHT_CANDIDATE_THRESH = 0.50
FIGHT_CONFIRM_THRESH = 0.70
FIGHT_CONFIRM_FRAMES = 3

# ---- C. COLLISION: CollisionScore = 0.15*track_stability
#                 + 0.20*relative_velocity + 0.20*distance_closing
#                 + 0.20*collision_geometry + 0.15*velocity_change
#                 + 0.10*temporal_score
COLLISION_W_TRACK_STABILITY = 0.15
COLLISION_W_RELATIVE_VELOCITY = 0.20
COLLISION_W_DISTANCE_CLOSING = 0.20
COLLISION_W_GEOMETRY = 0.20
COLLISION_W_VELOCITY_CHANGE = 0.15
COLLISION_W_TEMPORAL = 0.10
COLLISION_CANDIDATE_THRESH = 0.50
COLLISION_CONFIRM_THRESH = 0.75
COLLISION_CONFIRM_FRAMES = 3

# ---- D. FIRE: FireScore = 0.35*fire_model + 0.20*spatial_consistency
#                 + 0.20*temporal_persistence + 0.15*fire_motion
#                 + 0.10*smoke_corroboration ; >= 0.85 => FIRE_CONFIRMED
FIRE_W_MODEL = 0.35
FIRE_W_SPATIAL = 0.20
FIRE_W_TEMPORAL = 0.20
FIRE_W_MOTION = 0.15
FIRE_W_SMOKE = 0.10
FIRE_CANDIDATE_THRESH = 0.55
FIRE_CONFIRM_THRESH = 0.85
FIRE_CONFIRM_FRAMES = 6

# ---- E. SMOKE: SmokeScore + persistence + spatial_expansion + shape
SMOKE_W_MODEL = 0.35
SMOKE_W_TEMPORAL = 0.25
SMOKE_W_EXPANSION = 0.20
SMOKE_W_SHAPE = 0.20
SMOKE_CANDIDATE_THRESH = 0.50
SMOKE_CONFIRM_THRESH = 0.75
SMOKE_CONFIRM_FRAMES = 8
SMOKE_MIN_EXPANSION = 0.05      # Rule 6E: khói phải LAN RỘNG mới là khói thật

# ---- Tracking association (YOLO shared => ID phải gán theo camera) ----
TRACK_ASSOC_IOU = 0.20            # IoU tối thiểu để khớp detection vào track cũ

# ---- Embedding async (Rule 9) ----
EMBEDDING_ENABLED = True
EMBED_QUEUE_MAXSIZE = 32          # queue crop giới hạn (không vô hạn)
EMBEDDING_MODEL_PATH = ""         # Có thể tự tải: torchvision mobilenet_v3_large weights
EMBEDDING_FAISS_INDEX = ""        # Path index FAISS (.index). Rỗng -> dùng index in-memory
EMBEDDING_FEATURE_DIM = 512       # DIM vector embedding ghi vào index (MobileNetV3 cho 1024)

# Model / Inference
DEVICE = "cpu"                       # "cpu" hoặc "0" (GPU)
DET_MODEL_PATH = "weights/yolo11n.pt"
POSE_MODEL_PATH = "weights/yolo11n-pose.pt"
# ============================================================
# FINE-GRAINED VEHICLE CLASSIFICATION (xe máy / xe tải / xe chở dầu...)
# Model YOLO-cls (yolo11n-cls) đã train riêng cho các dạng xe.
# Nếu file model chưa tồn tại -> tự vô hiệu hoá, pipeline chạy như cũ.
# ============================================================
VEHICLE_CLS_ENABLED = True
VEHICLE_CLS_MODEL_PATH = "weights/vehicle_cls.pt"
VEHICLE_CLS_CROP_MARGIN = 1.15          # Mở rộng crop quanh bbox xe
VEHICLE_CLS_MIN_CONF = 0.55             # Conf tối thiểu để chấp nhận nhãn
# --- Chống CHE KHUẤT / xe KHÔNG HOÀN CHỈNH ---
# Xe bị che 1 số góc cạnh → bbox nhỏ/thiếu context. Xử lý:
#   - Bbox nhỏ → mở rộng context (CROP_MARGIN_OCCLUDED) + upscale crop.
#   - Temporal MAJORITY VOTE theo track: 1 frame phân loại sai/thiếu không
#     làm đổi nhãn; phải nhiều frame ủng hộ cùng 1 loại xe mới chốt.
VEHICLE_CLS_CROP_MARGIN_OCCLUDED = 1.6  # Mở rộng crop khi bbox nhỏ/bị che
VEHICLE_CLS_VOTE_MIN = 2                # Số frame ủng hộ tối thiểu cùng 1 loại xe
VEHICLE_CLS_VOTE_KEEP = 8               # Cửa sổ vote trượt (frame)
VEHICLE_CLS_IN_VEHICLE_REMOVE_RATIO = 0.70  # Bỏ track NGƯỜI nằm >= 70% diện tích trong bbox xe (tài xế/hành khách) — phân biệt rõ người/xe
VEHICLE_CLS_NAME_MAP = {                # Tên class của model -> tiếng Việt
    "xe_may": "xe máy",
    "motorbike": "xe máy",
    "xe_tai": "xe tải",
    "truck": "xe tải",
    "xe_cho_dau": "xe chở dầu",
    "tanker": "xe chở dầu",
    "xe_buyt": "xe buýt",
    "bus": "xe buýt",
    "o_to": "ô tô",
    "car": "ô tô",
    "xe_dap": "xe đạp",
    "bicycle": "xe đạp",
    "tau_hoa": "tàu hỏa",
    "train": "tàu hỏa",
    # --- Phân loại tinh (chi tiết trong từng nhóm COCO) ---
    # xe máy
    "moto": "mô tô",
    "motor": "mô tô",
    "xe_dien": "xe máy điện",
    "electric_motorcycle": "xe máy điện",
    "xe_ba_banh": "xe ba bánh",
    "tricycle": "xe ba bánh",
    # ô tô
    "o_to_con": "ô tô con",
    "sedan": "ô tô con",
    "suv": "ô tô SUV",
    "taxi": "taxi",
    "pickup": "ô tô bán tải",
    "pickup_truck": "ô tô bán tải",
    "xe_cuu_thuong": "xe cứu thương",
    "ambulance": "xe cứu thương",
    "xe_canh_sat": "xe cảnh sát",
    "police_car": "xe cảnh sát",
    "xe_banh": "xe cứu hỏa",
    "fire_truck": "xe cứu hỏa",
    # xe tải
    "xe_tai_container": "xe container",
    "container_truck": "xe container",
    "xe_tai_co_ben": "xe tải có ben",
    "dump_truck": "xe tải có ben",
    "xe_kich_hoat": "xe kéo / đầu kéo",
    "trailer_truck": "xe kéo / đầu kéo",
    "xe_do_xi_te": "xe bồn",
    "tanker_truck": "xe bồn",
    # xe buýt / tàu
    "xe_khach": "xe khách",
    "coach": "xe khách",
    "xichlo": "xe xích lô",
    "cyclo": "xe xích lô",
}
DETECT_CLASSES = [0, 1, 2, 3, 5, 6, 7]  # person + phương tiện giao thông (bỏ boat)
VEHICLE_CLASSES = [1, 2, 3, 5, 6, 7]    # bicycle, car, motorbike, bus, train, truck
CONF_THRESH = 0.30                   # Giảm nhẹ để recall tốt hơn (was 0.35)
ENABLE_POSE = True

# Event Type Constants
# ============================================================
# PHASE 1 — chỉ tập trung 4 nhóm event:
#   1. Person fall / conflict  (HUMAN_FALL, HUMAN_CONFLICT)
#   2. Vehicle collision       (VEHICLE_COLLISION — xe-xe)
#   3. Smoke / Fire            (FIRE_DETECTED)
#   4. Restricted-zone intrusion (RESTRICTED_INTRUSION)
# EVENT_TYPE_SMOKE_FIRE / EVENT_TYPE_INTRUSION dùng để đánh dấu ROI.
# ============================================================
EVENT_TYPE_SMOKE_FIRE = "SMOKE_FIRE"
EVENT_TYPE_INTRUSION = "INTRUSION"

# Stream URLs
CAMERA_STREAMS = {
    "cam09": "d:\\smartcity_prototype\\congnhan_kho_danhnhau_tranhchap.mp4",
    "cam10": "d:\\smartcity_prototype\\xeravao_tainan_khoi_chayno.mp4",
}

# ============================================================
# Dashboard Web — Hiển thị cảnh báo + snapshot lý do cảnh báo
# ============================================================
DASHBOARD_ENABLED = True
DASHBOARD_HOST = "0.0.0.0"
DASHBOARD_PORT = 8080
SNAPSHOT_DIR = "snapshots"            # Thư mục chụp ảnh khi có alert
SNAPSHOT_MAX_ALERTS = 200             # Số alert lưu trên dashboard (chỉ giữ trong memory)
SNAPSHOT_CROP_MARGIN = 0.4            # Mở rộng vùng cắt snapshot quanh bbox (tỷ lệ so với w/h bbox)
ALERT_LOG_FILE = "alerts.log"         # File log cảnh báo (mỗi alert 1 dòng)

# ============================================================
# Headless Mode — Chạy không cửa sổ video, chỉ log + snapshot
# Bật (True): không cv2.imshow, cảnh báo in ra console + ghi alerts.log
#             và chụp snapshot cắt đúng vị trí cảnh báo.
# ============================================================
HEADLESS = False
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
# EVENT RULE META (candidate / confirm scoring) — Rule 6 & 7
# MỌI event: confidence >= candidate => candidate (chưa alert)
#            duy trì >= frames & score >= confirm => CONFIRMED (alert + snapshot)
# ============================================================
EVENT_RULE_META = {
    "HUMAN_FALL": {
        "candidate": FALL_CANDIDATE_THRESH,
        "confirm": FALL_CONFIRM_THRESH,
        "frames": FALL_CONFIRM_FRAMES,
    },
    "HUMAN_CONFLICT": {
        "candidate": FIGHT_CANDIDATE_THRESH,
        "confirm": FIGHT_CONFIRM_THRESH,
        "frames": FIGHT_CONFIRM_FRAMES,
    },
    "VEHICLE_COLLISION": {
        "candidate": COLLISION_CANDIDATE_THRESH,
        "confirm": COLLISION_CONFIRM_THRESH,
        "frames": COLLISION_CONFIRM_FRAMES,
    },
    "FIRE_DETECTED": {
        "candidate": FIRE_CANDIDATE_THRESH,
        "confirm": FIRE_CONFIRM_THRESH,
        "frames": FIRE_CONFIRM_FRAMES,
    },
    "SMOKE_DETECTED": {
        "candidate": SMOKE_CANDIDATE_THRESH,
        "confirm": SMOKE_CONFIRM_THRESH,
        "frames": SMOKE_CONFIRM_FRAMES,
    },
    "RESTRICTED_INTRUSION": {
        "candidate": 0.60,
        "confirm": 0.80,
        "frames": 3,
    },
}


def rule_meta(event_type, default_candidate=0.0, default_confirm=0.0, default_frames=2):
    m = EVENT_RULE_META.get(event_type, {})
    return {
        "candidate": m.get("candidate", default_candidate),
        "confirm": m.get("confirm", default_confirm),
        "frames": int(m.get("frames", default_frames)),
    }


# ============================================================
# GRID ZONES — Chia frame thành LƯỚI VÙNG cho xô xát / va chạm người
# Chỉ xét xô xát giữa những người CÙNG ô lưới (dựa trên tâm bbox người).
# 2 người ở 2 ô khác nhau = 2 khu vực khác nhau trong frame → không xét
# dù bbox đôi khi chạm nhau. Chia theo MODEL_INPUT_SIZE vì toàn bộ bbox
# nằm trong toạ độ này.
# ============================================================
GRID_COLS = 3
GRID_ROWS = 2


def grid_row_col(cx, cy, frame_w=None, frame_h=None):
    """Ô lưới (row, col) chứa điểm (cx, cy)."""
    w = frame_w or MODEL_INPUT_SIZE[0]
    h = frame_h or MODEL_INPUT_SIZE[1]
    col = min(int(cx * GRID_COLS / w), GRID_COLS - 1)
    row = min(int(cy * GRID_ROWS / h), GRID_ROWS - 1)
    return row, col


def grid_zone(cx, cy, frame_w=None, frame_h=None):
    """Ô lưới (index) chứa điểm (cx, cy) — dùng để phân vùng người theo frame."""
    row, col = grid_row_col(cx, cy, frame_w, frame_h)
    return row * GRID_COLS + col


# ============================================================
# FALL DETECTION — tối giản: chỉ dựa tư thế nằm ngang + persistence
# ============================================================
FALL_ASPECT_RATIO_THRESH = 1.2       # Width/Height > threshold = tư thế nằm ngang

# ============================================================
# CONFLICT / FIGHT DETECTION — tối giản
# Xô xát = 2 người GẦN nhau + vận động bất thường (jitter/body speed),
# duy trì >= CONFLICT_SUSTAINED_FRAMES frames.
# ============================================================
CONFLICT_DIST_HARD_CAP = 2.0        # Khoảng cách tương đối TỐI ĐA (by avg height)
CONFLICT_SUSTAINED_FRAMES = 4       # Phải duy trì tín hiệu xung đột >= N frames
CONFLICT_AGITATION_THRESH = 3.0     # Jitter / body_speed*2 vượt mức này = bất thường

# ============================================================
# VEHICLE COLLISION — tối giản: proximity (gate) + closing/decel (kinetic)
# ============================================================
VEHICLE_PROXIMITY_DIST_RATIO = 0.25  # Dist / avg_diagonal < threshold = TIẾP CẬN
VEHICLE_IOU_THRESH = 0.10            # IoU overlap threshold — tiếp xúc = overlap thật
VEHICLE_DECEL_THRESH = 80.0          # Giảm tốc đột ngột (px/s²)
VEHICLE_CLOSING_SPEED_THRESH = 25.0  # Tốc độ tiến lại gần nhau
VEHICLE_COLLISION_SUSTAINED = 4      # Sustained proximity + anomaly frames
# Chỉ báo va chạm khi có TÍN HIỆU ĐỘNG HỌC thật: xe tiến lại nhanh (closing)
# HOẶC giảm tốc đột ngột (decel).
VEHICLE_COLLISION_MIN_KINETIC = 0.40   # Điểm tối thiểu của closing hoặc decel

# ============================================================
# VEHICLE DEFORMATION — xe ĐƠN bị thay đổi aspect ratio so với baseline
# (loại trừ bbox bị cắt bởi màn hình). Tín hiệu độc lập cho xe lật / đâm
# vật cản không được track.
# ============================================================
VEHICLE_DEFORM_OCL_FRAME_MARGIN = 3       # Bbox chạm frame edge -> bị màn hình cắt
VEHICLE_DEFORM_MIN_SCORE = 0.55           # Mức deform tối thiểu để coi là tai nạn

# ============================================================
# INTRUSION — Spatial Rules
# ============================================================
INTRUSION_DWELL_FRAMES = 10          # Thời gian lưu lại vùng cấm (was 5)
INTRUSION_MIN_MOVEMENT_PX = 12.0     # Dịch chuyển tối thiểu từ lúc vào vùng (was 8)
INTRUSION_DEPTH_RATIO = 0.15         # Phải vào sâu >= 15% chiều cao người vào trong polygon

# ============================================================
# SMOKE / FIRE — tối giản: fire mask (HSV) + hue spread + flicker + persistence
# ============================================================
SMOKE_FIRE_MIN_CONTOUR_AREA = 250    # Diện tích contour tối thiểu (px²)
SMOKE_FIRE_FIRE_PIXEL_THRESH = 250   # Pixel lửa tối thiểu
SMOKE_FIRE_FIRE_PERSIST_THRESH = 6   # Persistence frames cho lửa (cửa sổ xác nhận)
# LOẠI BỎ VẬT THỂ MÀU LỬA: nếu vùng "lửa" nằm chủ yếu BÊN TRONG bbox người
# (đã detect bởi YOLO), đó là áo quần màu đỏ/cam chứ không phải lửa → loại.
SMOKE_FIRE_MAX_PERSON_OVERLAP = 0.60       # > 60% pixel "lửa" nằm trong bbox người → loại