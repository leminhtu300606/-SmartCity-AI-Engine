import cv2

# Pipeline Settings
MODEL_INPUT_SIZE = (640, 384)
DETECTION_INTERVAL = 3
TEMPORAL_BUFFER_MAXLEN = 45          # Tăng để phân tích temporal tốt hơn (was 30)

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
MIN_ALERT_CONFIDENCE = 0.9           # Chỉ đưa alert (log/vẽ/snapshot) có confidence >= ngưỡng này

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


def grid_adjacent(cx1, cy1, cx2, cy2, frame_w=None, frame_h=None):
    """2 điểm ở CÙNG ô hoặc 2 ô KỀ NHAU (8 hướng) — bắt đánh xa qua biên ô."""
    r1, c1 = grid_row_col(cx1, cy1, frame_w, frame_h)
    r2, c2 = grid_row_col(cx2, cy2, frame_w, frame_h)
    return max(abs(r1 - r2), abs(c1 - c2)) <= 1

# ============================================================
# EVENT CONFIRMATION — Per-Event-Type
# Mỗi loại event cần số frame xác nhận khác nhau.
# MIN_CONFIRM_FRAMES = ngưỡng tối thiểu: MỌI hiện tượng phải xuất hiện
# ít nhất 2 detection frame trước khi bắn alert (tránh cảnh báo 1 frame).
# Smoke/Fire, VEHICLE_COLLISION có persistence riêng bên trong rule, nhưng
# confirm tracker vẫn giữ tối thiểu 2 frame cho thống nhất.
# ============================================================
MIN_CONFIRM_FRAMES = 2                # Ngưỡng tối thiểu cho MỌI event (>= 2 frame)
EVENT_CONFIRM_FRAMES_DEFAULT = 4
EVENT_CONFIRM_MAP = {
    "HUMAN_FALL": 5,
    "HUMAN_CONFLICT": 3,             # was 4 — xô xát có thể ngắn; hạ để không bỏ lọt
    "VEHICLE_COLLISION": 2,          # Rule collision đã tự sustained (VEHICLE_COLLISION_SUSTAINED=4); confirm 2 = tổng >= 2 frame
    "FIRE_DETECTED": 2,              # Smoke/fire dùng persistence riêng bên trong (>= 8 frame)
}

# ============================================================
# FALL DETECTION — tối giản: chỉ dựa tư thế nằm ngang + persistence
# ============================================================
FALL_ASPECT_RATIO_THRESH = 1.2       # Width/Height > threshold = tư thế nằm ngang
FALL_PERSIST_FRAMES = 4              # Phải nằm ngang >= N detection frames mới xác nhận

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
SMOKE_FIRE_FLICKER_THRESH = 0.07     # Tỷ lệ thay đổi frame-to-frame cho flicker
SMOKE_FIRE_FIRE_HUE_CIRC_VAR_THRESH = 0.025  # Circular variance hue tối thiểu (lửa 0.03+, vật đỏ đồng màu ≈ 0)
# LOẠI BỎ VẬT THỂ MÀU LỬA: nếu vùng "lửa" nằm chủ yếu BÊN TRONG bbox người
# (đã detect bởi YOLO), đó là áo quần màu đỏ/cam chứ không phải lửa → loại.
SMOKE_FIRE_MAX_PERSON_OVERLAP = 0.60       # > 60% pixel "lửa" nằm trong bbox người → loại

# COOLDOWN: sau khi báo 1 event cho 1 zone, KHÔNG báo lại zone đó trong N frame.
SMOKE_FIRE_ALERT_COOLDOWN = 120