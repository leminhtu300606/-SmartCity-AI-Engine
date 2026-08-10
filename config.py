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
MIN_ALERT_CONFIDENCE = 0.9           # Chỉ đưa alert (log/vẽ/snapshot) có confidence >= ngưỡng này

# Event Type Constants
EVENT_TYPE_HUMAN_ACTION = "HUMAN_ACTION"
EVENT_TYPE_HUMAN_GROUP_CONFLICT = "HUMAN_GROUP_CONFLICT"
EVENT_TYPE_VEHICLE_ACCIDENT = "VEHICLE_ACCIDENT"
EVENT_TYPE_SMOKE_FIRE = "SMOKE_FIRE"
EVENT_TYPE_INTRUSION = "INTRUSION"
EVENT_TYPE_VEHICLE_OBJECT_COLLISION = "VEHICLE_OBJECT_COLLISION"
EVENT_TYPE_OBJECT_FALLING = "OBJECT_FALLING_ON_VEHICLE"

# Stream URLs
CAMERA_STREAMS = {
    "cam09": "https://dathoc.net/induxvid/cam09/index.m3u8?cookieCheck=1",
    "cam10": "https://dathoc.net/induxvid/cam10/index.m3u8?cookieCheck=1",
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


def grid_zone(cx, cy, frame_w=None, frame_h=None):
    """Ô lưới (index) chứa điểm (cx, cy) — dùng để phân vùng người theo frame."""
    w = frame_w or MODEL_INPUT_SIZE[0]
    h = frame_h or MODEL_INPUT_SIZE[1]
    col = min(int(cx * GRID_COLS / w), GRID_COLS - 1)
    row = min(int(cy * GRID_ROWS / h), GRID_ROWS - 1)
    return row * GRID_COLS + col

# ============================================================
# EVENT CONFIRMATION — Per-Event-Type
# Mỗi loại event cần số frame xác nhận khác nhau.
# Smoke/Fire = 0 vì chúng có persistence logic riêng bên trong.
# ============================================================
EVENT_CONFIRM_FRAMES_DEFAULT = 4
EVENT_CONFIRM_MAP = {
    "HUMAN_FALL": 5,
    "HUMAN_CONFLICT": 3,             # was 4 — xô xát có thể ngắn; hạ để không bỏ lọt
    "HUMAN_GROUP_CONFLICT": 3,
    "PERSON_COLLISION": 3,           # Closing speed tính trung bình 3 frame; cửa sổ candidate ngắn khi vật gặp nhau
    "VEHICLE_COLLISION": 1,          # Rule collision đã tự sustained (VEHICLE_COLLISION_SUSTAINED=3); confirm=1 để khỏi đếm kép
    "VEHICLE_OBJECT_COLLISION": 2,   # Xe va chạm vật thể/người: sustained ngắn vì tương tác nhanh
    "OBJECT_FALLING_ON_VEHICLE": 3,  # Vật thể rơi vào xe: cần xác nhận nhiều frame (rơi + đáp)
    "VEHICLE_STOP_ANOMALY": 3,          # Rule hard_stop đã tự sustained (VEHICLE_HARD_STOP_SUSTAINED=3); confirm thấp để không bỏ sót
    "VEHICLE_ACCIDENT": 1,           # Rule vehicle_state đã tự sustained (VEHICLE_TILT_SUSTAINED=3); confirm=1 để khỏi đếm kép
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
# CONFLICT / FIGHT DETECTION — Pipeline 2 GIAI ĐOẠN (giống vehicle)
# Stage 1: XÁC ĐỊNH NGƯỜI + đánh giá TÌNH TRẠNG của từng người:
#   - Biên độ vung tay (wrist amplitude), tốc độ cổ tay (wrist speed).
#   - Tăng tốc/giảm tốc đột ngột (sudden accel).
#   - Tư thế nằm vật nhau (lying) — dấu hiệu quật ngã/đè nhau.
#   - Độ giật cục (jitter).
# Stage 2: KẾT LUẬN XÔ XÁT CHỈ KHI có TÌNH TRẠNG BẤT THƯỜNG + gần nhau.
# LOẠI BỎ "ở sát nhau": 2 người đứng/nói chuyện sát nhau nhưng CẢ HAI bình
# thường (không vung tay, không tăng tốc, không nằm, không giật) → KHÔNG xô xát.
# ============================================================
CONFLICT_DIST_THRESH = 0.85          # Khoảng cách tương đối tối đa (by avg height)
CONFLICT_SUSTAINED_FRAMES = 4        # Phải duy trì tín hiệu xung đột >= N frames
GROUP_INTERACTION_DIST_THRESH = 0.95 # 3 người chỉ được coi là cùng 1 cụm nếu đủ gần nhau

# Stage 1 — Tình trạng cá nhân (threshold riêng của từng người)
CONFLICT_CALM_AGITATION_THRESH = 3.0   # Dưới mức này = người bình thường (đứng yên, was 4.0)
CONFLICT_WRIST_AMPLITUDE_THRESH = 40.0 # Biên độ vung tay tối thiểu (px) — cổ tay vung xa
CONFLICT_ACCEL_THRESH = 90.0           # Tăng tốc/giảm tốc đột ngột (px/s²)
CONFLICT_LYING_ASPECT_THRESH = 1.2     # Aspect ratio nằm ngang = tư thế nằm vật

# Đường A (Pose): tốc độ cổ tay tối thiểu = "vung tay đấm" (px/s)
CONFLICT_WRIST_HIGH_THRESH = 26.0
# Mức kích động tối thiểu của người thứ 2 khi có người vung tay (đôi công)
CONFLICT_MUTUAL_AGITATION_THRESH = 5.0
# Phương sai khoảng cách tối thiểu (theo avg height) → "giằng co".
# Đo thực tế: giằng co dist dao động ~0.2–0.4 rel → var ≈ 0.01–0.03 (was 0.25 quá cao,
# khiến signal_oscillation không bao giờ fire → bỏ lọt vật lộn tại chỗ).
CONFLICT_DIST_VAR_THRESH = 0.02

# Đường B (BBox fallback): mỗi người phải jitter RẤT cao mới coi là đánh nhau
# (cao hơn hẳn việc đi lại/chen lấn thông thường trong đám đông)
CONFLICT_BBOX_JITTER_THRESH = 9.0

# Đường D (Một chiều tấn công): 1 người cực kỳ kích động (đấm/đẩy liên tục) khi
# rất gần người kia dù người kia gần như đứng yên (thủ/không kịp phản ứng).
# Đây là kiểu ẩu đả phổ biến nhất (1 phía tấn công dồn dập) mà đường A/B bỏ lọt
# vì không đạt signal_mutual. Bắt khi: 1 người jitter/wrist rất cao + khoảng
# cách gần sát + (khoảng cách dao động HOẶC 2 bbox chồng nhau).
CONFLICT_ONE_SIDED_AGITATION_THRESH = 14.0
CONFLICT_ONE_SIDED_DIST_THRESH = 0.70

# Đường C (Grapple / vật lộn): 2 người dính sát nhau, bbox chồng nhau (IoU cao),
# giằng co TẠI CHỖ với cường độ vừa phải — thường KHÔNG vung tay đấm (wrist thấp)
# và KHÔNG dịch chuyển xa (jitter thấp hơn đấm đá) nên 2 đường A/B dễ bỏ lọt.
# Phân biệt:
#   - Ôm nhau / đứng nói chuyện sát: IoU cao NHƯNG jitter ≈ 0 → không bắt.
#   - Đám đông chen lấn: jitter cao NHƯNG ít khi bbox chồng nhau mạnh (IoU thấp).
CONFLICT_GRAPPLE_IOU_THRESH = 0.15   # Bbox 2 người chồng nhau >= 15% (was 0.18)
CONFLICT_GRAPPLE_DIST_THRESH = 0.45  # Khoảng cách tương đối tối đa khi vật lộn (was 0.40)
CONFLICT_GRAPPLE_MIN_JITTER = 5.0    # Ít nhất 1 người giằng co đáng kể (was 8.5)

# Đường E (Người ngã trong cặp): 1 người có tín hiệu ngã khi 2 người gần nhau
# → xem như xô xát. Bonus điểm cố định để confidence vượt MIN_ALERT_CONFIDENCE
# (0.9): 0.80 + bonus/40 >= 0.9 → bonus >= 4.0.
CONFLICT_FALL_SCORE_BONUS = 5.0

# Điểm kết hợp (cho confidence)
CONFLICT_KINETIC_THRESH = 12.0       # Kinetic score threshold

# ============================================================
# VEHICLE ↔ OBJECT COLLISION — Xe va chạm vật thể / người
# Không chỉ xe-xe: xe đâm vật tĩnh, xe cán người, xe va chạm motorbike/bike...
# Tái sử dụng đặc trưng động học của VehicleAccidentRules nhưng chỉ cần
# MỘT bên là xe, bên kia là bất kỳ object nào (kể cả người).
# ============================================================
VEHICLE_OBJECT_IOU_THRESH = 0.08      # IoU overlap threshold (was 0.04)
VEHICLE_OBJECT_PROXIMITY_DIST_RATIO = 0.40  # Dist/avg_diagonal (was 0.65/0.55)
VEHICLE_OBJECT_CLOSING_SPEED_THRESH = 14.0  # Tốc độ tiến lại nhau (thấp hơn xe-xe)
VEHICLE_OBJECT_SUSTAINED = 4          # Duy trì >= N frames (was 2)
VEHICLE_OBJECT_SCORE_THRESH = 0.50    # Điểm tai nạn tối thiểu (was 0.38)

# ============================================================
# OBJECT FALLING ONTO VEHICLE — Vật thể rơi vào xe
# 1 object (người/hàng hóa/vật thể) rơi từ trên xuống và trúng xe.
# Tín hiệu: vận tốc dọc (vy) âm mạnh + vị trí rơi chồng/tiến vào bbox xe.
# ============================================================
OBJECT_FALL_VY_NORM_THRESH = 1.9      # |vy|/chiều cao object tối thiểu = rơi nhanh
OBJECT_FALL_OVERLAP_RATIO = 0.30      # Phần bbox object phải vào trong bbox xe
OBJECT_FALL_WINDOW = 6                # Cửa sổ vận tốc dọc để xác định quỹ đạo rơi
OBJECT_FALL_SUSTAINED = 4             # Duy trì quỹ đạo rơi >= N frames

# ============================================================
# PERSON COLLISION / APPROACH
# Va chạm giữa 2 người — 3 dạng tín hiệu bổ sung nhau:
#   1) Tiếp cận nhanh: closing speed (2 người lao vào nhau).
#   2) Di chuyển bất thường: ít nhất 1 người di chuyển nhanh lạ thường
#      (speed vượt đi bộ/chạy thường), gia tốc/giảm tốc đột ngột, hoặc
#      đổi hướng đột ngột — dấu hiệu chạy xô vào nhau / phanh gấp né.
#   3) Đẩy ngã: 1 người ngã (tư thế nằm ngang / rơi nhanh) khi 2 người
#      rất gần nhau — dấu hiệu bị va/đẩy ngã.
# ============================================================
PERSON_APPROACH_DIST_THRESH = 0.45   # Khoảng cách tương đối (gate bắt buộc)
PERSON_APPROACH_SPEED_THRESH = 40.0  # Closing speed (px/s) — tiếp cận nhanh
PERSON_ABNORMAL_SPEED_THRESH = 40.0  # Speed 1 người bất thường (px/s) — đi vội/chạy (đi thường ~22)
PERSON_ABNORMAL_ACCEL_THRESH = 100.0 # Gia tốc/giảm tốc đột ngột (px/s²)
PERSON_DIR_CHANGE_THRESH = 1.5       # Đổi hướng đột ngột (radians)
PERSON_FALL_PROX_DIST = 0.6          # Khoảng cách (rel) để coi là "bị va làm ngã"
PERSON_COLLISION_SUSTAINED = 2       # Duy trì tín hiệu va chạm >= N frames (va chạm rất nhanh)

# ============================================================
# VEHICLE COLLISION — Accident Classifier (Multi-Feature)
# Không chỉ dựa vào bbox overlap: xe chạy sát nhau / bị che khuất
# cũng tạo overlap giả. Kết hợp nhiều đặc trưng quỹ đạo.
# ============================================================
VEHICLE_PROXIMITY_DIST_RATIO = 0.25  # Dist / avg_diagonal < threshold = TIẾP CẬN (was 0.45).
                                     # Chỉ coi là va chạm tiềm năng khi bbox chồng nhau
                                     # (IoU) hoặc cực gần (≈ chạm nhau). Xe xếp hàng chờ
                                     # đèn đỏ cách nhau nửa thân xe → KHÔNG kích.
VEHICLE_IOU_THRESH = 0.10            # IoU overlap threshold (was 0.05) — tiếp xúc = overlap thật
VEHICLE_DECEL_THRESH = 80.0          # Giảm tốc đột ngột (px/s²)
VEHICLE_DIR_CHANGE_THRESH = 0.8      # Đổi hướng đột ngột (radians)
VEHICLE_CLOSING_SPEED_THRESH = 25.0  # Tốc độ tiến lại gần nhau
VEHICLE_COLLISION_SUSTAINED = 4      # Sustained proximity + anomaly frames (was 2)

# Accident Classifier: điểm tổng hợp từ nhiều đặc trưng
VEHICLE_DIST_DROP_THRESH = 10.0      # Khoảng cách giảm nhanh giữa 2 frame (px)
VEHICLE_COLLISION_SCORE_THRESH = 0.55  # Điểm tai nạn tối thiểu (was 0.42)
VEHICLE_POST_CONTACT_WINDOW = 4      # Số frame sau TIẾP XÚC THẬT để check hậu va chạm (was 8)
# Chỉ báo va chạm khi có TÍN HIỆU ĐỘNG HỌC thật: xe tiến lại nhanh (closing)
# HOẶC giảm tốc đột ngột (decel) đạt tối thiểu. Xe đậu sát nhau / đi song song
# (closing≈0, decel≈0) → không phải va chạm dù bbox rất gần.
VEHICLE_COLLISION_MIN_KINETIC = 0.40   # Điểm tối thiểu của closing hoặc decel (was 0.20)
VEHICLE_COLLISION_WEIGHTS = {
    "proximity": 0.25,   # 2 xe tiến rất gần / bbox giao nhau (gate bắt buộc)
    "closing": 0.18,     # Khoảng cách giảm nhanh (closing velocity)
    "dist_drop": 0.10,   # Khoảng cách tụt nhanh frame-to-frame
    "direction": 0.05,   # Đổi hướng / chuyển động đột ngột (giảm trọng số — nhiễu)
    "decel": 0.18,       # Giảm tốc đột ngột
    "tilt": 0.14,        # TÌNH TRẠNG xe: nghiêng/lật (góc nghiêng bất thường) — Stage 1
    "post_contact": 0.10 # Dừng/đổi hướng bất thường ngay sau tiếp xúc thật
}

# ============================================================
# VEHICLE CONDITION — Stage 1: Xác định phương tiện + tình trạng của nó
# TRƯỚC KHI kết luận tai nạn, phải đánh giá TÌNH TRẠNG của từng xe:
#   a) Góc nghiêng/lật (tilt): aspect ratio bbox lệch khỏi giá trị ổn định
#      riêng của xe (baseline) kèm đổi hướng đột ngột → xe nghiêng/lật.
#   b) Bị đè (crush): vật thể/người khác PHỦ >= ngưỡng diện tích bbox xe đang
#      đứng yên → xe bị đè/va mạnh (tai nạn chồng lên).
#   c) Bị xe khác va (collision): 2 xe va nhau → Stage 2 (check_collision).
# ============================================================
VEHICLE_TILT_ASPECT_RATIO_DEVIATION = 0.45  # |aspect_now - baseline|/baseline >= ngưỡng
VEHICLE_TILT_DIR_CHANGE_THRESH = 1.0        # Kèm đổi hướng đột ngột (rad) — không phải quẹo bình thường
VEHICLE_TILT_SPEED_LOW = 4.0                # Xe nghiêng/lật phải đứng yên hoặc rất chậm
VEHICLE_TILT_SUSTAINED = 4                  # Duy trì tư thế nghiêng >= N frames
VEHICLE_TILT_SCORE_THRESH = 0.60            # Điểm tối thiểu để báo VEHICLE_ACCIDENT (single vehicle)

VEHICLE_CRUSH_COVERAGE_THRESH = 0.55        # Vật khác phủ >= 55% diện tích bbox xe
VEHICLE_CRUSH_SUSTAINED = 4                 # Duy trì bị đè >= N frames
VEHICLE_CRUSH_SCORE_THRESH = 0.60           # Điểm tối thiểu để báo bị đè

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
# NGHIÊM NGẶT: ngưỡng cao → bắt chắc chắn, ít false positive.
# ============================================================
SMOKE_FIRE_MIN_CONTOUR_AREA = 350    # Diện tích contour tối thiểu (px², was 250)
SMOKE_FIRE_FIRE_PIXEL_THRESH = 350   # Pixel lửa tối thiểu (was 220) — siết chặt
SMOKE_FIRE_SMOKE_PIXEL_THRESH = 700  # Pixel khói tối thiểu (was 500) — siết chặt
SMOKE_FIRE_FIRE_PERSIST_THRESH = 8   # Persistence frames cho lửa (was 5) — siết chặt
SMOKE_FIRE_SMOKE_PERSIST_THRESH = 10 # Persistence frames cho khói (was 8) — siết chặt
SMOKE_FIRE_FLICKER_THRESH = 0.09     # Tỷ lệ thay đổi frame-to-frame cho flicker (was 0.06)
SMOKE_FIRE_GROWTH_WINDOW = 5         # Cửa sổ phân tích region growth (was 4)

# Fire: độ sáng tương phản cao so với nền tối + tần số nhấp nháy đặc thù lửa thật
SMOKE_FIRE_CONTRAST_THRESH = 40.0          # Chênh lệch mean(V) giữa lửa và nền xung quanh (was 30)
SMOKE_FIRE_FLICKER_FREQ_THRESH = 0.25      # Tần số dao động nhấp nháy lửa (0..1, was 0.20)
SMOKE_FIRE_FLICKER_FREQ_WINDOW = 10        # Cửa sổ mẫu để tính tần số flicker

# Fire: phân loại đối tượng (lửa thật vs vật thể màu tương tự: đèn, mây...)
SMOKE_FIRE_FIRE_EDGE_IRREGULARITY_THRESH = 1.15  # Cạnh sắc tối thiểu (was 1.08)
SMOKE_FIRE_FIRE_JAGGED_THRESH = 1.30            # Cạnh răng cưa rõ rệt (was 1.20)
SMOKE_FIRE_FIRE_CLASS_SCORE_THRESH = 0.45        # Điểm phân loại lửa tối thiểu (0..1, was 0.40)

# PHÂN BIỆT LỬA vs VẬT ĐỎ (áo quần, xe, biển báo):
# - Lửa ĐA SẮC: hue trải rộng (vàng → cam → đỏ). Vật đỏ đồng nhất một sắc độ
#   → circular variance của hue lửa cao, vật đỏ ≈ 0 (đo thực tế: lửa 0.06–0.09,
#   vật đỏ <= 0.03). Ngưỡng 0.05 đặt giữa 2 cụm.
# - Lửa có LÕI SÁNG trắng/vàng nhạt (V cao, S thấp) NẰM GIỮA ngọn lửa. Lõi được
#   đo trên vùng fire mask NỞ RỘNG (dilate) vì mask chính loại pixel S thấp.
#   Vật đỏ/cam thuần (S cao khắp nơi) không có pixel sáng bão hòa thấp → ≈ 0.
SMOKE_FIRE_FIRE_HUE_CIRC_VAR_THRESH = 0.05     # Circular variance hue tối thiểu (was 0.08)
SMOKE_FIRE_FIRE_BRIGHT_CORE_RATIO_THRESH = 0.03  # Tỷ lệ pixel lõi sáng tối thiểu (was 0.06)
# Lửa đồng màu (cháy âm ỉ): diện tích dao động mạnh theo thời gian (co/giãn),
# vật đỏ di chuyển giữ nguyên kích thước → phân biệt không cần lõi/hue.
SMOKE_FIRE_FIRE_AREA_FLUCT_THRESH = 0.15   # CV (std/mean) diện tích lửa tối thiểu (was 0.12)
SMOKE_FIRE_FIRE_CORE_V_THRESH = 225            # V tối thiểu của lõi sáng
SMOKE_FIRE_FIRE_CORE_S_MAX = 140               # S tối đa của lõi (vàng nhạt/trắng, không bão hòa)

# LOẠI BỎ VẬT THỂ MÀU LỬA: nếu vùng "lửa" nằm chủ yếu BÊN TRONG bbox người
# (đã detect bởi YOLO), đó là áo quần màu đỏ/cam chứ không phải lửa → loại.
# Chỉ áp dụng cho NGƯỜI (người mặc áo đỏ), KHÔNG áp dụng cho xe (xe đang cháy
# là lửa thật, vẫn nằm trong bbox xe). Ô tô đỏ không cháy bị loại bởi hue/core.
SMOKE_FIRE_MAX_PERSON_OVERLAP = 0.60       # > 60% pixel "lửa" nằm trong bbox người → loại

# Smoke: dạng mây mờ lan tỏa, đổi hình dạng + độ trong suốt theo thời gian
SMOKE_FIRE_SMOKE_SOFT_GRAD_THRESH = 40.0   # Gradient nội tại < ngưỡng = pixel khói mờ (was 50)
SMOKE_FIRE_SMOKE_SOFTNESS_THRESH = 0.50    # Tỷ lệ pixel mờ tối thiểu trong vùng khói (was 0.40)
SMOKE_FIRE_SMOKE_SHAPE_CHANGE_THRESH = 0.25  # Tỷ lệ hình dạng khói thay đổi (was 0.18)
SMOKE_FIRE_SMOKE_CHANGE_THRESH = 0.03     # Frame diff tối thiểu cho tín hiệu khói (was 0.015)

# ============================================================
# GRID / TILE DETECTION — Phát hiện nghi vấn theo TỪNG Ô NHỎ trong khung hình
# Thay vì chỉ phân tích toàn ROI (lửa ở góc nhỏ bị pha loãng, không định
# vị được vị trí), chia ROI thành lưới ô nhỏ và chạy phân tích độc lập từng ô.
# Mỗi ô có persistence + zone_name riêng → event kèm bbox vị trí chính xác.
# ============================================================
SMOKE_FIRE_GRID_ENABLED = True          # Bật/tắt phát hiện theo ô nhỏ
SMOKE_FIRE_GRID_COLS = 4                # Số cột lưới (640/4 = 160 px/ô)
SMOKE_FIRE_GRID_ROWS = 3                # Số hàng lưới (384/3 = 128 px/ô)
# Ngưỡng pixel tỉ lệ theo diện tích ô (ô nhỏ hơn toàn frame nên cần ngưỡng
# thấp hơn để bắt được đám cháy/khói cục bộ bên trong một ô).
# SIẾT CHẶT: hệ số tăng → ô cần nhiều pixel/contour hơn mới báo.
SMOKE_FIRE_GRID_FIRE_PIXEL_FACTOR = 0.55    # Nhân với ngưỡng toàn frame (350 → 192 px, was 0.45)
SMOKE_FIRE_GRID_SMOKE_PIXEL_FACTOR = 0.40   # Nhân với ngưỡng toàn frame (700 → 280 px)
SMOKE_FIRE_GRID_MIN_CONTOUR_FACTOR = 0.40   # Contour tối thiểu (350 → 140 px²)
SMOKE_FIRE_GRID_CONTRAST_THRESH = 30.0      # Tương phản tối thiểu trong ô (was 25)
# Diện tích lửa trong ô nhỏ dao động tự nhiên ít hơn toàn frame (ô nhỏ hơn →
# % thay đổi tuyệt đối nhỏ hơn) → hạ ngưỡng CV để không bỏ lọt lửa cục bộ.
SMOKE_FIRE_GRID_AREA_FLUCT_THRESH = 0.12    # CV diện tích trong ô (was 0.10)
SMOKE_FIRE_GRID_JAGGED_THRESH = 1.25        # Cạnh răng cưa trong ô (was 1.15)

# ============================================================
# SMALL / OCCLUDED FIRE — điểm cháy nhỏ & lửa bị che khuất
# Hai vấn đề pipeline cũ bỏ lọt:
#   1) Lửa NHỎ: vài chục pixel → morphological open 5x5 xoá sạch blob, và
#      dưới ngưỡng pixel của ô (99 px). → pipeline riêng ngưỡng thấp, dựa
#      vào flicker + hue/lõi sáng (không đòi contrast/edge — không tin cậy
#      với blob nhỏ).
#   2) Lửa BỊ CHE KHUẤT (tường, container, bờ kè...): ngọn lửa không hiện
#      nhưng ánh cam HẮT RA vật cản tạo quầng sáng ấm nhấp nháy theo ngọn
#      lửa bên trong → pipeline "glow" dải HSV mờ hơn (S/V thấp) bắt quầng.
# ============================================================
SMOKE_FIRE_SMALL_FIRE_ENABLED = True
# SIẾT CHẶT lửa nhỏ: ngưỡng 20px quá thấp — xe đèn hậu đỏ/cam, áo phản quang,
# phản chiếu nắng chiều di chuyển qua ô đều đủ flicker → báo lửa giả. Nâng
# pixel/contour/persistence lên nhiều để chỉ bắt blob lửa thật rõ ràng.
SMOKE_FIRE_SMALL_FIRE_PIXEL_THRESH = 80    # Pixel lửa nhỏ tối thiểu (mỗi ô grid, was 20)
SMOKE_FIRE_SMALL_FIRE_MIN_CONTOUR = 50     # Contour tối thiểu cho lửa nhỏ (px², was 15)
SMOKE_FIRE_SMALL_FIRE_PERSIST_THRESH = 8   # Persistence frames cho lửa nhỏ (was 4)
# Blob nhỏ có ít pixel nên hue spread / core ratio thấp hơn lửa lớn → hạ ngưỡng
SMOKE_FIRE_SMALL_FIRE_HUE_VAR_THRESH = 0.06
SMOKE_FIRE_SMALL_FIRE_CORE_RATIO_THRESH = 0.04

SMOKE_FIRE_GLOW_ENABLED = True
# SIẾT CHẶT glow: dải HSV quầng rất rộng (bắt tường nắng, xe cam, da người) nên
# phải đòi nhiều pixel, nhiễu nhiều frame, nhấp nháy rõ, đa sắc rõ mới báo.
SMOKE_FIRE_GLOW_MIN_PIXELS = 400           # Pixel quầng sáng tối thiểu (was 120)
SMOKE_FIRE_GLOW_MIN_CONTOUR = 250          # Contour tối thiểu (px², was 80)
SMOKE_FIRE_GLOW_PERSIST_THRESH = 10        # Persistence frames (was 5)
SMOKE_FIRE_GLOW_FLICKER_THRESH = 0.30      # Nhấp nháy tối thiểu (was 0.15)
SMOKE_FIRE_GLOW_HUE_VAR_THRESH = 0.09      # Đa sắc ấm tối thiểu (was 0.05)

# COOLDOWN: sau khi báo 1 event cho 1 zone, KHÔNG báo lại zone đó trong N frame
# dù tín hiệu vẫn còn. Diệt tình trạng cùng 1 ô báo lửa lặp lại liên tục nhiều
# phút (giữ counter >= ngưỡng → mỗi frame lại emit event). 120 frame ≈ 4s @30fps.
SMOKE_FIRE_ALERT_COOLDOWN = 120