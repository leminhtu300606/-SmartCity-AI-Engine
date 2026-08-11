# SmartCity Prototype - AI Vision Engine

Hệ thống giám sát thông minh đô thị: nhận diện sự kiện từ **nhiều camera HLS** trong thời gian thực, hiển thị cảnh báo trực tiếp lên video.

## Tính năng

- **4 logic phát hiện sự kiện (Rule-based + Temporal Confirm) — Phase 1:**
  - Person fall / conflict — người ngã (`HUMAN_FALL`), xô xát (`HUMAN_CONFLICT`)
  - Vehicle collision — va chạm xe-xe (`VEHICLE_COLLISION`)
  - Smoke / Fire — cháy (`FIRE_DETECTED`): phân tích màu HSV + tương phản sáng trên nền tối + tần số nhấp nháy lửa + khói là điều kiện hỗ trợ
  - Restricted-zone intrusion — xâm nhập vùng cấm (`RESTRICTED_INTRUSION`) bằng điểm trong polygon + thời gian lưu lại (dwell)
- **Detector YOLO11n + ByteTrack + Pose keypoints** (17 điểm) để phân tích hành vi.
- **Tracking dự đoán (constant velocity)** giữa các frame detect nhằm giảm tải CPU; chỉ detection thật mới ghi lịch sử kinematics để tránh cảnh báo sai thời điểm.
- **Dashboard Web** (Flask): hiển thị danh sách cảnh báo kèm **ảnh snapshot** chụp lại đúng lý do gây cảnh báo, lọc theo event type, tự refresh 2s.
- Chạy **1 nguồn mỗi lần** (single camera hoặc video local).
- Hiển thị ROI, bounding box, HUD đếm object và banner cảnh báo trên video.

## Cấu trúc dự án

```
smartcity_prototype/
├── config.py               # Cấu hình stream, model, ROI, ngưỡng sự kiện
├── main.py                 # Pipeline chính (CameraWorker, vòng lặp xử lý)
├── detector.py             # YOLO detect + ByteTrack + pose├── tracker/                # Quản lý bộ nhớ object (memory_manager, object_state)
├── events/                 # Các rule phát hiện + confirm + visualizer
│   ├── person_rules.py
│   ├── vehicle_rules.py
│   ├── smoke_fire_rules.py
│   ├── intrusion_rules.py
│   ├── confirm.py
│   ├── classifier.py
│   └── visualizer.py
├── dashboard/              # Web dashboard: alert + snapshot (store, server)
└── weights/                # Model: yolo11n.pt, yolo11n-pose.pt
```

## Cài đặt

### 1. Python & venv

Cần Python **3.10+** (khuyến nghị 3.11–3.14).

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/macOS
python3 -m venv venv
source venv/bin/activate
```

### 2. Cài dependencies

```bash
pip install -r requirements.txt
```

> **Lưu ý torch CPU:** `requirements.txt` cài torch (bản mặc định, có thể kèm CUDA). Nếu máy không có GPU, cài torch CPU cho nhẹ và nhanh:
>
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
> pip install -r requirements.txt
> ```

### 3. Tải model

Đặt 2 file weight vào thư mục `weights/`:

- `yolo11n.pt` — detect
- `yolo11n-pose.pt` — pose keypoints

```bash
# Tải nhanh bằng ultralytics (sau khi cài dependencies)
python -c "from ultralytics import YOLO; YOLO('yolo11n.pt'); YOLO('yolo11n-pose.pt')"
# Sau đó copy 2 file vào thư mục weights/
```

## Chạy

Chạy **riêng từng camera/video** (mỗi lần 1 nguồn), hiển thị trực tiếp trên cửa sổ OpenCV:

```bash
# Chạy stream HLS của 1 camera
python main.py cam09

# Chạy camera khác
python main.py cam10

# Phân tích 1 video local (vẫn áp dụng ROI/rule của cam09)
python main.py cam09 --video path/to/video.mp4

# Chạy HEADLESS: không cửa sổ video, chỉ log cảnh báo + chụp ảnh vị trí cảnh báo
python main.py cam09 --headless

# Xem danh sách camera / hướng dẫn
python main.py
```

- Mỗi lần chạy chỉ xử lý **1 nguồn** duy nhất. GUI mode hiển thị ROI, bounding box, HUD đếm object và banner cảnh báo trực tiếp trên video.
- **Headless mode (`--headless`)**: không mở cửa sổ video, chỉ in cảnh báo ra console + ghi `snapshots/alerts.log`, đồng thời **tự chụp ảnh vị trí cảnh báo** (snapshot toàn frame + ảnh `_crop` cắt đúng vùng bbox).
- Nhấn `q` để thoát, `Ctrl+C` để dừng. Với video local, pipeline tự dừng khi hết video.

### Dashboard cảnh báo

Khi chạy, pipeline tự bật **web dashboard** tại `http://localhost:8080`:

- **Danh sách cảnh báo**: mỗi alert gồm event type, mô tả, camera, thời gian, độ tin cậy.
- **Ảnh snapshot**: mỗi lần có cảnh báo mới, hệ thống tự chụp lại frame (đã vẽ bounding box / vùng nghi vấn) và lưu vào `snapshots/` — dùng làm **bằng chứng lý do cảnh báo**.
- Lọc theo từng loại event (người ngã, xô xát, xe va chạm, lửa/khói...), tự refresh mỗi 2 giây.
- **Chỉ đưa cảnh báo có `confidence >= MIN_ALERT_CONFIDENCE` (mặc định 0.9)** — loại bỏ alert có độ tin cậy thấp.

Tắt/bật trong `config.py`:

| Tham số | Giá trị mặc định | Ý nghĩa |
|---|---|---|
| `DASHBOARD_ENABLED` | True | Bật/tắt dashboard |
| `DASHBOARD_HOST` / `DASHBOARD_PORT` | `0.0.0.0` / `8080` | Địa chỉ truy cập dashboard |
| `SNAPSHOT_DIR` | `snapshots/` | Thư mục lưu ảnh snapshot |
| `SNAPSHOT_CROP_MARGIN` | `0.4` | Mở rộng vùng cắt ảnh `_crop` quanh bbox cảnh báo |
| `ALERT_LOG_FILE` | `alerts.log` | File log cảnh báo (mỗi alert 1 dòng kèm vị trí bbox) |
| `HEADLESS` | `False` | Chạy không cửa sổ video (có thể bật qua `--headless`) |
| `MIN_ALERT_CONFIDENCE` | `0.9` | Ngưỡng confidence tối thiểu để đưa alert |

> Cần cài `flask` (đã thêm vào `requirements.txt`).

## Cấu hình chính (`config.py`)

| Tham số | Giá trị mặc định | Ý nghĩa |
|---|---|---|
| `CAMERA_STREAMS` | cam09, cam10 | URL HLS/RTSP các camera |
| `DETECTION_INTERVAL` | 3 | Chạy detector mỗi N frame, frame giữa dùng tracking dự đoán |
| `EVENT_CONFIRM_FRAMES` | 3 | Số frame liên tiếp để xác nhận sự kiện trước khi cảnh báo |
| `MODEL_INPUT_SIZE` | (640, 384) | Kích thước input model |
| `CONF_THRESH` | 0.35 | Ngưỡng confidence detection || `CAMERA_ROIS` | — | Polygon vùng cấm (INTRUSION) / vùng rủi ro cháy (SMOKE_FIRE) |
| `INTRUSION_DWELL_FRAMES` | 5 | Số frame lưu lại trong vùng cấm để xác nhận xâm nhập |

Thay đổi ngưỡng phát hiện (ngã, xô xát, va chạm...) ngay trong `config.py`.

## Các event types (Phase 1)

| Event | Mô tả |
|---|---|
| `HUMAN_FALL` | Người ngã (aspect ratio / torso angle + gia tốc rơi) |
| `HUMAN_CONFLICT` | Xô xát / giằng co (kinetic + biến thiên khoảng cách) |
| `VEHICLE_COLLISION` | Va chạm xe-xe (proximity + tình trạng xe: giảm tốc/đổi hướng/nghiêng lật) |
| `FIRE_DETECTED` | Cháy trong vùng quan sát (khói là điều kiện hỗ trợ) |
| `RESTRICTED_INTRUSION` | Xâm nhập khu vực cấm |
