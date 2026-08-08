# SmartCity Prototype - AI Vision Engine

Hệ thống giám sát thông minh đô thị: nhận diện sự kiện từ **nhiều camera HLS** và **dữ liệu cảm biến MQTT** trong thời gian thực, hiển thị cảnh báo trực tiếp lên video.

## Tính năng

- **4 logic phát hiện sự kiện (Rule-based + Temporal Confirm):**
  - `HUMAN_ACTION` — người ngã (`HUMAN_FALL`), vung tay mạnh (`HUMAN_WILD_GESTURE`), xô xát (`HUMAN_CONFLICT`), va chạm người (`PERSON_COLLISION`)
  - `VEHICLE_ACCIDENT` — va chạm (`VEHICLE_COLLISION`), dừng bất thường (`VEHICLE_STOP_ANOMALY`)
  - `SMOKE_FIRE` — cháy (`FIRE_DETECTED`), khói (`SMOKE_DETECTED`) bằng phân tích màu HSV + persistence + region growth
  - `INTRUSION` — xâm nhập vùng cấm (`RESTRICTED_INTRUSION`) bằng điểm trong polygon + thời gian lưu lại (dwell)
- **Detector YOLOv8 + ByteTrack + Pose keypoints** (17 điểm) để phân tích hành vi.
- **Tracking dự đoán (constant velocity)** giữa các frame detect nhằm giảm tải CPU; chỉ detection thật mới ghi lịch sử kinematics để tránh cảnh báo sai thời điểm.
- **Cảnh báo cảm biến MQTT** (bất thường điện áp / quá tải) hiển thị chung với alert AI.
- Chạy **đa luồng**, mỗi camera một thread riêng.
- Hiển thị ROI, bounding box, HUD đếm object và banner cảnh báo trên video.

## Cấu trúc dự án

```
smartcity_prototype/
├── config.py               # Cấu hình stream, model, ROI, ngưỡng sự kiện
├── main_pipeline.py        # Pipeline chính (CameraWorker, vòng lặp xử lý)
├── detector.py             # YOLO detect + ByteTrack + pose
├── tracker/                # Quản lý bộ nhớ object (memory_manager, object_state)
├── events/                 # Các rule phát hiện + confirm + visualizer
│   ├── person_rules.py
│   ├── vehicle_rules.py
│   ├── smoke_fire_rules.py
│   ├── intrusion_rules.py
│   ├── confirm.py
│   ├── classifier.py
│   └── visualizer.py
├── sensors/                # Tiêu thụ dữ liệu MQTT
└── weights/                # Model: yolov8n.pt, yolov8n-pose.pt
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

- `yolov8n.pt` — detect
- `yolov8n-pose.pt` — pose keypoints

```bash
# Tải nhanh bằng ultralytics (sau khi cài dependencies)
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt'); YOLO('yolov8n-pose.pt')"
# Sau đó copy 2 file vào thư mục weights/
```

## Chạy

```bash
python main_pipeline.py
```

- Mỗi cửa sổ hiển thị một camera (`cam09`, `cam10`).
- Nhấn `q` để thoát, `Ctrl+C` để dừng toàn bộ pipeline.

## Cấu hình chính (`config.py`)

| Tham số | Giá trị mặc định | Ý nghĩa |
|---|---|---|
| `CAMERA_STREAMS` | cam09, cam10 | URL HLS/RTSP các camera |
| `DETECTION_INTERVAL` | 3 | Chạy detector mỗi N frame, frame giữa dùng tracking dự đoán |
| `EVENT_CONFIRM_FRAMES` | 3 | Số frame liên tiếp để xác nhận sự kiện trước khi cảnh báo |
| `MODEL_INPUT_SIZE` | (640, 384) | Kích thước input model |
| `CONF_THRESH` | 0.35 | Ngưỡng confidence detection |
| `CAMERA_ROIS` | — | Polygon vùng cấm (INTRUSION) / vùng rủi ro cháy (SMOKE_FIRE) |
| `INTRUSION_DWELL_FRAMES` | 5 | Số frame lưu lại trong vùng cấm để xác nhận xâm nhập |
| `MQTT_*` | — | Broker, topic, user/pass MQTT |

Thay đổi ngưỡng phát hiện (ngã, vung tay, va chạm...) ngay trong `config.py`.

## Các event types

| Event | Mô tả |
|---|---|
| `HUMAN_FALL` | Người ngã (aspect ratio / torso angle + gia tốc rơi) |
| `HUMAN_WILD_GESTURE` | Vung tay mạnh (wrist speed từ pose) |
| `HUMAN_CONFLICT` | Xô xát / giằng co (kinetic + biến thiên khoảng cách) |
| `PERSON_COLLISION` | 2 người tiếp cận nhanh / va chạm |
| `VEHICLE_COLLISION` | Va chạm phương tiện (IoU/proximity + giảm tốc/đổi hướng) |
| `VEHICLE_STOP_ANOMALY` | Xe đang chạy đột ngột dừng bất thường |
| `FIRE_DETECTED` / `SMOKE_DETECTED` | Cháy / khói trong vùng quan sát |
| `RESTRICTED_INTRUSION` | Xâm nhập khu vực cấm |
| `ELECTRICAL_ANOMALY` | Cảnh báo từ sensor MQTT (điện áp/quá tải) |
