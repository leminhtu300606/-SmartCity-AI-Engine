# SmartVision AI Engine

Hệ thống nhận diện sự kiện bất thường từ camera giám sát đô thị thông minh (Phase 1) bằng
detection + tracking + phân tích temporal, xuất cảnh báo qua MQTT.

## Tính năng

Phase 1 hỗ trợ 4 nhóm sự kiện:

| Event | Mô tả | Cơ chế |
|---|---|---|
| `HUMAN_FALL` | Người ngã | Vận tốc rơi dọc + aspect ratio + góc nghiêng thân (pose) theo temporal history |
| `HUMAN_CONFLICT` | Xô xát / đánh nhau | Khoảng cách tương đối + tốc độ cổ tay (pose) + variance quỹ đạo giữa các cặp người |
| `VEHICLE_ACCIDENT` | Va chạm giao thông | BBox overlap + giảm tốc đột ngột trên nhiều frame |
| `ZONE_INTRUSION` | Xâm nhập vùng cấm | Person đi vào ROI + dwell time (số frame tồn tại trong vùng) |

## Kiến trúc Pipeline

```
[Camera (HLS/RTSP)] ──> ThreadedFrameReader (queue nhỏ, drop frame cũ)
                              │
                              v
                    [Resize / Preprocess]
                              │
                              v
                [Detection mỗi SKIP_FRAMES]  (YOLOv8n + ByteTrack persist)
                              │
                    ┌─────────┴─────────┐
                    v                   v
            [Pose Analysis]      [Temporal Memory / History]
            (YOLOv8n-pose)       (bbox, center, velocity, pose, dwell_time)
                    └─────────┬─────────┘
                              v
                   [Rule-Based Event Classifier]
                              │
                              v
                 [MQTT Publish + Cooldown]
```

- **Detection**: chạy mỗi `SKIP_FRAMES` frame (mặc định 4), resize về `INFERENCE_SIZE`.
- **Tracking**: ByteTrack với `persist=True`, giữ track ID ổn định.
- **Temporal Memory**: mỗi track lưu history ngắn (~1-1.5s) trong RAM, không lưu video dài.
- **Event**: rule-based, có cooldown chống spam cho cùng 1 track + loại event.

## Cấu trúc thư mục

```
├── main.py                  # Entry point: khởi tạo model, pipeline, vòng lặp hiển thị
├── config.py                # Toàn bộ cấu hình (stream, model, MQTT, ngưỡng event, ROI)
├── requirements.txt         # Các dependency chính
│
├── core/                    # Pipeline xử lý chính
│   ├── frame_reader.py      #   Đọc stream bằng thread riêng, giữ buffer ngắn
│   ├── pipeline.py          #   CameraPipelineWorker: detect → track → pose → event
│   └── visualizer.py        #   Vẽ ROI, bbox, alerts, FPS lên frame
│
├── tracking/                # Theo dõi & temporal state
│   └── temporal_memory.py   #   TrackedObjectState + TemporalMemoryManager (history ngắn)
│
├── events/                  # Rule-based event classifier (tách theo nhóm sự kiện)
│   ├── classifier.py        #   RuleBasedEventClassifier: điều phối các nhóm rule
│   ├── person_rules.py      #   Fall + Conflict (xô xát/đánh nhau)
│   ├── vehicle_rules.py     #   Vehicle accident (va chạm giao thông)
│   └── intrusion_rules.py   #   Xâm nhập vùng cấm (ROI + dwell time)
│
├── output/                  # Xuất cảnh báo
│   └── mqtt_publisher.py    #   Push JSON qua MQTT kèm cooldown
│
├── weights/                 # Model files
│   ├── yolov8n.pt           #   Detector
│   └── yolov8n-pose.pt      #   Model pose (fall / wrist speed)
│
└── venv/                    # Python environment

## Cài đặt

```bash
pip install -r requirements.txt
```

Phụ thuộc chính: `ultralytics`, `torch`, `opencv-python`, `numpy`, `paho-mqtt`.

## Cấu hình

Sửa `config.py`:

```python
# Stream camera HLS/RTSP
CAMERA_STREAMS = {
    "cam09": "https://.../index.m3u8",
}

# Vùng cấm (nhiều zone/camera, mỗi zone có tên)
RESTRICTED_ROIS = {
    "cam09": [
        {"name": "ZONE_1", "polygon": [[100, 100], [400, 100], [400, 400], [100, 400]]},
        {"name": "ZONE_2", "polygon": [[450, 50], [600, 50], [600, 300], [450, 300]]},
    ],
}

# MQTT
MQTT_HOST = "mqtt.dathoc.net"
MQTT_TOPIC_PREFIX = "smartcity/vision/alerts"

# Ngưỡng event (xem chú thích trong file)
INTRUSION_DWELL_FRAMES = 15
FALL_V_Y_THRESH = 1.8
...
```

## Chạy

```bash
python main.py
```

Nhấn `q` trên cửa sổ video để thoát.

## Payload MQTT

Topic: `{MQTT_TOPIC_PREFIX}/{camera_id}` (QoS 1)

```json
{
  "event_id": "uuid",
  "schema_version": "1.0",
  "source": "smartvision_ai_engine",
  "camera_id": "cam09",
  "timestamp_iso": "2026-08-08T00:00:00.000Z",
  "event_type": "HUMAN_FALL",
  "confidence": 0.9,
  "track_ids": [3],
  "description": "Phát hiện người bị ngã"
}
```

## Ghi chú Phase 1

- **Chưa hỗ trợ Smoke/Fire** — đây là nhóm event quan trọng cần bổ sung ở giai đoạn sau
  (detector chuyên dụng hoặc heuristic màu + temporal persistence + region growth).
- **Vehicle collision đang ở mức tối giản**: bbox overlap + giảm tốc đột ngột. Chưa có
  relative velocity, direction change, proximity/near-miss, vật thể rơi vào xe, xe dừng bất thường.
- Các ngưỡng event (`FALL_*`, `CONFLICT_*`, `ACCIDENT_*`, `INTRUSION_DWELL_FRAMES`) là giá trị
  khởi đầu, cần benchmark và tinh chỉnh theo dữ liệu camera thực tế.
- Không đưa ra số liệu hiệu năng (FPS, latency, VRAM...) trước khi benchmark thực tế.

## Hướng phát triển

- Bổ sung Smoke/Fire event.
- Hoàn thiện logic vehicle accident (relative velocity, trajectory, temporal confirmation).
- Bổ sung bước Event Score / Confirmation để giảm false positive.
- Đối chiếu pose keypoint với track ID của detector (tránh gán sai người).
- Benchmark: FPS, latency P50/P95, GPU/VRAM/CPU, Precision/Recall, FP/FN, event latency.
- Về sau: cross-camera event fusion, identity/whitelist cho intrusion.
