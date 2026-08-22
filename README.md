# SmartCity Prototype - AI Vision Engine

Hệ thống giám sát thông minh đô thị: nhận diện sự kiện từ **nhiều camera** trong thời gian thực, hiển thị cảnh báo trực tiếp lên video và web dashboard.

Thiết kế tuân theo **AI PERFORMANCE CONTRACT** (Rule 1–15): 1 model dùng chung, camera chỉ capture, cascade + candidate → confirmation, ngân sách CPU/RAM per-camera có mục tiêu đo được.

## Tính năng

- **4 nhóm logic phát hiện sự kiện (Rule-based scoring + Temporal Confirm):**
  - Người ngã (`HUMAN_FALL`), xô xát/đánh nhau (`HUMAN_CONFLICT`)
  - Va chạm xe-xe (`VEHICLE_COLLISION`) — có gate confidence xe ≥ 0.65 + tín hiệu động học
  - Cháy (`FIRE_DETECTED`) — FireScore hợp nhất: model + spatial + temporal + motion + smoke
  - Khói (`SMOKE_DETECTED`) — SmokeScore: khói phải **lan rộng** (expansion) mới là khói
  - Xâm nhập vùng cấm (`RESTRICTED_INTRUSION`) — depth + movement + dwell
- **Shared models (Rule 1/2):** YOLO detect + pose + YOLO-cls xe dùng chung 1 instance/process cho MỌI camera — không load theo camera.
- **Camera chỉ capture (Rule 2/3):** giữ 1 frame mới nhất; AI chạy theo cadence (5 FPS detect, 1.5 FPS fire/accident), không theo camera FPS.
- **Cascade (Rule 4) + CANDIDATE ≠ EVENT (Rule 7/8):** specialized model chỉ chạy khi có candidate; snapshot/DB/notification CHỈ sau `CONFIRMED`.
- **Embedding async (Rule 9):** MobileNetV3 → FAISS chạy off critical path, không chặn detection.
- **Giới hạn CPU threads (Rule 11):** PyTorch/OpenMP/MKL bị giới hạn (mặc định 4 thread).
- **Dashboard Web (Flask):** alert + snapshot full frame & crop, lọc theo event, tự refresh.
- **Bộ test false-positive (Rule 15):** `python -m unittest discover -s tests -t .`

## Cấu trúc dự án

```
smartcity_prototype/
├── config.py               # Cấu hình stream, model, ROI, ngưỡng score (Rule 6)
├── main.py                 # Capture workers + AI Worker Pool + dashboard + summary
├── detector.py             # YOLO detect + pose (STATELESS — shared)
├── vehicle_classifier.py   # YOLO-cls phân loại tinh loại xe (shared, vote theo track)
├── inference/
│   ├── registry.py         # SharedModelRegistry: 1 model = 1 instance (Rule 1)
│   ├── scheduler.py        # InferenceScheduler: cadence detect/fire/accident (Rule 3)
│   ├── context.py          # CameraContext: latest frame slot + state per-camera
│   └── embedding.py        # MobileNetV3 → FAISS async (Rule 9)
├── tracker/                # Quản lý bộ nhớ object + gán track_id per-camera
├── events/                 # Rule-filters: candidate → confirm → visualizer
│   ├── scores.py           # Fall/Fight/Collision/Fire/Smoke score (Rule 6)
│   ├── classifier.py       # Cascade Level 2+4
│   ├── confirm.py          # EventConfirmTracker (Rule 7)
│   └── ...
├── dashboard/              # Web dashboard: alert + snapshot (store, server)
├── tests/                  # Rule 15: false-positive tests cho từng event
└── weights/                # yolo11n.pt, yolo11n-pose.pt
```

## Cài đặt

Cần Python **3.10+** (khuyến nghị 3.11–3.14).

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

> **torch CPU:** nếu máy không có GPU, cài torch CPU cho nhẹ:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
> pip install -r requirements.txt
> ```

Đặt 2 file weight vào `weights/`:
```bash
python -c "from ultralytics import YOLO; YOLO('yolo11n.pt'); YOLO('yolo11n-pose.pt')"
```
`weights/vehicle_cls.pt` (phân loại xe tinh) là **tuỳ chọn** — thiếu thì pipeline tự vô hiệu hoá. `EMBEDDING_MODEL_PATH` (MobileNetV3) cũng tuỳ chọn — thiếu thì tự tải `MobileNet_V3_Large_Weights.DEFAULT`.

## Chạy

```bash
# 1 camera với GUI video
python main.py cam09

# Chạy TẤT CẢ camera song song (headless — OpenCV HighGUI không thread-safe)
python main.py --all-cameras

# Headless: log + snapshot không cửa sổ video
python main.py cam09 --headless

# Để log stdout hiện ngay khi redirect/benchmark:
python -u main.py cam09 --headless

# Danh sách camera
python main.py
```

- Khi thoát (hoặc `Ctrl+C`), pipeline in **AI PERFORMANCE SUMMARY**: shared models, AI detect FPS/camera, drop rate (cadence-miss), RSS MB/camera so với target, và **so sánh với Rule 13** (CPU avg / RAM / detect FPS / frame-drop theo số camera, đánh dấu `OK`/`WARN`).
- Nhấn `q` để thoát GUI. Với video local, worker tự dừng khi hết video.

## AI PERFORMANCE CONTRACT — cách kiến trúc đáp ứng

| Rule | Yêu cầu | Hiện thực |
|---|---|---|
| 1 | Model KHÔNG load theo camera | `inference/registry.py` singleton + lock serial hoá predict |
| 2 | Camera chỉ capture + latest frame | `CameraContext.push_frame` ghi đè 1 frame; `main.CaptureWorker` không inference |
| 3 | Camera FPS ≠ AI FPS | `AI_DETECT_FPS=5`, `AI_FIRE_FPS=1.5`, `AI_ACCIDENT_FPS=1.5`; tick fire/accident giữa 2 tick detect KHÔNG chạy lại YOLO (detect giữ đúng ≤ 5 FPS) |
| 4 | Cascade | fire/accident chạy ở cadence riêng; object rules chỉ chạy khi có detection mới; pose chạy khi có người aspect ngang; vehicle-cls chỉ khi có track xe |
| 6 | Score formula từng hành vi | `events/scores.py` (weights trong config) |
| 7 | CANDIDATE ≠ EVENT | `events/confirm.py` + `STAGE_CANDIDATE/CONFIRMED` |
| 8 | Snapshot sau CONFIRMED | `dashboard/store.py:record` chặn stage < CONFIRMED |
| 9 | Embedding async | `inference/embedding.py` worker riêng, queue giới hạn |
| 11 | Giới hạn CPU threads | `registry.limit_cpu_threads()` (trước khi load model) |
| 14 | Mọi event có candidate → confirmation | `classifier.evaluate` → `confirmer.process` |
| 15 | Test false-positive mỗi event | `tests/` (chạy bằng unittest) |

Ngân sách RAM (Rule 12): model shared KHÔNG tính lặp lại cho từng camera; per-camera thêm ≤ 300–400 MB (hard ≤ 500 MB) — đo ở summary (`RSS / N camera`).

## Dashboard

Tự bật tại `http://localhost:8080` (poll `/api/alerts`). Mỗi alert gồm event type, mô tả, camera, thời gian, confidence kèm **snapshot full frame + crop** làm bằng chứng. Chỉ alert `CONFIRMED` mới vào dashboard (threshold thật do confirm threshold của từng event).

## Cấu hình chính (`config.py`)

| Nhóm | Tham số | Giá trị mặc định | Ý nghĩa |
|---|---|---|---|
| CPU | `AI_MAX_THREADS` | 4 | PyTorch/OpenMP dùng tối đa N thread (không chiếm 12) |
| CPU | `AI_WORKER_POOL_SIZE` | 2 | Worker pool dùng chung cho mọi camera |
| Cadence | `AI_DETECT_FPS` / `AI_FIRE_FPS` / `AI_ACCIDENT_FPS` | 5 / 1.5 / 1.5 | Điều chỉnh nhịp AI |
| Gate xe | `VEHICLE_COLLISION_MIN_CONF` | 0.65 | Conf xe tối thiểu để vào collision pipeline |
| Score | `FALL_W_*`, `FIGHT_W_*`, `COLLISION_W_*`, `FIRE_W_*`, `SMOKE_W_*` | — | Trọng số công thức (Rule 6), tổng = 1 |
| Confirm | `*_CANDIDATE_THRESH` / `*_CONFIRM_THRESH` / `*_CONFIRM_FRAMES` | — | Cấu hình candidate → confirmed từng event |
| Khói | `SMOKE_MIN_EXPANSION` | 0.05 | Rule 6E: khói phải lan rộng |
| Embedding | `EMBEDDING_ENABLED`, `EMBED_QUEUE_MAXSIZE` | True / 32 | Async embedding |
| Camera | `CAMERA_STREAMS` | cam09, cam10 | Đường dẫn/URL video |

## Tests (Rule 15)

```bash
python -m unittest discover -s tests -t . -v
```

Coverage: false-positive của fall/fight/collision/fire/smoke/intrusion + matrix candidate→confirmed + trọng số score + cadence scheduler (fire/accident không bị nuốt bởi detect tick).

## Event types (Phase 1)

| Event | Cơ chế phát hiện |
|---|---|
| `HUMAN_FALL` | FallScore: posture + vertical motion + aspect change + center velocity + temporal |
| `HUMAN_CONFLICT` | FightScore: person-pair + contact + relative motion + intensity + temporal |
| `VEHICLE_COLLISION` | CollisionScore: track stability + relative velocity + closing + geometry + velocity change |
| `FIRE_DETECTED` | FireScore: model + spatial + temporal + motion (flicker) + smoke corroboration |
| `SMOKE_DETECTED` | SmokeScore: model + temporal + spatial expansion + shape |
| `RESTRICTED_INTRUSION` | depth check + movement + dwell time trong polygon |