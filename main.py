import os
import sys
import time
import argparse
import threading

import cv2
import config
import contract
import inference.registry as registry
from inference.scheduler import InferenceScheduler
from inference.context import CameraContext
from inference.embedding import EmbeddingPipeline
from dashboard.store import AlertStore
from dashboard.server import start_dashboard
from events.visualizer import EventVisualizer


# Console Windows mặc định cp1252 -> print tiếng Việt sẽ crash. Ép UTF-8 + replace.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


# ============================================================
# CAPTURE WORKER — Rule 2/3: camera KHÔNG chạy inference.
# Chỉ làm: capture → resize → giữ latest frame → bỏ frame cũ.
# ============================================================
class CaptureWorker(threading.Thread):
    def __init__(self, camera_id, stream_url, context, headless=False):
        super().__init__(daemon=True, name=f"capture-{camera_id}")
        self.camera_id = camera_id
        self.stream_url = stream_url
        self.context = context
        self.headless = headless
        self.stopped = False
        self._is_file_source = self._looks_like_file(stream_url)

    @staticmethod
    def _looks_like_file(source):
        if "://" in source:
            return False
        return os.path.exists(source)

    def _open_capture(self):
        if self._is_file_source:
            return cv2.VideoCapture(self.stream_url)
        return cv2.VideoCapture(self.stream_url, cv2.CAP_FFMPEG)

    def run(self):
        print(f"[{self.camera_id}] Source: {self.stream_url}"
              + (" (video local)" if self._is_file_source else " (HLS/RTSP)"))
        window_name = f"SmartVision AI Engine - {self.camera_id}"
        frame_idx = 0
        retry_log_printed = False
        cap = None
        last_displayed_idx = -1

        while not self.stopped:
            if cap is None or not cap.isOpened():
                if cap is not None:
                    cap.release()
                cap = self._open_capture()
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                if not cap.isOpened():
                    if not retry_log_printed:
                        print(f"[{self.camera_id}] KHÔNG mở được nguồn: {self.stream_url}")
                        retry_log_printed = True
                    if self._is_file_source:
                        print(f"[{self.camera_id}] Lỗi file local, dừng worker.")
                        break
                    if not self.headless:
                        cv2.waitKey(1)
                    time.sleep(2.0)
                    continue
                retry_log_printed = False

            ret, frame = cap.read()
            if not ret or frame is None:
                if self._is_file_source:
                    print(f"[{self.camera_id}] Hết video / không đọc được frame. Dừng xử lý.")
                    break
                print(f"[{self.camera_id}] Stream drop, retrying in 1s")
                cap.release()
                cap = None
                time.sleep(1.0)
                continue

            frame_idx += 1
            frame_resized = cv2.resize(frame, config.MODEL_INPUT_SIZE)

            # Rule 2: CHỈ đưa latest frame vào context (capture-only)
            self.context.push_frame(frame_resized, frame_idx)

            # GUI: hiển thị frame AI đã vẽ (capture không tự vẽ/chạy model)
            if not self.headless:
                display = self.context.last_annotated
                if display is None:
                    display = frame_resized
                cv2.imshow(window_name, display)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    self.stopped = True
                    break

        if cap is not None:
            cap.release()
        if not self.headless:
            try:
                cv2.destroyWindow(window_name)
            except cv2.error:
                pass
        # Thông báo dừng: đẩy sentinel để scheduler không chờ
        self.stopped = True


# ============================================================
# EVENT HANDLER — Rule 7: SNAPSHOT CHỈ sau CONFIRMED.
# candidate KHÔNG: snapshot / DB / notification.
# ============================================================
def make_event_handler(alert_store, embedding):
    visualizer = EventVisualizer()
    # Dedup theo camera: tránh spam alert cho cùng 1 sự kiện lặp nhiều pass
    active_keys = {}          # camera_id -> set key
    current_keys = {}         # camera_id -> set key (record tại pass này)

    def handler(camera_id, confirmed_events, frame, memory_mgr):
        keys = active_keys.setdefault(camera_id, set())
        cur = current_keys.setdefault(camera_id, set())
        for ev in confirmed_events:
            if not ev.get("stage") == config.STAGE_CONFIRMED:
                continue  # phòng thủ: chỉ CONFIRMED được đưa dashboard
            key = (ev["event_type"],
                   tuple(ev.get("track_ids", [])),
                   ev.get("zone_name"))
            cur.add(key)
            if key in keys:
                continue  # đã cảnh báo sự kiện này
            conf = ev.get("confidence", 0.0)
            print(f"[ALERT:{camera_id}] {ev['event_type']}_{config.STAGE_CONFIRMED} | "
                  f"{ev['description']} | score={conf:.2f}")

            # Vẽ lên frame để làm bằng chứng (snapshot)
            annotated = visualizer.draw(
                frame, memory_mgr, [ev], camera_id=camera_id)
            alert = alert_store.record(camera_id, ev, annotated)
            if alert is not None:
                print(f"[ALERT-IMG:{camera_id}] Snapshot CONFIRMED: "
                      f"{alert['snapshot']} (crop: {alert['crop']})")

            # Rule 9: embedding ASYNC — gửi crop vào queue, không chặn loop
            if embedding is not None and ev.get("bbox") is not None:
                crop = _crop_bbox(frame, ev["bbox"])
                if crop is not None:
                    embedding.submit(
                        crop, {"camera_id": camera_id,
                               "event_type": ev["event_type"]})
        # giải phóng key không còn xuất hiện -> cho phép alert lại khi tái diễn
        keys &= cur
        cur.clear()
        return None

    return handler


def _crop_bbox(frame_bgr, bbox):
    try:
        x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
    except (TypeError, ValueError):
        return None
    h, w = frame_bgr.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return frame_bgr[y1:y2, x1:x2]


def main():
    print("=" * 60)
    print(" SMARTVISION AI ENGINE - PERFORMANCE CONTRACT (shared models)")
    print("=" * 60)

    parser = argparse.ArgumentParser(
        description="SmartVision AI - camera capture / shared AI worker pool.")
    parser.add_argument("camera_id", nargs="?", help="ID camera cần chạy.")
    parser.add_argument("--video", default=None,
                        help="Video local nếu không dùng stream của camera.")
    parser.add_argument("--headless", action="store_true",
                        help="Không cửa sổ video: chỉ log + snapshot.")
    parser.add_argument("--all-cameras", action="store_true",
                        help="Chạy tất cả camera trong config.CAMERA_STREAMS.")
    args = parser.parse_args()

    if not args.camera_id and not args.all_cameras:
        print("[SYSTEM] Chưa chọn camera. Chạy riêng từng camera:")
        for cid in config.CAMERA_STREAMS:
            print(f"    python main.py {cid}")
        print("[SYSTEM] Hoặc chạy tất cả camera song song:")
        print("    python main.py --all-cameras")
        return

    if args.camera_id and args.camera_id not in config.CAMERA_STREAMS:
        print(f"[SYSTEM] Không tìm thấy camera '{args.camera_id}'. "
              f"Có: {list(config.CAMERA_STREAMS.keys())}")
        return

    if args.video and not os.path.exists(args.video):
        print(f"[SYSTEM] Không tìm thấy file video: {args.video}")
        return

    if args.headless:
        config.HEADLESS = True
    headless = config.HEADLESS

    # Rule 11: giới hạn CPU threads TRƯỚC khi load model
    registry.limit_cpu_threads()

    # AI Performance Contract (Video2Action): kiểm tra cấu hình có vi phạm
    # hợp đồng (model shared, FPS <= 5, giới hạn thread, RAM/CPU target...).
    print(contract.report(config))

    # Prime psutil CPU% (lần gọi đầu trả 0) — để summary đo chuẩn.
    try:
        import psutil
        _PROC = psutil.Process()
        _PROC.cpu_percent(interval=None)
    except Exception:
        _PROC = None

    # Rule 9: embedding async pipeline (off critical path)
    embedding = EmbeddingPipeline()
    embedding.start()

    # Dashboard + AlertStore (snapshot)
    alert_store = AlertStore(
        snapshot_dir=config.SNAPSHOT_DIR,
        max_alerts=config.SNAPSHOT_MAX_ALERTS,
    )
    start_dashboard(alert_store, config.SNAPSHOT_DIR, enabled=config.DASHBOARD_ENABLED)

    # Chọn cameras
    if args.all_cameras:
        sources = {cid: url for cid, url in config.CAMERA_STREAMS.items()}
    else:
        sources = {args.camera_id: (args.video or config.CAMERA_STREAMS[args.camera_id])}

    # Tạo context cho từng camera (state per camera; model SHARED)
    contexts = {cid: CameraContext(cid, url) for cid, url in sources.items()}

    # OpenCV HighGUI `imshow`/`waitKey` không thread-safe giữa các thread
    # capture -> nhiều camera + GUI sẽ crash/treo. Đa camera = benchmark headless.
    if len(contexts) > 1 and not headless:
        print("[SYSTEM] Chạy nhiều camera → tắt cửa sổ GUI "
              "(OpenCV HighGUI không thread-safe). Chạy 1 camera để xem video.")
        headless = True

    # Scheduler (time-based) + worker pool SHARED cho tất cả camera
    scheduler = InferenceScheduler(
        contexts,
        on_event=make_event_handler(alert_store, embedding),
    )
    scheduler.start()

    # Capture workers (chỉ capture, không inference)
    workers = []
    for cid, ctx in contexts.items():
        worker = CaptureWorker(cid, sources[cid], ctx, headless=headless)
        worker.start()
        workers.append(worker)

    print(f"[SYSTEM] Đang chạy {len(workers)} camera. "
          f"AI detect {config.AI_DETECT_FPS} FPS - worker pool {config.AI_WORKER_POOL_SIZE}.")

    try:
        while any(w.is_alive() for w in workers):
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[SYSTEM] Stopping pipeline...")
        for w in workers:
            w.stopped = True
    finally:
        scheduler.stop()
        embedding.stop()

    # Benchmark summary — §13: chỉ số đã tối ưu (CPU %, RAM/camera, ...)
    print("\n[SYSTEM] === AI PERFORMANCE SUMMARY ===")
    print(f"  Shared models: 1 process (YOLO+pose+vehicle_cls) — KHÔNG load theo camera")
    print(f"  AI detect target: {config.AI_DETECT_FPS} FPS/camera | worker pool: "
          f"{config.AI_WORKER_POOL_SIZE} | torch threads: {config.AI_MAX_THREADS}")

    n_cams = max(len(contexts), 1)
    cpu_pct = 0.0
    rss = 0.0
    try:
        import psutil
        proc = _PROC or psutil.Process()
        cpu_pct = proc.cpu_percent(interval=0.5)
        rss = proc.memory_info().rss / 1024 / 1024
        per_cam_ram = rss / n_cams
        print(f"  PROCESS: CPU {cpu_pct:.1f}% | RSS {rss:.0f} MB | "
              f"≈ {per_cam_ram:.0f} MB/camera "
              f"(target ≤ {config.PER_CAMERA_RAM_TARGET_MB} MB, "
              f"hard ≤ {config.PER_CAMERA_RAM_HARD_MB} MB)")
    except Exception:
        pass

    # Rule 13 — so sánh với mục tiêu benchmark theo số camera
    def _bench(label, got, target, op="<=", unit=""):
        if target is None:
            print(f"    {label}: {got:.0f}{unit} (phải benchmark thêm — chưa có mục tiêu)")
            return
        ok = (got <= target) if op == "<=" else (got >= target)
        mark = "OK" if ok else "WARN"
        print(f"    {label}: {got:.1f}{unit} vs {target:.2f}{unit} [{mark}]")

    # psutil cpu_percent() cho process trả % theo 1 core (100% = 1 thread).
    # Rule 10/13 mục tiêu tính theo TOÀN MÁY (12 logical threads):
    #   frac = cpu_pct / (100 * CPU_LOGICAL_THREADS)
    n_threads = max(int(getattr(config, "CPU_LOGICAL_THREADS", 12)), 1)
    cpu_frac = cpu_pct / (100.0 * n_threads)
    ram_gb = rss / 1024.0
    print(f"  BENCHMARK (n={len(contexts)} camera):")
    _bench("CPU avg   ", cpu_frac, config.BENCH_CPU_TARGET.get(len(contexts)))
    _bench("RAM       ", ram_gb, config.BENCH_RAM_TARGET_GB.get(len(contexts)), unit="GB")
    for cid, ctx in sorted(contexts.items()):
        drop = ctx.drop_rate()
        print(f"  [{cid}] AI detect fps={ctx.fps():.2f} | "
              f"cadence-drop={drop*100:.1f}% | "
              f"alerts={len(alert_store.alerts)}")
        _bench(f"  drop {cid}", drop, config.BENCH_FRAME_DROP_MAX.get(len(contexts)), unit="")
        _bench(f"  fps  {cid}", ctx.fps(), config.BENCH_DETECT_FPS_MIN, op=">=", unit="")


if __name__ == "__main__":
    main()