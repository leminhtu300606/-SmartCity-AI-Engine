"""InferenceScheduler — Rule 2/3: camera không tự chạy inference.

Camera capture     -> context.latest_frame (1 frame)           [kernel thread]
Scheduler (time)   -> mỗi camera: check cadence (detect/fire/accident)
                     -> submit job vào AI Worker Pool (shared models)
AI Worker Pool     -> chạy detection + cascade quyết định sự kiện.

Scheduler GIỚI HẠN số job/camera đang chạy = 1 và nhịp time-based,
nên AI không bao giờ phải "bắt kịp" camera FPS (Rule 3).
"""
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import config
import inference.registry as registry
from events.classifier import RuleBasedEventClassifier
from events.visualizer import EventVisualizer
from inference.context import CameraContext


class InferenceScheduler:
    def __init__(
        self,
        contexts: dict[str, CameraContext],
        on_event: Callable | None = None,
    ):
        """contexts: dict camera_id -> CameraContext
        on_event: callback(camera_id, confirmed_event, frame, memory_mgr) —
                  gọi khi event CONFIRMED (để snapshot/log/dashboard).
        """
        self.contexts = contexts
        self.on_event = on_event
        self.detect_interval = 1.0 / max(config.AI_DETECT_FPS, 0.1)
        self.fire_interval = 1.0 / max(config.AI_FIRE_FPS, 0.1)
        self.accident_interval = 1.0 / max(config.AI_ACCIDENT_FPS, 0.1)

        self._lock = threading.Lock()
        self._inflight = {cid: False for cid in contexts}
        self._stopped = False
        self._visualizer = EventVisualizer()
        self._classifiers: dict[str, RuleBasedEventClassifier] = {}
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="ai-scheduler")
        self.pool = ThreadPoolExecutor(
            max_workers=config.AI_WORKER_POOL_SIZE,
            thread_name_prefix="ai-worker")

    # ------------------------------------------------------------
    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stopped = True
        self.pool.shutdown(wait=False)

    def _loop(self) -> None:
        while not self._stopped:
            now = time.time()
            for cid, ctx in self.contexts.items():
                self._maybe_schedule(cid, ctx, now)
            time.sleep(0.005)

    def _maybe_schedule(self, cid: str, ctx: CameraContext, now: float) -> None:
        if not ctx.has_new_frame:
            return
        due_detect = now >= ctx.next_detect_at
        due_fire = now >= ctx.next_fire_at
        due_accident = now >= ctx.next_accident_at

        if not (due_detect or due_fire or due_accident):
            return

        with self._lock:
            if self._inflight.get(cid, False):
                # Miss cadence: job trước chưa xong khi 1 tick đến hạn ->
                # đếm ĐÚNG 1 lần + advance timer của stage để tick này không
                # bị đếm lại ở vòng loop kế (5ms).
                if due_detect:
                    ctx.next_detect_at = now + self.detect_interval
                    ctx.record_missed_cadence()
                if due_fire:
                    ctx.next_fire_at = now + self.fire_interval
                if due_accident:
                    ctx.next_accident_at = now + self.accident_interval
                return
            self._inflight[cid] = True

        # Rule 3: cadence ĐỘC LẬP — chỉ advance timer của stage thực sự đến hạn.
        if due_detect:
            ctx.next_detect_at = now + self.detect_interval
        if due_fire:
            ctx.next_fire_at = now + self.fire_interval
        if due_accident:
            ctx.next_accident_at = now + self.accident_interval

        frame, idx, _ = ctx.take_latest()
        if frame is None:
            self._inflight[cid] = False
            return
        self.pool.submit(
            self._process, cid, ctx, frame, idx,
            due_detect, due_fire, due_accident)

    # ------------------------------------------------------------
    def _process(self, cid: str, ctx: CameraContext, frame: Any, idx: int,
                 run_detect: bool, run_fire: bool, run_vehicle: bool) -> None:
        try:
            self._run_pipeline(cid, ctx, frame, idx, run_detect, run_fire,
                               run_vehicle)
        except Exception as e:
            print(f"[AI:{cid}] Lỗi worker: {e}")
        finally:
            ctx.mark_ai_processed(idx)
            with self._lock:
                self._inflight[cid] = False

    def _run_pipeline(self, cid: str, ctx: CameraContext, frame: Any, idx: int,
                      run_detect: bool, run_fire: bool, run_vehicle: bool) -> None:
        # Rule 3: detection CHỈ chạy khi tick detect đến hạn (<=5 FPS).
        # Tick fire/accident đến hạn mà detect chưa đến hạn -> chỉ chạy
        # specialized stage, KHÔNG chạy thêm YOLO detect (giữ đúng 5 FPS).
        if run_detect:
            detector = registry.get_detector()

            # LEVEL 1 — object detection (shared YOLO, stateless). 5 FPS (Rule 3).
            # predict được serial hoá qua registry lock (1 model, KHÔNG thread-safe).
            with registry.get_inference_lock():
                dets = detector.predict(frame)
                ctx.detect_count += 1

                # LEVEL 3 — pose cho candidate người (chỉ khi có người)
                if getattr(config, "ENABLE_POSE", True):
                    persons = [d for d in dets if d["cls_id"] == 0]
                    if persons:
                        detector.predict_pose(frame, dets)

            # Track association per camera (shared YOLO không giữ tracker id)
            ctx.memory_mgr.update_detections(dets, idx, time.time())

            # LEVEL 3 — vehicle fine-grained classifier (model shared) cho track xe
            self._classify_vehicles(cid, ctx, frame)

        # LEVEL 2+4 — candidate rules + temporal confirm (Rule 4/6/7)
        classifier = self._get_classifier(cid)
        confirmed = classifier.evaluate(
            frame_bgr=frame,
            memory_manager=ctx.memory_mgr,
            frame_idx=idx,
            run_detect=run_detect,
            run_fire=run_fire,
            run_vehicle=run_vehicle,
            camera_id=cid,
        )

        # Vẽ overlay một lần mỗi AI pass (GUI hiển thị frame đã vẽ)
        ctx.last_annotated = self._visualizer.draw(
            frame, ctx.memory_mgr, confirmed, camera_id=cid)

        # Chỉ CONFIRMED -> event (Rule 7: candidate không hiển thị)
        if confirmed and self.on_event is not None:
            self.on_event(cid, confirmed, frame, ctx.memory_mgr)

    def _classify_vehicles(self, cid: str, ctx: CameraContext, frame: Any) -> None:
        """Level 3: vehicle fine-grained cls cho track xe (model shared, serial hoá)."""
        vc = registry.get_vehicle_classifier()
        if not vc.available:
            return
        vehicles = [o for o in ctx.memory_mgr.visible_objects().values()
                    if o.cls_id in config.VEHICLE_CLASSES and len(o.bbox_history)]
        with registry.get_inference_lock():
            for obj in vehicles:
                vtype, vconf = vc.classify(
                    frame, obj.bbox_history[-1],
                    track_id=obj.track_id, camera_id=cid)
                if vtype is not None and vconf >= config.VEHICLE_CLS_MIN_CONF:
                    obj.vehicle_type = vtype
            vc.prune_votes([o.track_id for o in vehicles], camera_id=cid)

    # ------------------------------------------------------------
    # Classifier per camera (state per camera, Rule 10)
    # ------------------------------------------------------------
    def _get_classifier(self, cid: str) -> RuleBasedEventClassifier:
        if cid not in self._classifiers:
            self._classifiers[cid] = RuleBasedEventClassifier(camera_id=cid)
        return self._classifiers[cid]
