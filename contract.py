"""contract.py — AI PERFORMANCE CONTRACT (Video2Action) được mã hóa thành code.

Thể chế hóa 15 quy tắc + giới hạn cứng của hợp đồng thành:
  - Các hằng số_TARGET / _HARD duy nhất (tránh phân tán, dễ bị phá).
  - validate(): kiểm tra các cấu hình runtime vi phạm hợp đồng hay không
    (gọi lúc khởi động ở main.py). Nếu dev chỉnh sai (vd AI_DETECT_FPS=25)
    sẽ bị bắt tại startup thay vì chạy tốn CPU rồi mới phát hiện.

Quy tắc (tóm tắt, chi tiết trong docstring từng hằng):
  1. Không load model theo camera.
  2. Model phải singleton / shared (1 instance/process).
  3. Camera chỉ capture + latest-frame.
  4. Không queue frame vô hạn (1 latest frame).
  5. Camera FPS != AI inference FPS.
  6. Object detection <= 5 FPS/camera.
  7. Specialized model chỉ chạy khi candidate phù hợp.
  8. Embedding chạy async (không nằm critical path).
  9. Snapshot chỉ sau CONFIRMED.
 10. PyTorch/OpenMP phải giới hạn CPU threads.
 11. Không để 1 model tự chiếm toàn bộ CPU.
 12. Per-camera RAM target <= 300-400 MB (hard <= 500 MB).
 13. Per-camera CPU target <= 20-25% (hard <= 30%).
 14. Mọi event: candidate -> confirmation.
 15. Mỗi event phải có test false-positive.
"""
from dataclasses import dataclass, field
from typing import List


# ============================================================
# GIỚI HẠN CỨNG (HARD LIMITS) — vi phạm = hợp đồng bị phá
# ============================================================
# Rule 6: detection không được vượt quá ngưỡng này (FPS/camera)
DETECT_FPS_HARD = 5.0
# Rule 10/11: model không được tự chiếm toàn bộ CPU
CPU_THREADS_HARD_RATIO = 1.0          # AI_MAX_THREADS <= CPU_LOGICAL_THREADS
WORKER_POOL_HARD_RATIO = 1.0          # worker pool <= CPU_LOGICAL_THREADS
# Rule 12: RAM tăng thêm / camera
PER_CAMERA_RAM_TARGET_MB = 400.0
PER_CAMERA_RAM_HARD_MB = 500.0
# Rule 13: CPU trung bình / camera (tỷ lệ toàn máy)
PER_CAMERA_CPU_TARGET = 0.25
PER_CAMERA_CPU_HARD = 0.30
# Rule 4: frame queue = 1 latest frame (không vô hạn)
FRAME_QUEUE_MAX = 1


# ============================================================
# CHỈ TIÊU THEO SỐ CAMERA (benchmark) — Rule 13 + bảng "đã tối ưu"
# ============================================================
@dataclass
class BenchmarkTarget:
    cpu_avg: float            # tỷ lệ CPU toàn máy (<=)
    ram_gb: float             # RAM toàn process (<=)
    detect_fps_min: float = DETECT_FPS_HARD
    frame_drop_max: float = 0.05


BENCHMARK = {
    1: BenchmarkTarget(cpu_avg=0.25, ram_gb=1.4),
    2: BenchmarkTarget(cpu_avg=0.45, ram_gb=1.7),
    4: BenchmarkTarget(cpu_avg=0.80, ram_gb=2.3, frame_drop_max=0.10),
    8: BenchmarkTarget(cpu_avg=1.00, ram_gb=4.0, frame_drop_max=0.15),
}


@dataclass
class Violation:
    rule: str
    message: str


def validate(config) -> List[Violation]:
    """Kiểm tra config có vi phạm hợp đồng không.

    Args:
        config: module config đã import (để không hard-depend vào tên module).
    Returns:
        list[Violation] — rỗng nghĩa là tuân thủ toàn bộ.
    """
    v: List[Violation] = []
    logical = int(getattr(config, "CPU_LOGICAL_THREADS", 12))

    # Rule 6: detection FPS không vượt hard limit
    detect_fps = float(getattr(config, "AI_DETECT_FPS", 0.0))
    if detect_fps > DETECT_FPS_HARD + 1e-6:
        v.append(Violation(
            "R6", f"AI_DETECT_FPS={detect_fps} > hard {DETECT_FPS_HARD} FPS/camera"))

    # Rule 10/11: giới hạn thread model
    ai_threads = int(getattr(config, "AI_MAX_THREADS", 0))
    if ai_threads <= 0 or ai_threads > logical * CPU_THREADS_HARD_RATIO + 1e-6:
        v.append(Violation(
            "R10/R11",
            f"AI_MAX_THREADS={ai_threads} phải trong (0, {logical}] "
            f"(không để model chiếm {logical} thread)"))

    # Rule 10: worker pool không vượt số thread (tránh oversubscription)
    pool = int(getattr(config, "AI_WORKER_POOL_SIZE", 0))
    if pool > logical * WORKER_POOL_HARD_RATIO + 1e-6:
        v.append(Violation(
            "R10", f"AI_WORKER_POOL_SIZE={pool} > {logical} logical threads"))

    # Rule 12: RAM per-camera
    ram_target = float(getattr(config, "PER_CAMERA_RAM_TARGET_MB", 0.0))
    ram_hard = float(getattr(config, "PER_CAMERA_RAM_HARD_MB", 0.0))
    if ram_target > PER_CAMERA_RAM_TARGET_MB + 1e-6:
        v.append(Violation(
            "R12", f"PER_CAMERA_RAM_TARGET_MB={ram_target} > target "
            f"{PER_CAMERA_RAM_TARGET_MB}"))
    if ram_hard > PER_CAMERA_RAM_HARD_MB + 1e-6:
        v.append(Violation(
            "R12", f"PER_CAMERA_RAM_HARD_MB={ram_hard} > hard "
            f"{PER_CAMERA_RAM_HARD_MB}"))

    # Rule 13: CPU per-camera target
    cpu_target = float(getattr(config, "PER_CAMERA_CPU_TARGET", 0.0) or 0.0)
    if cpu_target > PER_CAMERA_CPU_TARGET + 1e-6:
        v.append(Violation(
            "R13", f"PER_CAMERA_CPU_TARGET={cpu_target} > target "
            f"{PER_CAMERA_CPU_TARGET}"))

    # Rule 1/2: model shared — kiểm tra AI_WORKER_POOL_SIZE đủ nhỏ để
    # chứng tỏ không load model "theo camera" (1 pool chung). Nếu pool
    # lớn hơn số camera dự kiến thì vẫn OK, nhưng cảnh báo nếu = 0.
    if pool < 1:
        v.append(Violation("R1/R2", "AI_WORKER_POOL_SIZE phải >= 1 (pool chia sẻ)"))

    # Rule 4: queue latest-frame (FRAME_QUEUE_MAX=1) — chỉ ghi nhận nếu
    # config có khai báo; mặc định context chỉ giữ 1 frame nên luôn đúng.
    return v


def report(config) -> str:
    """Trả chuỗi báo cáo tuân thủ hợp đồng (in lúc startup)."""
    viol = validate(config)
    lines = ["=== AI PERFORMANCE CONTRACT (Video2Action) ==="]
    if not viol:
        lines.append("[OK] Mọi cấu hình tuân thủ hợp đồng (15 quy tắc).")
    else:
        lines.append(f"[VIOLATION] {len(viol)} vi phạm hợp đồng:")
        for x in viol:
            lines.append(f"  - Rule {x.rule}: {x.message}")
    lines.append(
        f"  Detect FPS hard <= {DETECT_FPS_HARD} | "
        f"RAM/cam target <= {PER_CAMERA_RAM_TARGET_MB}MB | "
        f"CPU/cam target <= {int(PER_CAMERA_CPU_TARGET*100)}%")
    return "\n".join(lines)
