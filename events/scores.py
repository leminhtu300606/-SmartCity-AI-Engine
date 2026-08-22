"""events/scores.py — Rule 6: Công thức chấm điểm cho từng hành vi.

Mỗi sự kiện = Detection + Geometry + Motion + Temporal + Context,
với trọng số/ngưỡng khai báo trong config. Mỗi component là giá trị [0,1].

FallScore      = 0.25*posture      + 0.25*vertical_motion + 0.20*bbox_aspect_change
               + 0.15*center_velocity + 0.15*temporal
FightScore     = 0.20*person_pair  + 0.20*contact + 0.25*relative_motion
               + 0.20*motion_intensity + 0.15*temporal
CollisionScore = 0.15*track_stability + 0.20*relative_velocity + 0.20*distance_closing
               + 0.20*collision_geometry + 0.15*velocity_change + 0.10*temporal
FireScore      = 0.35*fire_model + 0.20*spatial_consistency + 0.20*temporal_persistence
               + 0.15*fire_motion + 0.10*smoke_corroboration
SmokeScore     = 0.35*smoke_model + 0.25*temporal + 0.20*spatial_expansion + 0.20*shape
"""
from typing import Any

import numpy as np

import config
import geometry
from tracker.object_state import TrackedObjectState


def weighted(total_components: dict[str, float], weights: dict[str, float]) -> float:
    """Tổng trọng số: total = sum(c * w). components là dict name->[0,1]."""
    total = 0.0
    for name, w in weights.items():
        total += float(total_components.get(name, 0.0)) * w
    return min(1.0, max(0.0, total))


def _clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return min(hi, max(lo, float(x)))


# ----------------------------------------------------------------
# A. FALL
# ----------------------------------------------------------------
def fall_score(obj: TrackedObjectState) -> tuple[dict[str, float], float]:
    """FallScore từ TrackedObjectState — các component trong [0,1]."""
    w = {
        "posture": config.FALL_W_POSTURE,
        "vertical_motion": config.FALL_W_VERTICAL_MOTION,
        "bbox_aspect_change": config.FALL_W_ASPECT_CHANGE,
        "center_velocity": config.FALL_W_CENTER_VELOCITY,
        "temporal": config.FALL_W_TEMPORAL,
    }
    comps = {
        "posture": _fall_posture(obj),
        "vertical_motion": _fall_vertical_motion(obj),
        "bbox_aspect_change": _fall_aspect_change(obj),
        "center_velocity": _fall_center_velocity(obj),
        "temporal": _fall_temporal(obj),
    }
    return comps, weighted(comps, w)


def _fall_posture(obj: TrackedObjectState) -> float:
    """Tư thế nằm ngang: aspect (w/h) càng lớn càng nằm ngang. >= thresh -> 1."""
    box = obj.bbox_history[-1]
    w = max(box[2] - box[0], 0.0)
    h = max(box[3] - box[1], 1e-5)
    aspect = w / h
    thresh = config.FALL_ASPECT_RATIO_THRESH
    # plateau từ 1.0 -> thresh+0.4
    return _clip((aspect - (thresh - 0.45)) / 0.85)


def _fall_vertical_motion(obj: TrackedObjectState) -> float:
    """Vận tốc theo trục dọc (rơi/ngã xuống). Tính trên cửa sổ ngắn."""
    if len(obj.center_history) < 3 or not obj.bbox_history:
        return 0.0
    centers = list(obj.center_history)[-4:]
    box_h = max(obj.bbox_history[-1][3] - obj.bbox_history[-1][1], 1e-5)
    dt = 1e-5
    if len(obj.time_history) >= 2:
        dt = max(obj.time_history[-1] - obj.time_history[-2], 1e-5)
    vy = (centers[-1][1] - centers[0][1]) / max(dt * (len(centers) - 1), 1e-5)
    # ngã = tâm dồn xuống nhanh: vy / (2*height per second)
    return _clip(vy / (2.0 * box_h))


def _fall_aspect_change(obj: TrackedObjectState) -> float:
    """Mức thay đổi aspect ratio so với baseline (đột ngột chuyển đứng->ngang)."""
    if len(obj.bbox_history) < 5:
        return 0.0
    window = min(10, len(obj.bbox_history))
    aspects = []
    for b in list(obj.bbox_history)[-window:]:
        w = max(b[2] - b[0], 1e-5)
        h = max(b[3] - b[1], 1e-5)
        aspects.append(w / h)
    baseline = float(np.median(aspects[: max(1, len(aspects) // 2)]))
    cur = aspects[-1]
    if baseline < 1e-5:
        return 0.0
    return _clip(abs(cur - baseline) / max(baseline, 1e-5) / 0.6)


def _fall_center_velocity(obj: TrackedObjectState) -> float:
    """Tốc độ dịch chuyển tâm (chuẩn hoá theo chiều cao)."""
    if not obj.velocity_history or not obj.bbox_history:
        return 0.0
    box_h = max(obj.bbox_history[-1][3] - obj.bbox_history[-1][1], 1e-5)
    speed = float(np.linalg.norm(obj.velocity_history[-1]))
    return _clip(speed / (3.0 * box_h))


def _fall_temporal(obj: TrackedObjectState) -> float:
    """FallScore phải duy trì >= 3/5 frames. Dùng persist counter."""
    return _clip(getattr(obj, "fall_persist_count", 0) / max(config.FALL_CONFIRM_FRAMES, 1))


# ----------------------------------------------------------------
# B. FIGHT
# ----------------------------------------------------------------
def fight_score(objA: TrackedObjectState, objB: TrackedObjectState,
                agitation: float, sustained_count: int) -> tuple[dict[str, float], float]:
    """FightScore giữa 2 người.

    Quan trọng (Rule 6B): khoảng cách GẦN không được là điều kiện đủ — 2 người
    đứng/sát/nói chuyện phải ra điểm rất thấp. Contact + motion mạnh mới cao.
    """
    w = {
        "person_pair": config.FIGHT_W_PERSON_PAIR,
        "contact": config.FIGHT_W_CONTACT,
        "relative_motion": config.FIGHT_W_RELATIVE_MOTION,
        "motion_intensity": config.FIGHT_W_MOTION_INTENSITY,
        "temporal": config.FIGHT_W_TEMPORAL,
    }
    boxA, boxB = objA.bbox_history[-1], objB.bbox_history[-1]
    avg_h = max(((boxA[3] - boxA[1]) + (boxB[3] - boxB[1])) / 2.0, 1e-5)

    centA = np.array(objA.center_history[-1])
    centB = np.array(objB.center_history[-1])
    dist = float(np.linalg.norm(centA - centB))
    rel_dist = dist / avg_h

    # person_pair: gần nhưng bão hoà ở 1.0 khi rel_dist <= 1.0
    pair = _clip(1.0 - max(0.0, (rel_dist - 0.7)) / (config.CONFLICT_DIST_HARD_CAP + 0.3))

    # contact: mức chồng lấn bbox thực sự (chạm người)
    iou = geometry.iou(boxA, boxB)
    contact = _clip(iou / 0.3)

    # relative_motion: tốc độ tiến/giật tương hỗ (chuẩn hoá theo chiều cao)
    rel_motion = 0.0
    if objA.velocity_history and objB.velocity_history:
        va = np.array(objA.velocity_history[-1])
        vb = np.array(objB.velocity_history[-1])
        rel = float(np.linalg.norm(va - vb))
        rel_motion = _clip(rel / (3.0 * avg_h) / 1.5)

    # motion_intensity: agitation (jitter/body speed) — tín hiệu giật cục
    motion_intensity = _clip(agitation / (config.CONFLICT_AGITATION_THRESH * 1.5))

    temporal = _clip(sustained_count / max(config.FIGHT_CONFIRM_FRAMES, 1))

    comps = {
        "person_pair": pair,
        "contact": contact,
        "relative_motion": rel_motion,
        "motion_intensity": motion_intensity,
        "temporal": temporal,
    }
    return comps, weighted(comps, w)


# ----------------------------------------------------------------
# C. VEHICLE COLLISION
# ----------------------------------------------------------------
def collision_score(objA: TrackedObjectState, objB: TrackedObjectState,
                    closing_speed: float, decel: float, sustained_count: int,
                    track_frames: int = 4) -> tuple[dict[str, float], float]:
    """CollisionScore. Chỉ gọi khi cả 2 track ổn định + conf xe đủ cao."""
    w = {
        "track_stability": config.COLLISION_W_TRACK_STABILITY,
        "relative_velocity": config.COLLISION_W_RELATIVE_VELOCITY,
        "distance_closing": config.COLLISION_W_DISTANCE_CLOSING,
        "collision_geometry": config.COLLISION_W_GEOMETRY,
        "velocity_change": config.COLLISION_W_VELOCITY_CHANGE,
        "temporal": config.COLLISION_W_TEMPORAL,
    }
    boxA, boxB = objA.bbox_history[-1], objB.bbox_history[-1]

    # track_stability: cả 2 xe có lịch sử ổn định (track ID duy trì)
    stab = _clip(min(len(objA.bbox_history), len(objB.bbox_history)) / max(track_frames, 1))

    # relative_velocity: closing speed (đang lao vào nhau)
    rel_vel = _clip(closing_speed / config.VEHICLE_CLOSING_SPEED_THRESH)

    # distance_closing: khoảng cách đang giảm dần
    dist_closing = _falling_distance_rate(objA, objB)

    # collision_geometry: ov
    iou = geometry.iou(boxA, boxB)
    cntA = np.array(objA.center_history[-1])
    cntB = np.array(objB.center_history[-1])
    diagA = np.sqrt((boxA[2] - boxA[0]) ** 2 + (boxA[3] - boxA[1]) ** 2)
    diagB = np.sqrt((boxB[2] - boxB[0]) ** 2 + (boxB[3] - boxB[1]) ** 2)
    avg_diag = max((diagA + diagB) / 2.0, 1e-5)
    rel_dist = float(np.linalg.norm(cntA - cntB)) / avg_diag
    geom = max(_clip(iou / config.VEHICLE_IOU_THRESH),
               _clip((config.VEHICLE_PROXIMITY_DIST_RATIO - rel_dist) / config.VEHICLE_PROXIMITY_DIST_RATIO + 0.3))
    geom = _clip(geom)

    # velocity_change: giảm tốc đột ngột (phanh gấp)
    vel_change = _clip(decel / config.VEHICLE_DECEL_THRESH)

    temporal = _clip(sustained_count / max(config.COLLISION_CONFIRM_FRAMES, 1))

    comps = {
        "track_stability": stab,
        "relative_velocity": rel_vel,
        "distance_closing": dist_closing,
        "collision_geometry": geom,
        "velocity_change": vel_change,
        "temporal": temporal,
    }
    return comps, weighted(comps, w)


def _falling_distance_rate(objA: TrackedObjectState, objB: TrackedObjectState) -> float:
    """Tốc độ khoảng cách 2 tâm đang giảm (closing rate của distance)."""
    if len(objA.center_history) < 3 or len(objB.center_history) < 3:
        return 0.0
    cA = list(objA.center_history)
    cB = list(objB.center_history)
    n = min(3, len(cA), len(cB))
    d_prev = float(np.linalg.norm(np.array(cA[-n]) - np.array(cB[-n])))
    d_cur = float(np.linalg.norm(np.array(cA[-1]) - np.array(cB[-1])))
    if d_prev < 1e-5:
        return 1.0
    return _clip((d_prev - d_cur) / d_prev)


# ----------------------------------------------------------------
# D. FIRE
# ----------------------------------------------------------------
def fire_score(fire_signal: float, hue_spread: float, flicker: float,
               persist_hits: int, persist_window: int,
               smoke_score: float) -> tuple[dict[str, float], float]:
    """FireScore từ tín hiệu HSV/motion của smoke_fire_rules."""
    w = {
        "fire_model": config.FIRE_W_MODEL,
        "spatial_consistency": config.FIRE_W_SPATIAL,
        "temporal_persistence": config.FIRE_W_TEMPORAL,
        "fire_motion": config.FIRE_W_MOTION,
        "smoke_corroboration": config.FIRE_W_SMOKE,
    }
    model = _clip(fire_signal)
    spatial = _clip(hue_spread * 12.0)          # đa sắc = lửa thật
    spatial = max(spatial, _clip(min(1.0, fire_signal * 4.0) - 0.5))
    temporal = _clip(persist_hits / max(config.FIRE_CONFIRM_FRAMES, 1))
    motion = _clip(flicker / 0.25)              # nhấp nháy
    comps = {
        "fire_model": model,
        "spatial_consistency": spatial,
        "temporal_persistence": temporal,
        "fire_motion": motion,
        "smoke_corroboration": _clip(smoke_score),
    }
    return comps, weighted(comps, w)


# ----------------------------------------------------------------
# E. SMOKE
# ----------------------------------------------------------------
def smoke_score(smoke_signal: float, spatial_expansion: float,
                 shape_consistency: float, persist_hits: int,
                 persist_window: int, required_frames: int | None = None,
                 ) -> tuple[dict[str, float], float]:
    """SmokeScore: khói phải phát triển/lan/biến dạng, không phải vùng xám tĩnh."""
    w = {
        "smoke_model": config.SMOKE_W_MODEL,
        "temporal": config.SMOKE_W_TEMPORAL,
        "spatial_expansion": config.SMOKE_W_EXPANSION,
        "shape": config.SMOKE_W_SHAPE,
    }
    comps = {
        "smoke_model": _clip(smoke_signal),
        "temporal": _clip(persist_hits / max(config.SMOKE_CONFIRM_FRAMES, 1)),
        "spatial_expansion": _clip(spatial_expansion),
        "shape": _clip(shape_consistency),
    }
    return comps, weighted(comps, w)