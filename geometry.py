"""geometry.py — Helper hình học dùng chung (DRY).

Loại bỏ các bản sao `_iou` / `area` / `contained_ratio` từng định nghĩa
riêng trong detector.py, tracker/memory_manager.py, events/scores.py.
Mọi hàm nhận bbox dạng [x1, y1, x2, y2] (list/tuple/ndarray) và trả float.
"""
import numpy as np
from typing import Sequence, Union

# bbox: [x1, y1, x2, y2] — chấp nhận list / tuple / numpy array (chỉ dùng chỉ số).
BBox = Union[Sequence[float], "Sequence[int]"]


def box_area(box: BBox) -> float:
    """Diện tích bbox (luôn >= 0)."""
    return max(float(box[2]) - float(box[0]), 0.0) * max(float(box[3]) - float(box[1]), 0.0)


def iou(box_a: BBox, box_b: BBox) -> float:
    """IoU của 2 bbox. Trả 0 nếu không giao nhau."""
    x_a = max(float(box_a[0]), float(box_b[0]))
    y_a = max(float(box_a[1]), float(box_b[1]))
    x_b = min(float(box_a[2]), float(box_b[2]))
    y_b = min(float(box_a[3]), float(box_b[3]))
    inter = max(0.0, x_b - x_a) * max(0.0, y_b - y_a)
    area_a = box_area(box_a)
    area_b = box_area(box_b)
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def containment_ratio(inner: BBox, outer: BBox) -> float:
    """Tỷ lệ diện tích `inner` nằm trong `outer` (so với diện tích `inner`)."""
    a = max(float(inner[0]), float(outer[0]))
    b = max(float(inner[1]), float(outer[1]))
    c = min(float(inner[2]), float(outer[2]))
    d = min(float(inner[3]), float(outer[3]))
    inter = max(0.0, c - a) * max(0.0, d - b)
    inner_area = box_area(inner)
    if inner_area <= 0.0:
        return 0.0
    return inter / inner_area


def bbox_center(box: BBox):
    """Tâm bbox trả về dạng numpy array [cx, cy]."""
    return np.array(
        [(float(box[0]) + float(box[2])) / 2.0, (float(box[1]) + float(box[3])) / 2.0]
    )


def bbox_diagonal(box: BBox) -> float:
    """Đường chéo bbox (chuẩn hoá khoảng cách)."""
    w = max(float(box[2]) - float(box[0]), 0.0)
    h = max(float(box[3]) - float(box[1]), 0.0)
    return float(np.hypot(w, h))
