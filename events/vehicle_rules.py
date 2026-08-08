import numpy as np
import config

class VehicleAccidentRules:
    """Rules cho tai nạn giao thông giữa các phương tiện (car, motorbike, bus, truck)."""
    VEHICLE_CLASS_IDS = [2, 3, 5, 7]  # COCO: Car, Motorbike, Bus, Truck

    def __init__(self, dt):
        self.dt = dt

    def check_collision(self, objA, objB):
        """BBox overlap + giảm tốc đột ngột trên cửa sổ thời gian. Trả về True/False."""
        if len(objA.velocity_history) < 5 or len(objB.velocity_history) < 5:
            return False

        boxA, boxB = objA.bbox_history[-1], objB.bbox_history[-1]
        inter_w = max(0, min(boxA[2], boxB[2]) - max(boxA[0], boxB[0]))
        inter_h = max(0, min(boxA[3], boxB[3]) - max(boxA[1], boxB[1]))
        if inter_w * inter_h <= 0:
            return False

        # Kiểm tra sự giảm tốc đột ngột (Sudden Deceleration)
        vA = [np.linalg.norm(v) for v in objA.velocity_history]
        accA = np.diff(vA) / self.dt
        has_sudden_stop = np.min(accA) < config.ACCIDENT_DECEL_THRESH if len(accA) > 0 else False

        return has_sudden_stop