import cv2
import numpy as np
import config


class IntrusionRules:
    """Spatial Rules ưu tiên vùng không gian (ROI), không dùng heuristic hành vi
    (đi chậm + quay ngó = kẻ lạ):

    Person detection
        -> Restricted / Security ROI
        -> Person enters ROI (depth check: phải vào sâu, không chỉ chạm rìa)
        -> Movement + Dwell time
        -> Intrusion candidate

    Nếu sau này có identity / whitelist thì bổ sung vào bước cuối.
    """

    def check_intrusion(self, obj, roi_zones):
        events = []

        # 1. Person detection (chỉ xét Class 0)
        if obj.cls_id != 0:
            return events

        if len(obj.bbox_history) == 0 or len(obj.center_history) == 0:
            return events

        curr_center = obj.center_history[-1]
        curr_bbox = obj.bbox_history[-1]
        person_h = max(curr_bbox[3] - curr_bbox[1], 1e-5)

        for zone in roi_zones:
            # 2. Restricted / Security ROI
            if zone["event_type"] != config.EVENT_TYPE_INTRUSION:
                continue

            zone_name = zone["name"]
            polygon = np.array(zone["polygon"], np.int32)

            # 3. Person enters ROI — Depth Check
            # pointPolygonTest với measureDist=True trả về khoảng cách có dấu:
            #   dương = bên trong, âm = bên ngoài, 0 = trên cạnh
            # Yêu cầu người phải vào SÂU bên trong polygon (không chỉ chạm rìa)
            # → giảm false positive khi người đi ngang qua biên polygon
            depth = cv2.pointPolygonTest(
                polygon,
                (float(curr_center[0]), float(curr_center[1])),
                True,  # measureDist = True
            )

            min_depth = person_h * config.INTRUSION_DEPTH_RATIO
            is_inside = depth >= min_depth

            obj.update_zone_state(zone_name, is_inside, center=curr_center)
            obj.tick_dwell(zone_name, is_inside)

            if not is_inside:
                continue

            # 4. Movement + Dwell time trong vùng
            # Movement = displacement không gian kể từ lúc bước vào vùng (spatial),
            # KHÔNG dùng ngưỡng tốc độ hành vi (tránh "đi chậm = kẻ lạ").
            entry_pos = obj.zone_entry.get(zone_name)
            displacement = (
                np.linalg.norm(np.array(curr_center) - entry_pos)
                if entry_pos is not None
                else 0.0
            )
            if displacement < config.INTRUSION_MIN_MOVEMENT_PX:
                continue
            if obj.dwell_times[zone_name] < config.INTRUSION_DWELL_FRAMES:
                continue

            # 5. Intrusion candidate
            # Confidence tăng theo thời gian lưu lại + displacement vào sâu
            dwell_bonus = min(0.1, obj.dwell_times[zone_name] / 200.0)
            depth_bonus = min(0.05, displacement / 200.0)

            events.append({
                "event_type": "RESTRICTED_INTRUSION",
                "track_ids": [obj.track_id],
                "zone_name": zone_name,
                "confidence": min(0.98, 0.85 + dwell_bonus + depth_bonus),
                "description": f"Phát hiện xâm nhập khu vực cấm {zone_name}",
            })

        return events
