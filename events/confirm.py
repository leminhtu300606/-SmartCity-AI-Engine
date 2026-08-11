import config


class EventConfirmTracker:
    """Event Score / Confirm trước khi bắn Alert chính thức.

    Cải tiến: mỗi loại event có ngưỡng confirm riêng (per-event-type thresholds),
    nhưng MỌI event phải xuất hiện ít nhất MIN_CONFIRM_FRAMES (>= 2) detection
    frame trước khi được xác nhận — tránh cảnh báo chỉ từ 1 frame duy nhất.
    Smoke/Fire, VEHICLE_COLLISION có persistence riêng bên trong rule; ngưỡng
    tại đây là lớp confirm chung tối thiểu.
    """

    def __init__(self, default_frames=4):
        self.default_frames = default_frames
        self.pending_events = {}  # key -> score_count

    def process(self, candidate_events, decay=True):
        """Xử lý candidates.

        Args:
            candidate_events: Danh sách event candidate frame này.
            decay: True → giảm counter cho các key không còn xuất hiện.
                Object-based candidates CHỈ xuất hiện trên detection frame, nên decay
                phải chạy trên detection frame để counter tích lũy đúng nhịp 1:1.
                Trên predicted frame (decay=False) chỉ cộng dồn, không trừ.
        """
        confirmed_events = []
        current_keys = set()

        for ev in candidate_events:
            key = (ev["event_type"],
                   tuple(ev.get("track_ids", [])),
                   ev.get("zone_name"))
            current_keys.add(key)

            # Lấy ngưỡng confirm riêng cho loại event này, nhưng KHÔNG thấp
            # hơn MIN_CONFIRM_FRAMES (2) — mọi hiện tượng cần >= 2 frame.
            required = max(
                config.MIN_CONFIRM_FRAMES,
                config.EVENT_CONFIRM_MAP.get(ev["event_type"], self.default_frames),
            )

            # Cap counter tại required + 2 để khi tín hiệu hết,
            # alert được thu hồi NHANH (chỉ cần decay 2 frame)
            self.pending_events[key] = min(
                required + 2,
                self.pending_events.get(key, 0) + 1,
            )

            if self.pending_events[key] >= required:
                confirmed_events.append(ev)

        # Decay / Clean events không còn xuất hiện (chỉ trên detection frame)
        if decay:
            for key in list(self.pending_events.keys()):
                if key not in current_keys:
                    self.pending_events[key] -= 1
                    if self.pending_events[key] <= 0:
                        del self.pending_events[key]

        return confirmed_events