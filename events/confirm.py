import config


class EventConfirmTracker:
    """Event Score / Confirm trước khi bắn Alert chính thức (Rule 7).

    Nomenclature thống nhất:
      _DETECTED   — tín hiệu thô (rule sinh ra, chưa qua candidate). KHÔNG alert.
      _CANDIDATE  — confidence >= candidate_thresh. KHÔNG hiển thị alert.
      _CONFIRMED  — duy trì >= frames frames & confidence >= confirm_thresh.
                    CHỈ CONFIRMED mới: snapshot, lưu DB, gửi notification,
                    đưa chart, phát cảnh báo.

    MỌI event phải có candidate -> confirmation (Rule 14).
    """

    def __init__(self):
        self.pending_events = {}  # key -> {"count": int, "score": float}

    def process(self, candidate_events, decay=True):
        """Xử lý candidate events → trả list CONFIRMED (đã gắn stage).

        Args:
            candidate_events: Danh sách event CANDIDATE (confidence = score).
            decay: True → giảm counter cho các key không còn xuất hiện.
        """
        confirmed_events = []
        current_keys = set()

        for ev in candidate_events:
            event_type = ev["event_type"]
            meta = config.rule_meta(event_type)
            key = (event_type,
                   tuple(ev.get("track_ids", [])),
                   ev.get("zone_name"))
            score = ev.get("confidence", 0.0)
            current_keys.add(key)

            # 1. Candidate gate: score >= candidate_thresh mới bắt đầu đếm
            if score < meta["candidate"]:
                continue

            state = self.pending_events.setdefault(
                key, {"count": 0, "score": 0.0})
            state["count"] = min(meta["frames"] + 2, state["count"] + 1)
            state["score"] = max(state["score"], score)

            # 2. Confirmed: đủ frames + score >= confirm_thresh
            if (state["count"] >= meta["frames"]
                    and score >= meta["confirm"]):
                confirmed = dict(ev)
                confirmed["stage"] = config.STAGE_CONFIRMED
                confirmed["confidence"] = round(min(1.0, score), 3)
                confirmed_events.append(confirmed)

        # Decay các key không còn xuất hiện (trên detection frame)
        if decay:
            for key in list(self.pending_events.keys()):
                if key not in current_keys:
                    self.pending_events[key]["count"] -= 1
                    if self.pending_events[key]["count"] <= 0:
                        del self.pending_events[key]

        return confirmed_events