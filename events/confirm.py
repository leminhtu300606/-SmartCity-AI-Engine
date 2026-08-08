class EventConfirmTracker:
    """Event Score / Confirm trước khi bắn Alert chính thức."""
    def __init__(self, confirm_frames=3):
        self.confirm_frames = confirm_frames
        self.pending_events = {}  # key -> score_count

    def process(self, candidate_events):
        confirmed_events = []
        current_keys = set()

        for ev in candidate_events:
            key = (ev["event_type"], tuple(ev.get("track_ids", [])), ev.get("zone_name"))
            current_keys.add(key)
            self.pending_events[key] = self.pending_events.get(key, 0) + 1

            if self.pending_events[key] >= self.confirm_frames:
                confirmed_events.append(ev)

        # Decay/Clean missing events
        for key in list(self.pending_events.keys()):
            if key not in current_keys:
                self.pending_events[key] -= 1
                if self.pending_events[key] <= 0:
                    del self.pending_events[key]

        return confirmed_events