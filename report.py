import html
import threading
import time


class ReportManager:
    """Thu thập dữ liệu vận hành và sinh báo cáo HTML dashboard.

    Dữ liệu được gom từ các CameraWorker threads (an toàn bằng lock),
    dashboard tự refresh trong trình duyệt (meta refresh) và đọc lại file mỗi lần.
    """

    def __init__(self, report_path="report.html"):
        self.lock = threading.Lock()
        self.report_path = report_path
        self.start_time = time.time()
        self.cameras = {}   # cam_id -> state
        self.events = []    # list event dicts

    # ---------------- Record ----------------
    def register_camera(self, camera_id):
        with self.lock:
            self.cameras.setdefault(camera_id, {
                "online": False,
                "fps": 0.0,
                "frame_idx": 0,
                "counts": {},
                "last_seen": 0,
            })

    def update_camera(self, camera_id, online=True, fps=0.0, frame_idx=0, counts=None):
        with self.lock:
            cam = self.cameras.setdefault(camera_id, {
                "online": False,
                "fps": 0.0,
                "frame_idx": 0,
                "counts": {},
                "last_seen": 0,
            })
            cam["online"] = online
            cam["frame_idx"] = frame_idx
            if online:
                cam["fps"] = fps
                cam["counts"] = counts or {}
                cam["last_seen"] = time.time()
            else:
                cam["fps"] = 0.0

    def record_events(self, camera_id, events):
        now_str = time.strftime("%H:%M:%S")
        with self.lock:
            for ev in events:
                self.events.append({
                    "time": now_str,
                    "camera_id": camera_id,
                    "event_type": ev.get("event_type", "?"),
                    "description": ev.get("description", ""),
                    "confidence": float(ev.get("confidence", 0.0)),
                    "track_ids": [str(x) for x in ev.get("track_ids", [])],
                    "zone_name": ev.get("zone_name", "-"),
                })

    # ---------------- Summary ----------------
    def summary(self):
        with self.lock:
            uptime_s = int(time.time() - self.start_time)
            online = [c for c in self.cameras.values() if c["online"]]
            frames = sum(c["frame_idx"] for c in self.cameras.values())
            counts = {}
            for c in self.cameras.values():
                for k, v in c["counts"].items():
                    counts[k] = counts.get(k, 0) + v
            by_type = {}
            for ev in self.events:
                by_type[ev["event_type"]] = by_type.get(ev["event_type"], 0) + 1
            return {
                "uptime": uptime_s,
                "total_alerts": len(self.events),
                "online_cams": len(online),
                "total_cams": len(self.cameras),
                "frames": frames,
                "counts": counts,
                "by_type": by_type,
            }

    # ---------------- Render ----------------
    def render_html(self):
        s = self.summary()
        uptime = f"{s['uptime'] // 3600:02d}:{(s['uptime'] % 3600) // 60:02d}:{s['uptime'] % 60:02d}"

        cls_names = {0: "person", 2: "car", 3: "motorbike", 5: "bus", 7: "truck"}
        obj_cells = "".join(
            f'<span class="obj-chip">{cls_names.get(int(k), k)}: {v}</span>'
            for k, v in sorted(s["counts"].items(), key=lambda x: int(x[0]))
        )

        cam_cards = []
        for cam_id in sorted(self.cameras.keys()):
            cam = self.cameras[cam_id]
            status_cls = "ok" if cam["online"] else "off"
            status_txt = "Online" if cam["online"] else "Offline"
            counts_txt = " ".join(
                f"{cls_names.get(int(k), k)} {v}" for k, v in sorted(cam["counts"].items(), key=lambda x: int(x[0]))
            ) or "-"
            cam_cards.append(
                f'<div class="cam-card"><div class="cam-head"><strong>{html.escape(cam_id)}</strong>'
                f'<span class="badge {status_cls}">{status_txt}</span></div>'
                f'<div class="cam-meta">FPS: <b>{cam["fps"]:.1f}</b> &nbsp; Frame: <b>{cam["frame_idx"]}</b></div>'
                f'<div class="cam-counts">{counts_txt}</div></div>'
            )
        cams_html = "".join(cam_cards) if cam_cards else '<p class="empty">Chưa có dữ liệu camera</p>'

        by_type_rows = "".join(
            f"<tr><td>{html.escape(k)}</td><td class='num'>{v}</td></tr>"
            for k, v in sorted(s["by_type"].items(), key=lambda x: -x[1])
        )
        stats_html = by_type_rows or '<tr><td colspan="2" class="empty">Chưa có sự kiện</td></tr>'

        detail_rows = "".join(
            f"<tr><td>{html.escape(ev['time'])}</td><td>{html.escape(ev['camera_id'])}</td>"
            f"<td><span class='ev-badge'>{html.escape(ev['event_type'])}</span></td>"
            f"<td>{html.escape(ev['zone_name'])}</td><td>{ev['track_ids'] or '-'}</td>"
            f"<td class='num'>{ev['confidence']:.2f}</td>"
            f"<td>{html.escape(ev['description'])}</td></tr>"
            for ev in reversed(self.events[-200:])
        )
        detail_html = detail_rows or '<tr><td colspan="7" class="empty">Chưa có sự kiện nào</td></tr>'

        updated = time.strftime("%d/%m/%Y %H:%M:%S")

        body = f"""
<header>
  <h1>SMARTVISION AI &mdash; B&Aacute;O C&Aacute;O VẬN H&Agrave;NH</h1>
  <div class="sub">Cập nhật: {updated} &nbsp;|&nbsp; Thời gian chạy: {uptime}</div>
</header>

<section class="kpis">
  <div class="kpi"><div class="kpi-val">{s['total_alerts']}</div><div class="kpi-lbl">Tổng sự kiện</div></div>
  <div class="kpi"><div class="kpi-val">{s['online_cams']}/{s['total_cams']}</div><div class="kpi-lbl">Camera Online</div></div>
  <div class="kpi"><div class="kpi-val">{s['frames']}</div><div class="kpi-lbl">Frames xử lý</div></div>
  <div class="kpi"><div class="kpi-val obj-kpi">{obj_cells or '-'}</div><div class="kpi-lbl">Object phát hiện</div></div>
</section>

<section class="cameras"><h2>TRẠNG THÁI CAMERA</h2><div class="cam-grid">{cams_html}</div></section>

<div class="cols">
  <section><h2>THỐNG KÊ THEO LOẠI SỰ KIỆN</h2>
    <table><thead><tr><th>Loại sự kiện</th><th>Số lần</th></tr></thead><tbody>{stats_html}</tbody></table>
  </section>
  <section><h2>THỐNG KÊ CHI TIẾT (200 gần nhất)</h2>
    <table><thead><tr><th>Thời gian</th><th>Camera</th><th>Sự kiện</th><th>Zone</th><th>Track ID</th><th>Conf</th><th>Mô tả</th></tr></thead><tbody>{detail_html}</tbody></table>
  </section>
</div>
"""

        return ("<!DOCTYPE html><html lang='vi'><head><meta charset='utf-8'>"
                "<meta name='viewport' content='width=device-width, initial-scale=1'>"
                "<meta http-equiv='refresh' content='5'>"
                "<title>SmartVision AI &mdash; Báo cáo vận hành</title><style>"
                + _CSS + "</style></head><body>" + body + "</body></html>")

    def save(self):
        try:
            with open(self.report_path, "w", encoding="utf-8") as f:
                f.write(self.render_html())
        except Exception as e:
            print(f"[REPORT] save error: {e}")


_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { background: #0f1420; color: #e6edf3; font-family: 'Segoe UI', Arial, sans-serif; padding: 20px; }
header { text-align: center; margin-bottom: 20px; }
h1 { font-size: 22px; letter-spacing: 2px; color: #58a6ff; }
.sub { color: #8b949e; font-size: 13px; margin-top: 4px; }
h2 { font-size: 14px; color: #79c0ff; letter-spacing: 1px; margin-bottom: 10px; border-bottom: 1px solid #21262d; padding-bottom: 6px; }
section { background: #161b26; border: 1px solid #21262d; border-radius: 10px; padding: 16px; }
.kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; margin-bottom: 16px; }
.kpi { background: #161b26; border: 1px solid #21262d; border-radius: 10px; padding: 14px; text-align: center; }
.kpi-val { font-size: 26px; font-weight: 700; color: #58a6ff; }
.kpi-lbl { font-size: 12px; color: #8b949e; margin-top: 4px; }
.obj-kpi { font-size: 15px; display: flex; flex-wrap: wrap; gap: 6px; justify-content: center; }
.obj-chip { background: #1f2633; border: 1px solid #2d3646; padding: 3px 8px; border-radius: 12px; font-weight: 400; }
.cameras { margin-bottom: 16px; }
.cam-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; }
.cam-card { background: #121826; border: 1px solid #21262d; border-radius: 8px; padding: 12px; }
.cam-head { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }
.cam-meta { font-size: 13px; color: #8b949e; margin-bottom: 8px; }
.cam-meta b { color: #e6edf3; }
.cam-counts { font-size: 12px; color: #9da7b3; }
.badge { font-size: 11px; font-weight: 700; padding: 2px 8px; border-radius: 10px; }
.badge.ok { background: #0e3a24; color: #3fb950; }
.badge.off { background: #3a1d1d; color: #f85149; }
.cols { display: grid; grid-template-columns: 1fr 2fr; gap: 16px; }
@media (max-width: 900px) { .cols { grid-template-columns: 1fr; } }
table { width: 100%; border-collapse: collapse; font-size: 12px; }
th, td { text-align: left; padding: 6px 8px; border-bottom: 1px solid #21262d; }
th { color: #8b949e; text-transform: uppercase; font-size: 11px; letter-spacing: 0.5px; }
td.num { text-align: right; }
.ev-badge { background: #1f2633; color: #ffd33d; border: 1px solid #3d3a1d; padding: 1px 6px; border-radius: 8px; font-size: 11px; }
.empty { color: #8b949e; font-style: italic; }
"""
