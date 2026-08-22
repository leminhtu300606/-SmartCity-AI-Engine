"""Dashboard Web (Flask) — hiển thị cảnh báo + ảnh lý do cảnh báo.

Endpoints:
  GET /                -> Giao diện HTML (polling /api/alerts).
  GET /api/alerts      -> JSON danh sách alert.
  GET /api/stats       -> Thống kê nhanh (số alert theo event type).
  GET /snapshots/<key> -> Ảnh (full/crop) phục vụ TRỰC TIẾP từ bộ nhớ AlertStore.
"""
import time

from flask import Flask, jsonify, Response

from dashboard.store import AlertStore

# Màu banner cho từng event type (hex, dùng trong JS)
EVENT_COLORS = {
    "HUMAN_FALL": "#ff9800",
    "HUMAN_CONFLICT": "#f44336",
    "VEHICLE_COLLISION": "#ff9800",
    "FIRE_DETECTED": "#f44336",
    "RESTRICTED_INTRUSION": "#d32f2f",
    "SMOKE_DETECTED": "#9e9e9e",
}

_INDEX_HTML = """<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SmartVision AI - Alert Dashboard</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: 'Segoe UI', Roboto, sans-serif;
    background: #0f1420; color: #e6e9f0; padding: 20px;
  }
  header {
    display: flex; align-items: center; justify-content: space-between;
    padding-bottom: 14px; border-bottom: 1px solid #232b3b; margin-bottom: 20px;
  }
  header h1 { font-size: 20px; color: #7eb6ff; letter-spacing: .5px; }
  .badge {
    background: #1c2536; border: 1px solid #2c3a55; border-radius: 999px;
    padding: 6px 14px; font-size: 12px; color: #9fb2cf;
  }
  .badge span { color: #ffd166; font-weight: 700; margin-left: 4px; }
  .filters { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 18px; }
  .filters button {
    background: #1c2536; color: #9fb2cf; border: 1px solid #2c3a55;
    border-radius: 999px; padding: 5px 12px; font-size: 12px; cursor: pointer;
  }
  .filters button.active { background: #2a6df4; color: #fff; border-color: #2a6df4; }
  .grid {
    display: grid; grid-template-columns: repeat(auto-fill, minmax(360px, 1fr));
    gap: 16px;
  }
  .card {
    background: #161d2c; border: 1px solid #232b3b; border-radius: 12px;
    overflow: hidden; transition: transform .1s;
  }
  .card:hover { transform: translateY(-2px); }
  .card .imgbox { position: relative; background: #000; }
  .card .imgbox img { width: 100%; display: block; }
  .card .cropbox { border-top: 1px solid #232b3b; background: #000; }
  .card .cropbox .croplabel {
    font-size: 10px; color: #7eb6ff; padding: 6px 10px 2px; letter-spacing: .3px;
  }
  .card .cropbox img { width: 100%; display: block; max-height: 180px; object-fit: contain; }
  .card .tag {
    position: absolute; top: 8px; left: 8px; padding: 4px 10px;
    border-radius: 6px; font-size: 11px; font-weight: 700; color: #fff;
    text-transform: uppercase; letter-spacing: .4px;
  }
  .card .info { padding: 12px 14px; }
  .card .info .desc { font-size: 13px; color: #cdd6e4; margin-bottom: 8px; }
  .card .info .evidence { font-size: 11px; color: #93a6c8; margin-bottom: 8px; line-height: 1.4; }
  .card .meta { display: flex; justify-content: space-between; font-size: 11px; color: #7e8aa0; }
  .card .meta .cam { color: #7eb6ff; }
  .card .meta .conf { color: #ffd166; }
  .empty { text-align: center; color: #5a6a78; padding: 60px 0; font-size: 14px; }
  footer { margin-top: 24px; text-align: center; font-size: 11px; color: #5a6a78; }
</style>
</head>
<body>
<header>
  <h1>&#128225; SmartVision AI &mdash; Alert Dashboard</h1>
  <div class="badge">Tổng cảnh báo: <span id="total">0</span></div>
</header>

<div class="filters">
  <button class="active" data-type="ALL">Tất cả</button>
  <button data-type="HUMAN_FALL">Người ngã</button>
  <button data-type="HUMAN_CONFLICT">Xô xát</button>
  <button data-type="VEHICLE_COLLISION">Xe-xe</button>
  <button data-type="FIRE_DETECTED">Lửa</button>
  <button data-type="SMOKE_DETECTED">Khói</button>
  <button data-type="RESTRICTED_INTRUSION">Xâm nhập</button>
</div>

<div class="grid" id="grid"></div>

<footer>Auto-refresh mỗi 2 giây &mdash; dữ liệu từ /api/alerts</footer>

<script>
const COLORS = __EVENT_COLORS__;
let currentFilter = 'ALL';

function esc(s) {
  const div = document.createElement('div');
  div.textContent = (s == null ? '' : String(s));
  return div.innerHTML;
}

function tagColor(type) { return COLORS[type] || '#2a6df4'; }

async function refresh() {
  try {
    const r = await fetch('/api/alerts');
    const data = await r.json();
    document.getElementById('total').textContent = data.total;

    let list = data.alerts;
    if (currentFilter !== 'ALL') {
      list = list.filter(a => a.event_type === currentFilter);
    }

    const grid = document.getElementById('grid');
    if (!list.length) {
      grid.innerHTML = '<div class="empty">Chưa có cảnh báo nào.</div>';
      return;
    }

    grid.innerHTML = list.map(a => `
      <div class="card">
        <div class="imgbox">
          <img src="/snapshots/${encodeURIComponent(a.snapshot)}" alt="snapshot">
          <span class="tag" style="background:${tagColor(a.event_type)}">${esc(a.event_type)}</span>
        </div>
        ${a.crop ? `
        <div class="cropbox">
          <div class="croplabel">Vị trí cảnh báo (crop)</div>
          <img src="/snapshots/${encodeURIComponent(a.crop)}" alt="crop">
        </div>` : ''}
        <div class="info">
          <div class="desc">${esc(a.description)}</div>
          ${a.evidence_objects && a.evidence_objects.length ? `
          <div class="evidence">${a.evidence_objects.map(o => `#${esc(o.track_id)} ${esc(o.cls_id)} spd:${esc(o.speed)}`).join(' | ')}</div>` : ''}
          <div class="meta">
            <span class="cam">${esc(a.camera_id)} &mdash; ${esc(a.time_str)}</span>
            <span class="conf">conf: ${a.confidence}</span>
          </div>
        </div>
      </div>
    `).join('');
  } catch (e) {
    // server chưa sẵn sàng -> bỏ qua, refresh lần sau
  }
}

document.querySelectorAll('.filters button').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.filters button').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    currentFilter = btn.dataset.type;
    refresh();
  });
});

refresh();
setInterval(refresh, 2000);
</script>
</body>
</html>
"""


def create_app(store: AlertStore, snapshot_dir: str):
    """Tạo Flask app phục vụ dashboard từ AlertStore (ảnh trong bộ nhớ)."""
    app = Flask(__name__)

    @app.get("/")
    def index():
        html = _INDEX_HTML.replace("__EVENT_COLORS__", str(EVENT_COLORS))
        return Response(html, mimetype="text/html")

    @app.get("/api/alerts")
    def api_alerts():
        alerts = store.recent(limit=100)
        return jsonify({
            "total": store.count(),
            "alerts": alerts,
        })

    @app.get("/api/stats")
    def api_stats():
        by_type = {}
        for a in store.recent(limit=200):
            by_type[a["event_type"]] = by_type.get(a["event_type"], 0) + 1
        return jsonify({
            "total": store.count(),
            "by_type": by_type,
        })

    @app.get("/snapshots/<path:filename>")
    def snapshot(filename):
        img = store.get_image(filename)
        if img is None:
            return Response("Không tìm thấy ảnh", status=404)
        return Response(img, mimetype="image/jpeg")

    return app


def run_dashboard(store: AlertStore, snapshot_dir: str, host="0.0.0.0", port=8080):
    """Chạy dashboard trong thread riêng (không block pipeline)."""
    import threading

    app = create_app(store, snapshot_dir)

    def _serve():
        # production=False: dùng werkzeug dev server, đủ cho prototype
        app.run(host=host, port=port, debug=False, use_reloader=False)

    t = threading.Thread(target=_serve, daemon=True)
    t.start()
    print(f"[DASHBOARD] Alert dashboard chạy tại http://localhost:{port}")
    return t


def start_dashboard(store, snapshot_dir, enabled=True):
    """Khởi động dashboard an toàn (Flask chưa cài thì in hướng dẫn)."""
    if not enabled:
        return None
    try:
        import flask  # noqa: F401
    except ImportError:
        print("[DASHBOARD] Chưa cài Flask. Chạy: pip install flask "
              "hoặc thêm vào requirements.txt")
        return None
    return run_dashboard(store, snapshot_dir, host=config_host(), port=config_port())


def config_host():
    import config
    return getattr(config, "DASHBOARD_HOST", "0.0.0.0")


def config_port():
    import config
    return int(getattr(config, "DASHBOARD_PORT", 8080))
