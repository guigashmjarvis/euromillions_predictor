#!/usr/bin/env python3
"""EuroMillions Predictor Dashboard - stdlib only, no Flask needed."""

import json
import socket
import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

BOT_DIR = Path(__file__).parent
DATA_FILE = BOT_DIR / "data" / "draws.json"
RECORDS_FILE = BOT_DIR / "predictions" / "records.json"
CONFIG_FILE = BOT_DIR / "models" / "config.json"

def load_json(fp, default=None):
    if default is None: default = {}
    if Path(fp).exists():
        try:
            with open(fp) as f: return json.load(f)
        except: return default
    return default

def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]; s.close(); return ip
    except: return "localhost"

def build_html(data):
    draws = data["draws"]
    records = data["records"]
    cfg = data["config"]
    local_ip = data["local_ip"]
    
    # Latest draw info
    latest_draw = draws[-1] if draws else None
    next_draw_num = (draws[-1]["draw_number"] + 1) if draws else "?"
    
    # Pending predictions
    pending = [r for r in records if not r.get("checked")]
    checked = [r for r in records if r.get("checked")]
    
    # Accuracy summary
    acc_rows = ""
    if checked:
        for rec in reversed(checked[-5:]):
            actual = rec.get("actual_draw", {})
            a_nums = actual.get("numbers", [])
            a_stars = actual.get("stars", [])
            a_date = actual.get("date", "?")
            preds_html = ""
            for i, pred in enumerate(rec.get("predictions", [])):
                pn = set(pred["numbers"]); ps = set(pred["stars"])
                rn = set(a_nums); rs = set(a_stars)
                hn = sorted(pn & rn); hs = sorted(ps & rs)
                cls = "match" if len(hn) >= 2 or len(hs) >= 1 else ""
                n_str = " ".join(f'<span class="hl" if x in rn else "">{x}</span>' for x in pred["numbers"])
                s_str = " ".join(f'<span class="hl" if x in rs else "">{x}</span>' for x in pred["stars"])
                preds_html += f'<tr class="{cls}"><td>#{i+1}</td><td>{n_str}</td><td>{s_str}</td><td>{len(hn)}/5</td><td>{len(hs)}/2</td></tr>'
            
            acc_rows += f"""
            <div class="section card">
              <h3>Draw #{rec.get('for_draw_number')} ({a_date})</h3>
              <p>Actual: <strong>{" ".join(str(x) for x in a_nums)}</strong> Stars: <strong>{" ".join("★"+str(x) for x in a_stars)}</strong></p>
              <table><thead><tr><th>Set</th><th>Numbers</th><th>Stars</th><th>Match N</th><th>Match S</th></tr></thead>
              <tbody>{preds_html}</tbody></table>
            </div>"""
    
    pending_rows = ""
    for rec in reversed(pending):
        for i, pred in enumerate(rec.get("predictions", [])):
            n_str = " ".join(f"{x:2d}" for x in pred["numbers"])
            s_str = " ".join(f"★{x}" for x in pred["stars"])
            pending_rows += f'<tr><td>{rec.get("for_draw_number","?")}</td><td>{n_str}</td><td>{s_str}</td><td>{rec.get("predicted_at","?")}</td></tr>'

    # Recent actual draws
    draw_rows = ""
    for d in reversed(draws[-10:]):
        n_str = " ".join(f"{x:2d}" for x in d["numbers"])
        s_str = " ".join(f"★{x}" for x in d["stars"])
        draw_rows += f'<tr><td>{d["date"]}</td><td>#{d["draw_number"]}</td><td>{n_str}</td><td>{s_str}</td></tr>'

    model_info = "Not trained"
    if cfg:
        model_info = f'Trained ({cfg.get("trained_at","?")}) | {cfg.get("n_draws","?")} draws | {cfg.get("hidden","?")} hidden'

    return f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>🇪🇺 EuroMillions Predictor</title>
<style>
:root{{--bg:#0d1117;--card:#161b22;--border:#30363d;--text:#c9d1d9;--muted:#8b949e;--green:#2ea043;--red:#da3633;--blue:#58a6ff;--yellow:#d29922}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:var(--bg);color:var(--text);font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif;padding:20px;max-width:1100px;margin:0 auto}}
h1{{font-size:1.6rem;margin-bottom:2px}}.subtitle{{color:var(--muted);font-size:.85rem;margin-bottom:20px}}
.card{{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:16px;margin-bottom:16px}}
.section h3{{margin-bottom:8px;color:var(--blue)}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin-bottom:20px}}
.card-label{{color:var(--muted);font-size:.75rem;text-transform:uppercase;letter-spacing:.05em}}
.card-value{{font-size:1.3rem;font-weight:700;margin-top:4px}}
.card-value.blue{{color:var(--blue)}}.card-value.green{{color:var(--green)}}.card-value.red{{color:var(--red)}}
table{{width:100%;border-collapse:collapse;font-size:.85rem;margin-top:8px}}
th{{text-align:left;padding:8px;border-bottom:1px solid var(--border);color:var(--muted);font-weight:600}}
td{{padding:8px;border-bottom:1px solid var(--border)}}
tr.match{{background:#2386361a}}
.hl{{color:var(--green);font-weight:600}}
.refresh-btn{{background:var(--border);color:var(--text);border:none;padding:6px 14px;border-radius:6px;cursor:pointer;font-size:.8rem;margin-left:8px}}
.refresh-btn:hover{{background:#484f58}}
.stats{{display:flex;gap:12px;flex-wrap:wrap}}
.badge{{display:inline-block;padding:2px 8px;border-radius:12px;font-size:.75rem;font-weight:600;background:var(--border)}}
</style></head><body>
<h1>🇪🇺 EuroMillions Predictor</h1>
<p class="subtitle">LSTM + Frequency Analysis | Server: <a href="http://{local_ip}:3002" style="color:var(--blue)">http://{local_ip}:3002</a> <button class="refresh-btn" onclick="location.reload()">🔄 Refresh</button></p>

<div class="grid">
  <div class="card"><div class="card-label">Latest Draw</div><div class="card-value blue">{latest_draw['date'] if latest_draw else '—'}</div></div>
  <div class="card"><div class="card-label">Next Draw #</div><div class="card-value blue">{next_draw_num}</div></div>
  <div class="card"><div class="card-label">Historical Data</div><div class="card-value">{len(draws)} draws</div></div>
  <div class="card"><div class="card-label">Model</div><div class="card-value" style="font-size:.9rem">{model_info}</div></div>
</div>

<div class="section card"><h3>🎯 Pending Predictions</h3>
{f'<table><thead><tr><th>For Draw #</th><th>Numbers</th><th>Stars</th><th>Predicted</th></tr></thead><tbody>{pending_rows}</tbody></table>' if pending_rows else '<p style="color:var(--muted);margin-top:8px">No pending predictions. Run <code>app.py predict</code> to generate.</p>'}
</div>

<div class="section card"><h3>✅ Past Results</h3>
{acc_rows if acc_rows else '<p style="color:var(--muted);margin-top:8px">No checked predictions yet.</p>'}
</div>

<div class="section card"><h3>📊 Recent Draws</h3>
<table><thead><tr><th>Date</th><th>Draw #</th><th>Numbers</th><th>Stars</th></tr></thead>
<tbody>{draw_rows}</tbody></table></div>

<p style="color:var(--muted);font-size:.7rem;margin-top:16px">Auto-refresh: 30s | Cron runs Mon & Thu</p>
<script>setTimeout(()=>location.reload(),30000)</script>
</body></html>"""


class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        draws = load_json(DATA_FILE, [])
        records = load_json(RECORDS_FILE, [])
        config = load_json(CONFIG_FILE, {})
        data = {"draws": draws, "records": records, "config": config, "local_ip": get_local_ip()}
        html = build_html(data)
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(html.encode())
    
    def log_message(self, format, *args):
        pass

if __name__ == "__main__":
    port = 3002
    local_ip = get_local_ip()
    server = HTTPServer(("0.0.0.0", port), Handler)
    print(f"Dashboard running: http://{local_ip}:{port}")
    try: server.serve_forever()
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
        server.server_close()
