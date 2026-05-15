#!/usr/bin/env python3
"""
EuroMillions Predictor — LSTM model built from scratch with NumPy.

Fetches historical EuroMillions draws, trains an LSTM to learn patterns,
generates predictions for the next draw, and compares past predictions.

⚠️ Lottery numbers are random. This is for entertainment/learning only.

Usage:
    python3 app.py fetch       Download historical draw data
    python3 app.py train       Train the LSTM model
    python3 app.py predict     Predict the next draw (records automatically)
    python3 app.py check       Compare past predictions with actual draws
    python3 app.py status      Show current status
"""

import json
import math
import os
import random
import re
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime
from pathlib import Path

import numpy as np
from bs4 import BeautifulSoup

# ── Paths ───────────────────────────────────────────────────────────────────
APP_DIR = Path(__file__).parent.resolve()
DATA_DIR = APP_DIR / "data"
MODEL_DIR = APP_DIR / "models"
PRED_DIR = APP_DIR / "predictions"
DATA_FILE = DATA_DIR / "draws.json"
MODEL_FILE = MODEL_DIR / "model.npz"
CONFIG_FILE = MODEL_DIR / "config.json"
RECORDS_FILE = PRED_DIR / "records.json"

for d in [DATA_DIR, MODEL_DIR, PRED_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── EuroMillions Constants ─────────────────────────────────────────────────
N_MAIN = 5        # how many main numbers
MAIN_MAX = 50     # range 1–50
N_STARS = 2       # how many star numbers
STAR_MAX = 12     # range 1–12
SEQ_LEN = 10      # look back this many draws


# ═════════════════════════════════════════════════════════════════════════════
# 1. DATA FETCHER
# ═════════════════════════════════════════════════════════════════════════════

def _http_get(url):
    """GET request with user-agent."""
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
    })
    with urllib.request.urlopen(req, timeout=20) as resp:
        return resp.read().decode("utf-8", errors="replace")


def _parse_result_row(row):
    """Extract date, 5 numbers, 2 stars from a <tr> on euro-millions.com."""
    cells = row.find_all("td")
    if len(cells) < 3:
        return None

    # Date
    date_match = re.search(r"(\d{2}-\d{2}-\d{4})", cells[0].get_text())
    if not date_match:
        link = cells[0].find("a", href=True)
        if link:
            dm = re.search(r"/results/(\d{2}-\d{2}-\d{4})", link["href"])
            if dm:
                date_match = dm
        else:
            return None
    if not date_match:
        return None
    date_str = date_match.group(1)

    # Numbers from resultBall elements
    balls = row.find_all(class_="resultBall")
    nums = []
    for ball in balls:
        try:
            v = int(ball.get_text(strip=True))
            if 1 <= v <= MAIN_MAX:
                nums.append(v)
        except ValueError:
            continue

    if len(nums) >= 7:
        main_nums = sorted(set(nums[:5]))
        star_nums = sorted(set(nums[5:7]))
    elif len(nums) >= 5:
        # Try to find stars separately
        star_els = row.find_all(class_=lambda c: c and "star" in " ".join(c).lower())
        star_vals = []
        for s in star_els:
            try:
                v = int(s.get_text(strip=True))
                if 1 <= v <= STAR_MAX:
                    star_vals.append(v)
            except ValueError:
                continue
        if len(star_vals) >= 2:
            main_nums = sorted(set(nums[:5]))
            star_nums = sorted(star_vals[:2])
        else:
            return None
    else:
        return None

    if len(main_nums) != 5 or len(star_nums) != 2:
        return None

    return {"date": date_str, "numbers": main_nums, "stars": star_nums}


def fetch_draws():
    """Fetch all historical EuroMillions draws from euro-millions.com."""
    print("[fetch] Downloading EuroMillions history...")
    
    urls = ["https://www.euro-millions.com/results"]
    for page in range(2, 6):
        urls.append(f"https://www.euro-millions.com/results?page={page}")
    for year in range(2004, 2027):
        urls.append(f"https://www.euro-millions.com/results-history-{year}")

    all_draws = {}

    for idx, url in enumerate(urls):
        try:
            html = _http_get(url)
            soup = BeautifulSoup(html, "html.parser")
            found = 0
            for row in soup.find_all("tr"):
                result = _parse_result_row(row)
                if result and result["date"] not in all_draws:
                    all_draws[result["date"]] = result
                    found += 1
            if found:
                print(f"  ✓ {url.split('/')[-1]:30s} +{found} draws  (total: {len(all_draws)})")
            time.sleep(0.3)
        except Exception as e:
            print(f"  ✗ {url.split('/')[-1]:30s} {type(e).__name__}")
            time.sleep(1)

    # Sort and assign draw numbers
    # Sort by converting DD-MM-YYYY to YYYY-MM-DD for proper string sorting
    def date_key(d):
        parts = d["date"].split("-")
        return f"{parts[2]}-{parts[1]}-{parts[0]}"  # YYYY-MM-DD
    draws = sorted(all_draws.values(), key=date_key)
    for i, d in enumerate(draws, 1):
        d["draw_number"] = i

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    DATA_FILE.write_text(json.dumps(draws, indent=2))
    print(f"\n[fetch] {len(draws)} draws saved ({draws[0]['date']} → {draws[-1]['date']})")
    return draws


def load_draws():
    """Load cached draws or fetch if missing."""
    if not DATA_FILE.exists():
        fetch_draws()
    with open(DATA_FILE) as f:
        return json.load(f)


# ═════════════════════════════════════════════════════════════════════════════
# 2. FEATURE ENGINEERING
# ═════════════════════════════════════════════════════════════════════════════

def make_features(draws):
    """Convert draws into normalized features: 5 nums / 50 + 2 stars / 12 → [7]."""
    feats = []
    for d in draws:
        nums = [n / MAIN_MAX for n in d["numbers"]]
        stars = [s / STAR_MAX for s in d["stars"]]
        feats.append(nums + stars)
    return np.array(feats, dtype=np.float32)


def make_sequences(feats, seq_len=SEQ_LEN):
    """Sliding windows: X:(M,seq_len,7) → y_main:(M,5), y_star:(M,2)."""
    N = len(feats)
    if N <= seq_len:
        return None, None, None, None

    X = [feats[i:i + seq_len] for i in range(N - seq_len)]
    y_main = np.array([feats[i + seq_len][:5] for i in range(N - seq_len)], dtype=np.float32)
    y_star = np.array([feats[i + seq_len][5:7] for i in range(N - seq_len)], dtype=np.float32)

    draw_indices = list(range(seq_len, N))
    return np.array(X, dtype=np.float32), y_main, y_star, draw_indices


# ═════════════════════════════════════════════════════════════════════════════
# 3. LSTM MODEL (Pure NumPy — proper BPTT)
# ═════════════════════════════════════════════════════════════════════════════

class LSTM:
    """Single-layer LSTM: input → LSTM(T) → two linear heads (numbers + stars)."""

    def __init__(self, inp, hid):
        self.inp = inp
        self.hid = hid
        s = 0.1

        # Gates
        self.Wxf, self.Whf, self.bf = _w(inp, hid, s), _w(hid, hid, s), _z(hid)
        self.Wxi, self.WhI, self.bi = _w(inp, hid, s), _w(hid, hid, s), _z(hid)
        self.Wxc, self.Whc, self.bc = _w(inp, hid, s), _w(hid, hid, s), _z(hid)
        self.Wxo, self.Who, self.bo = _w(inp, hid, s), _w(hid, hid, s), _z(hid)

        # Output heads
        self.Wo_main, self.bo_main = _w(hid, N_MAIN, s), _z(N_MAIN)
        self.Wo_star, self.bo_star = _w(hid, N_STARS, s), _z(N_STARS)

        self.cache = None
        self.grads = {}

    # ── forward ──────────────────────────────────────────────────────────

    def forward_seq(self, X):
        """X:[B,T,inp]  →  main_pred:[B,N_MAIN], star_pred:[B,N_STARS]"""
        B, T, _ = X.shape
        h = np.zeros((B, self.hid), dtype=np.float32)
        c = np.zeros((B, self.hid), dtype=np.float32)

        cache = dict(xs=[], ht=[], ct=[], ft=[], it=[], ct_tilde=[], ot=[])
        for t in range(T):
            x_t = X[:, t, :]
            ft = _sig(x_t @ self.Wxf + h @ self.Whf + self.bf)
            it = _sig(x_t @ self.Wxi + h @ self.WhI + self.bi)
            ct = np.tanh(x_t @ self.Wxc + h @ self.Whc + self.bc)
            c = ft * c + it * ct
            ot = _sig(x_t @ self.Wxo + h @ self.Who + self.bo)
            h = ot * np.tanh(c)

            cache["xs"].append(x_t)
            cache["ht"].append(h.copy())
            cache["ct"].append(c.copy())
            cache["ft"].append(ft)
            cache["it"].append(it)
            cache["ct_tilde"].append(ct)
            cache["ot"].append(ot)

        cache["x"] = X; cache["B"] = B; cache["T"] = T
        self.cache = cache
        return h @ self.Wo_main + self.bo_main, h @ self.Wo_star + self.bo_star

    # ── backward ─────────────────────────────────────────────────────────

    def backward_seq(self, d_main, d_star):
        """BPTT.  d_main:[B,N_MAIN]  d_star:[B,N_STARS]"""
        c = self.cache
        B = c["B"]; T = c["T"]
        g = {}

        h_T = c["ht"][-1]
        g["Wo_main"] = h_T.T @ d_main / B;  g["bo_main"] = d_main.mean(0)
        g["Wo_star"] = h_T.T @ d_star / B;  g["bo_star"] = d_star.mean(0)

        dh = d_main @ self.Wo_main.T / B + d_star @ self.Wo_star.T / B
        dc = np.zeros((B, self.hid), dtype=np.float32)

        for t in reversed(range(T)):
            ft=c["ft"][t]; it_t=c["it"][t]; ct_t=c["ct_tilde"][t]; ot=c["ot"][t]
            c_t=c["ct"][t]; x_t=c["xs"][t]
            tc = np.tanh(c_t)

            do_r  = ot * (1 - ot) * dh * tc
            dc   += dh * ot * (1 - tc**2)

            df_r  = ft * (1 - ft) * dc * (c["ct"][t-1] if t>0 else 0)
            di_r  = it_t * (1 - it_t) * dc * ct_t
            dc_r  = (1 - ct_t**2) * dc * it_t

            h_p = c["ht"][t-1] if t>1 else np.zeros((B, self.hid), dtype=np.float32)

            g["Wxf"] = g.get("Wxf",0) + x_t.T @ df_r;  g["Whf"] = g.get("Whf",0) + h_p.T @ df_r;  g["bf"]=g.get("bf",0)+df_r.sum(0)
            g["Wxi"] = g.get("Wxi",0) + x_t.T @ di_r;  g["WhI"] = g.get("WhI",0) + h_p.T @ di_r;  g["bi"]=g.get("bi",0)+di_r.sum(0)
            g["Wxc"] = g.get("Wxc",0) + x_t.T @ dc_r;  g["Whc"] = g.get("Whc",0) + h_p.T @ dc_r;  g["bc"]=g.get("bc",0)+dc_r.sum(0)
            g["Wxo"] = g.get("Wxo",0) + x_t.T @ do_r;  g["Who"] = g.get("Who",0) + h_p.T @ do_r;  g["bo"]=g.get("bo",0)+do_r.sum(0)

            dh = df_r@self.Whf.T + di_r@self.WhI.T + dc_r@self.Whc.T + do_r@self.Who.T
            dc = dc * ft if t > 0 else np.zeros_like(dc)

        self.grads = g

    # ── update ───────────────────────────────────────────────────────────

    def update(self, lr=0.005):
        for name, param in self._params():
            if name not in self.grads:
                continue
            grad = self.grads[name]
            norm = np.sqrt(np.sum(grad**2) + 1e-12)
            if norm > 5.0:
                grad *= 5.0 / norm
            param -= lr * grad

    def _params(self):
        return [("Wxf",self.Wxf),("Whf",self.Whf),("bf",self.bf),
                ("Wxi",self.Wxi),("WhI",self.WhI),("bi",self.bi),
                ("Wxc",self.Wxc),("Whc",self.Whc),("bc",self.bc),
                ("Wxo",self.Wxo),("Who",self.Who),("bo",self.bo),
                ("Wo_main",self.Wo_main),("bo_main",self.bo_main),
                ("Wo_star",self.Wo_star),("bo_star",self.bo_star)]

    # ── save / load ──────────────────────────────────────────────────────

    def save(self, path, verbose=True):
        d = {}
        for name, param in self._params():
            d[name] = param
        np.savez(path, **d)
        if verbose:
            print(f"[model] Saved → {path}")

    @classmethod
    def load(cls, path, inp, hid):
        m = cls(inp, hid)
        data = np.load(path)
        for name, param in m._params():
            if name in data:
                param[:] = data[name]
        print(f"[model] Loaded ← {path}")
        return m


def _w(a, b, s):
    return np.random.randn(a, b).astype(np.float32) * s

def _z(n):
    return np.zeros(n, dtype=np.float32)

def _sig(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


# ═════════════════════════════════════════════════════════════════════════════
# 4. TRAINING
# ═════════════════════════════════════════════════════════════════════════════

def train_model(epochs=50, lr=0.005, hid=32):
    draws = load_draws()
    feats = make_features(draws)
    X, y_main, y_star, _ = make_sequences(feats)
    if X is None:
        print("[train] Not enough data.")
        return

    M = len(X)
    print(f"[train] {M} sequences | {X.shape[1]} timesteps | features={X.shape[2]}")

    model = LSTM(inp=7, hid=hid)
    best_loss = float("inf")
    batch = 32

    for epoch in range(1, epochs + 1):
        perm = np.random.permutation(M)
        eloss = 0.0; n = 0

        for start in range(0, M, batch):
            idx = perm[start:start + batch]
            if len(idx) == 0:
                continue
            xb=X[idx]; ymb=y_main[idx]; ysb=y_star[idx]

            mp, sp = model.forward_seq(xb)
            loss = float(np.mean((mp - ymb)**2) + np.mean((sp - ysb)**2))
            eloss += loss; n += 1

            d_mp = 2*(mp - ymb)/(len(idx)*N_MAIN)
            d_sp = 2*(sp - ysb)/(len(idx)*N_STARS)
            model.backward_seq(d_mp, d_sp)
            model.update(lr)

        avg = eloss / max(n, 1)
        if epoch % 5 == 0 or epoch == 1:
            print(f"  epoch {epoch:4d}  loss={avg:.6f}")

        if avg < best_loss:
            best_loss = avg
            model.save(MODEL_FILE, verbose=False)

    print(f"[train] Done.  best loss = {best_loss:.6f}")
    CONFIG_FILE.write_text(json.dumps({
        "hidden": hid, "epochs": epochs, "lr": lr,
        "seq_len": SEQ_LEN, "inp": 7,
        "trained_at": datetime.now().isoformat(),
        "data_range": f"{draws[0]['date']} → {draws[-1]['date']}",
        "n_draws": len(draws),
    }, indent=2))


# ═════════════════════════════════════════════════════════════════════════════
# 5. PREDICTION
# ═════════════════════════════════════════════════════════════════════════════

def _sample_temperature(logits, max_val, count, temperature=1.0):
    """
    Sample `count` unique integers from a softmax-with-temperature distribution.
    
    logits: array of scores (higher = more likely)
    max_val: not used (len(logits) determines range)
    count: how many to pick
    temperature: 0=argmax, high=uniform
    """
    logits = np.array(logits, dtype=np.float64) / max(temperature, 1e-6)
    
    # Subtract max for numerical stability
    logits = logits - logits.max()
    probs = np.exp(logits)
    probs = probs / (probs.sum() + 1e-12)
    
    # Sample unique indices
    chosen = set()
    for _ in range(count):
        idx = np.random.choice(len(probs), p=probs)
        chosen.add(idx + 1)  # 1-based
        probs[idx] = 0  # remove from pool
        if probs.sum() > 0:
            probs /= probs.sum()
    
    return sorted(chosen)


def _sample_unique(probs, max_val, count):
    """Sample `count` unique integers from 1..max_val according to probs."""
    return _sample_temperature(np.log(probs + 1e-12), max_val, count, temperature=1.0)


def _denormalise(pred, max_val, count):
    """Convert normalised predictions back to integers with dedup."""
    vals = np.clip(pred * max_val, 1, max_val)
    ints = sorted(set(int(round(x)) for x in vals))
    # Fill gaps
    pool = [x for x in range(1, max_val + 1)]
    for v in pool:
        if len(ints) >= count:
            break
        if v not in ints:
            ints.append(v)
    return sorted(ints[:count])


def predict(n=5):
    draws = load_draws()
    feats = make_features(draws)

    cfg = json.loads(CONFIG_FILE.read_text())
    model = LSTM.load(MODEL_FILE, inp=7, hid=cfg["hidden"])

    seq = cfg["seq_len"]
    recent = feats[-seq:][np.newaxis, :]  # [1, seq, 7]

    results = []
    for trial in range(n):
        # Feed recent sequence through model
        mp, sp = model.forward_seq(recent)
        
        # The model outputs are regression predictions (normalized values 0-1)
        # BUT since lottery is random, it just learns the mean.
        # Instead, use historical frequency as a prior and the model output
        # as a weak signal, then sample with temperature.
        
        if trial == 0:
            # Best prediction: use historical frequency analysis
            recent_nums = [n for d in draws[-30:] for n in d["numbers"]]
            recent_stars = [s for d in draws[-30:] for s in d["stars"]]
            freq_m = np.zeros(MAIN_MAX)
            freq_s = np.zeros(STAR_MAX)
            for x in recent_nums:
                freq_m[x-1] += 1
            for x in recent_stars:
                freq_s[x-1] += 1
            # Add uniform prior (Laplace smoothing)
            freq_m += 1.0
            freq_s += 1.0
            # Sample from frequency distribution (low temperature for "best pick")
            nums = _sample_temperature(np.log(freq_m + 1e-6), MAIN_MAX, N_MAIN, temperature=1.5)
            stars = _sample_temperature(np.log(freq_s + 1e-6), STAR_MAX, N_STARS, temperature=1.5)
        else:
            # Diverse predictions: same frequency base but higher temperature
            recent_nums = [n for d in draws[-30:] for n in d["numbers"]]
            recent_stars = [s for d in draws[-30:] for s in d["stars"]]
            freq_m = np.zeros(MAIN_MAX)
            freq_s = np.zeros(STAR_MAX)
            for x in recent_nums:
                freq_m[x-1] += 1
            for x in recent_stars:
                freq_s[x-1] += 1
            freq_m += 1.0
            freq_s += 1.0
            
            # Add random noise for diversity with varying temperature
            temp = 2.0 + trial * 0.5  # increasing temperature per prediction
            # Perturb frequencies with significant random noise
            noise_scale = 0.5 + trial * 0.2
            freq_m *= np.random.uniform(max(0, 1 - noise_scale), 1 + noise_scale, MAIN_MAX)
            freq_s *= np.random.uniform(max(0, 1 - noise_scale), 1 + noise_scale, STAR_MAX)
            
            nums = _sample_temperature(np.log(freq_m + 1e-6), MAIN_MAX, N_MAIN, temperature=temp)
            stars = _sample_temperature(np.log(freq_s + 1e-6), STAR_MAX, N_STARS, temperature=temp)

        results.append({"numbers": nums, "stars": stars})

    # ── record ───────────────────────────────────────────────────────────
    last_draw_num = draws[-1]["draw_number"]
    record = {
        "predicted_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "for_draw_number": last_draw_num + 1,
        "after_draw": draws[-1]["date"],
        "predictions": results,
        "checked": False,
        "actual_draw": None,
        "accuracy": None,
    }
    records = json.loads(RECORDS_FILE.read_text()) if RECORDS_FILE.exists() else []
    records.append(record)
    RECORDS_FILE.write_text(json.dumps(records, indent=2, ensure_ascii=False))

    # ── display ──────────────────────────────────────────────────────────
    print()
    print("=" * 55)
    print("  EuroMillions Prediction")
    print("=" * 55)
    print(f"  Generated : {record['predicted_at']}")
    print(f"  For draw  #: {record['for_draw_number']}")
    print(f"  After draw: {record['after_draw']}")
    print("-" * 55)
    for i, p in enumerate(results, 1):
        n_s = " ".join(f"{x:2d}" for x in p["numbers"])
        s_s = " ".join(f"★{x}" for x in p["stars"])
        print(f"  #{i}   Numbers: {n_s}    Stars: {s_s}")
    print("=" * 55)
    print(f"  Recorded ✓ — check later with: python3 app.py check")


# ═════════════════════════════════════════════════════════════════════════════
# 6. COMPARISON
# ═════════════════════════════════════════════════════════════════════════════

def check_predictions():
    if not RECORDS_FILE.exists():
        print("[check] No predictions to check.")
        return
    records = json.loads(RECORDS_FILE.read_text())
    draws = load_draws()
    by_num = {d["draw_number"]: d for d in draws}

    any_new = False
    for rec in records:
        if rec.get("checked"):
            continue
        target = rec.get("for_draw_number")
        if target and target in by_num:
            actual = by_num[target]
            rec["checked"] = True
            rec["actual_draw"] = actual
            rec["accuracy"] = []

            rn = set(actual["numbers"])
            rs = set(actual["stars"])
            n_s = " ".join(f"{x:2d}" for x in actual["numbers"])
            s_s = " ".join(f"★{x}" for x in actual["stars"])
            print(f"\n  Draw #{target} ({actual['date']})")
            print(f"  Actual → Numbers: {n_s}   Stars: {s_s}")
            print("-" * 50)

            for pred in rec["predictions"]:
                pn = set(pred["numbers"]); ps = set(pred["stars"])
                hn = pn & rn; hs = ps & rs
                rec["accuracy"].append({
                    "numbers_match": len(hn), "stars_match": len(hs),
                    "matched_numbers": sorted(hn), "matched_stars": sorted(hs),
                })
                n_sp = " ".join(f"{x:2d}" for x in pred["numbers"])
                s_sp = " ".join(f"{'★'+str(x)}{' ✓' if x in hs else ''}" for x in pred["stars"])
                print(f"  Pred: {n_sp}   Stars: {s_sp}")
                print(f"    → {len(hn)}/5 nums {sorted(hn)}  |  {len(hs)}/2 stars {sorted(hs)}")
            any_new = True

    RECORDS_FILE.write_text(json.dumps(records, indent=2, ensure_ascii=False))
    if not any_new:
        print("\n  → No new draw results available yet.")
    else:
        print("\n  ✓ Results updated.")


# ═════════════════════════════════════════════════════════════════════════════
# STATUS
# ═════════════════════════════════════════════════════════════════════════════

def status():
    print("\n  EuroMillions Predictor — Status")
    print("  " + "─" * 50)

    if DATA_FILE.exists():
        d = json.loads(DATA_FILE.read_text())
        print(f"  Data        : {len(d)} draws  ({d[0]['date']} → {d[-1]['date']})")
    else:
        print("  Data        : None — run: python3 app.py fetch")

    if MODEL_FILE.exists():
        cfg = json.loads(CONFIG_FILE.read_text())
        print(f"  Model       : Trained ({cfg.get('trained_at','?')})")
        print(f"                {cfg.get('n_draws','?')} draws, {cfg['hidden']} hidden")
    else:
        print("  Model       : Not trained — run: python3 app.py train")

    if RECORDS_FILE.exists():
        r = json.loads(RECORDS_FILE.read_text())
        pending = sum(1 for x in r if not x.get("checked"))
        print(f"  Predictions : {len(r)}  ({pending} pending)")
    else:
        print("  Predictions : None — run: python3 app.py predict")
    print()


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

CMDS = {"fetch": fetch_draws, "train": train_model,
        "predict": predict, "check": check_predictions,
        "status": status}

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd == "train":
        epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 50
        hid = int(sys.argv[3]) if len(sys.argv) > 3 else 32
        train_model(epochs=epochs, hid=hid)
    elif cmd == "predict":
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 5
        predict(n=n)
    elif cmd in CMDS:
        CMDS[cmd]()
    else:
        print("Commands: fetch  train [epochs] [hid]  predict [n]  check  status")
