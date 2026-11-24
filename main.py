"""
Telegram Signal Chart Bot v2  (single file)
-------------------------------------------
Sirf WORKING setups mark hote hain (rule follow hua + move aaya):

  Order Block : structure break (displacement ke saath) -> price OB me wapas
                aaya (retest) -> reaction mila -> SL hit hone se pehle 1%+ move.
                Retest me OB ke bahar wick (liquidity sweep) ho to  *  lagta hai.
  FVG         : displacement wala gap -> retest -> reaction -> 1%+ move before SL.
  Liq Sweep   : swing high/low ka wick se sweep + close wapas andar + rejection
                wick -> SL (sweep wick) hit hone se pehle 1%+ move.
  EMA         : 100 / 200 lines.

Timeframes: 5m, 15m, 1h   |   Date: IST   |   Output: PDF

Commands
  /ob  btc 23-11-2026        /fvg eth 23-11-2026
  /liq sol 23-11-2026        /ema xauusdt 23-11-2026
  /all btc 23-11-2026        /ob btc,eth,sol 23-11-2026  (max 3)

LIVE (auto alert + PDF, same rules):
  /live btc,eth,xau      -> in coins ka live scan on (new trade bante hi alert + PDF)
  /live                  -> list dekho        /stoplive [btc]  -> band karo

Setup:  pip install -r requirements.txt   ->   BOT_TOKEN set karo   ->   python bot-3.py
"""
import asyncio
import io
import json
import logging
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch, Rectangle

# ----------------------------- CONFIG ---------------------------------
BOT_TOKEN = os.getenv("BOT_TOKEN", "PASTE_YOUR_BOT_TOKEN_HERE")

MOVE_PCT = 1.0        # entry ke baad kam se kam itna % move (target)
LOOKAHEAD = 30        # entry ke baad itne candles me target aana chahiye
RETEST_WINDOW = 60    # OB/FVG banne ke baad itne candles me retest aana chahiye
SWING_N = 3           # swing high/low strength (candles each side)
DISP_ATR = 1.2        # displacement: impulse candle body >= 1.2 x ATR
MAX_ZONE_ATR = 3.0    # OB zone ATR ke 3 guna se bada ho to ignore
FVG_ATR = 0.15        # FVG gap kam se kam 0.15 x ATR
MIN_FVG_PCT = 0.02    # ya price ka 0.02%
SWEEP_ATR = 0.10      # sweep wick level ke paar kam se kam 0.10 x ATR
SWEEP_WICK = 0.40     # rejection wick candle range ka kam se kam 40%
WARMUP = 220          # day se pehle extra candles (EMA 200 + structure)
MAX_SYMBOLS = 3

TFS = ["5m", "15m", "1h"]
TF_MS = {"5m": 300_000, "15m": 900_000, "1h": 3_600_000}
CONTEXT = {"5m": 0, "15m": 0, "1h": 24}  # 1h chart khali na lage isliye pichle candles

IST = timezone(timedelta(hours=5, minutes=30))
ENDPOINTS = [
    "https://api.binance.com/api/v3/klines",
    "https://data-api.binance.vision/api/v3/klines",
    "https://fapi.binance.com/fapi/v1/klines",  # futures (XAUUSDT yahi milta hai)
]
ALIASES = {"GOLD": "XAUUSDT", "XAU": "XAUUSDT"}
QUOTES = ("USDT", "USDC", "FDUSD", "BUSD")

LOCK = threading.Lock()  # matplotlib thread-safe nahi hai
log = logging.getLogger("signal-bot")


# ----------------------------- HELPERS --------------------------------
def fmt(p):
    if p < 1:
        return f"{p:.6f}"
    if p < 100:
        return f"{p:.4f}"
    return f"{p:,.2f}"


def norm_symbol(tok):
    t = tok.upper().strip()
    t = ALIASES.get(t, t)
    return t if t.endswith(QUOTES) else t + "USDT"


def parse_date(s):
    m = re.fullmatch(r"(\d{1,2})[-/.](\d{1,2})[-/.](\d{4})", s)
    if not m:
        return None
    d, mo, y = map(int, m.groups())
    try:
        return datetime(y, mo, d, tzinfo=IST)
    except ValueError:
        return None


def parse_args(args):
    text = " ".join(args)
    m = re.search(r"\d{1,2}[-/.]\d{1,2}[-/.]\d{4}", text)
    if m:
        day = parse_date(m.group(0))
        text = text.replace(m.group(0), " ")
        if day is None:
            return [], None
    else:
        day = datetime.now(IST).replace(hour=0, minute=0, second=0, microsecond=0)
    toks = [t for t in re.split(r"[,/\s]+", text) if t and t.lower() != "today"]
    return [norm_symbol(t) for t in toks], day


# ----------------------------- DATA -----------------------------------
def fetch_klines(symbol, tf, start_ms, end_ms):
    rows, last_err = [], None
    for url in ENDPOINTS:
        rows, cur = [], start_ms
        try:
            while cur < end_ms:
                r = requests.get(
                    url,
                    params={"symbol": symbol, "interval": tf, "startTime": cur,
                            "endTime": end_ms, "limit": 1000},
                    timeout=20,
                )
                if r.status_code != 200:
                    raise RuntimeError(f"HTTP {r.status_code}")
                data = r.json()
                if not data:
                    break
                rows += data
                cur = data[-1][0] + 1
                if len(data) < 1000:
                    break
            if rows:
                break
        except Exception as e:  # next endpoint try karo
            last_err = e
            rows = []
    if not rows:
        raise RuntimeError(f"{symbol} ka data nahi mila ({last_err or 'symbol galat?'})")
    df = pd.DataFrame(rows).iloc[:, :6]
    df.columns = ["ot", "open", "high", "low", "close", "vol"]
    df = df.drop_duplicates("ot").reset_index(drop=True)
    for c in ["open", "high", "low", "close", "vol"]:
        df[c] = df[c].astype(float)
    df["t"] = pd.to_datetime(df["ot"], unit="ms", utc=True).dt.tz_convert(IST)
    return df


# --------------------------- DETECTION --------------------------------
def atr_arr(h, l, c, n=14):
    pc = np.r_[c[0], c[:-1]]
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    return pd.Series(tr).rolling(n, min_periods=1).mean().values


def swing_points(h, l, n=SWING_N):
    sh, sl = [], []
    for i in range(n, len(h) - n):
        wh, wl = h[i - n:i + n + 1], l[i - n:i + n + 1]
        if h[i] == wh.max() and (wh == h[i]).sum() == 1:
            sh.append(i)
        if l[i] == wl.min() and (wl == l[i]).sum() == 1:
            sl.append(i)
    return sh, sl


def trade_outcome(d, entry, stop, start, h, l):
    """
    Entry ke baad candles scan karo. SL pehle hit hua to fail.
    Target (MOVE_PCT) SL se pehle hit hua to success.
    Return (ok, move_pct, rr, peak_idx).  Same candle me dono -> fail (conservative).
    """
    risk = abs(entry - stop)
    if risk <= 0:
        return False, 0.0, 0.0, start
    bull = d == "bull"
    target = entry * (1 + MOVE_PCT / 100) if bull else entry * (1 - MOVE_PCT / 100)
    peak, pk, hit = entry, start, False
    for m in range(start, min(len(h), start + LOOKAHEAD)):
        if bull:
            if l[m] <= stop:
                break
            if h[m] > peak:
                peak, pk = h[m], m
            if peak >= target:
                hit = True
        else:
            if h[m] >= stop:
                break
            if l[m] < peak or peak == entry:
                if l[m] < peak:
                    peak, pk = l[m], m
            if peak <= target:
                hit = True
    if not hit:
        return False, 0.0, 0.0, pk
    mv = abs(peak - entry) / entry * 100
    return True, mv, abs(peak - entry) / risk, pk


def detect_sweeps(o, h, l, c, atr):
    """Wick ne swing high/low liya, candle wapas andar close hui, rejection wick."""
    sh, sl = swing_points(h, l)
    out, used, N = [], set(), len(c)
    for i in range(SWING_N * 2, N):
        rngc = h[i] - l[i]
        if rngc <= 0:
            continue
        # bearish: swing high ke upar ka liquidity liya
        for j in reversed([x for x in sh if x + SWING_N < i and x >= i - 150]):
            if ("h", j) in used:
                continue
            lvl = h[j]
            if j + 1 < i and h[j + 1:i].max() >= lvl:
                continue
            if h[i] > lvl and c[i] < lvl:
                used.add(("h", j))
                up_wick = h[i] - max(o[i], c[i])
                if h[i] - lvl >= SWEEP_ATR * atr[i] and up_wick >= SWEEP_WICK * rngc:
                    ok, mv, rr, pk = trade_outcome("bear", c[i], h[i], i + 1, h, l)
                    out.append(dict(kind="LIQ", dir="bear", anchor=i, formed=j, x0=j, x1=i,
                                    top=lvl, bottom=lvl, entry=c[i], stop=h[i],
                                    move=mv, rr=rr, pk=pk, ok=ok))
                break
        # bullish: swing low ke neeche ka liquidity liya
        for j in reversed([x for x in sl if x + SWING_N < i and x >= i - 150]):
            if ("l", j) in used:
                continue
            lvl = l[j]
            if j + 1 < i and l[j + 1:i].min() <= lvl:
                continue
            if l[i] < lvl and c[i] > lvl:
                used.add(("l", j))
                lo_wick = min(o[i], c[i]) - l[i]
                if lvl - l[i] >= SWEEP_ATR * atr[i] and lo_wick >= SWEEP_WICK * rngc:
                    ok, mv, rr, pk = trade_outcome("bull", c[i], l[i], i + 1, h, l)
                    out.append(dict(kind="LIQ", dir="bull", anchor=i, formed=j, x0=j, x1=i,
                                    top=lvl, bottom=lvl, entry=c[i], stop=l[i],
                                    move=mv, rr=rr, pk=pk, ok=ok))
                break
    return out


def find_retest(d, top, bot, start, h, l, c, N):
    """Zone ka pehla retest. Zone ke through close ho gaya to invalid (None)."""
    for m in range(start, min(N, start + RETEST_WINDOW)):
        if d == "bull":
            if c[m] < bot:
                return None
            if l[m] <= top:
                return m
        else:
            if c[m] > top:
                return None
            if h[m] >= bot:
                return m
    return None


def detect_obs(o, h, l, c, atr, sweeps):
    """
    Working OB = displacement ke saath BOS -> retest -> reaction -> 1%+ move before SL.
    * = retest me OB ke bahar wick (liquidity sweep) ya sweep candle confluence.
    """
    sh, sl = swing_points(h, l)
    out, seen, N = [], set(), len(c)

    def sweep_conf(m, d):
        return any(s["dir"] == d and m - 2 <= s["anchor"] <= m for s in sweeps)

    for j in sh:  # bullish OB: swing high ke upar close
        lvl, st = h[j], j + SWING_N + 1
        end = min(N, j + 120)
        if st >= end:
            continue
        idx = np.where(c[st:end] > lvl)[0]
        if not len(idx):
            continue
        brk = st + idx[0]
        k = next((x for x in range(brk - 1, j, -1) if c[x] < o[x]), None)
        if k is None or (k, "bull") in seen:
            continue
        seen.add((k, "bull"))
        top, bot, a = h[k], l[k], atr[k]
        if not (0 < top - bot <= MAX_ZONE_ATR * a):
            continue
        if (c[k + 1:brk + 1] - o[k + 1:brk + 1]).max() < DISP_ATR * a:
            continue
        m = find_retest("bull", top, bot, brk + 1, h, l, c, N)
        if m is None:
            continue
        entry, stop = top, min(bot, l[m])
        ok, mv, rr, pk = trade_outcome("bull", entry, stop, m + 1, h, l)
        if not ok:
            continue
        star = l[m] < bot or sweep_conf(m, "bull")
        out.append(dict(kind="OB", dir="bull", anchor=m, formed=k, x0=k, x1=pk, top=top,
                        bottom=bot, entry=entry, stop=stop, move=mv, rr=rr, pk=pk,
                        star=star, ok=True))

    for j in sl:  # bearish OB: swing low ke neeche close
        lvl, st = l[j], j + SWING_N + 1
        end = min(N, j + 120)
        if st >= end:
            continue
        idx = np.where(c[st:end] < lvl)[0]
        if not len(idx):
            continue
        brk = st + idx[0]
        k = next((x for x in range(brk - 1, j, -1) if c[x] > o[x]), None)
        if k is None or (k, "bear") in seen:
            continue
        seen.add((k, "bear"))
        top, bot, a = h[k], l[k], atr[k]
        if not (0 < top - bot <= MAX_ZONE_ATR * a):
            continue
        if (o[k + 1:brk + 1] - c[k + 1:brk + 1]).max() < DISP_ATR * a:
            continue
        m = find_retest("bear", top, bot, brk + 1, h, l, c, N)
        if m is None:
            continue
        entry, stop = bot, max(top, h[m])
        ok, mv, rr, pk = trade_outcome("bear", entry, stop, m + 1, h, l)
        if not ok:
            continue
        star = h[m] > top or sweep_conf(m, "bear")
        out.append(dict(kind="OB", dir="bear", anchor=m, formed=k, x0=k, x1=pk, top=top,
                        bottom=bot, entry=entry, stop=stop, move=mv, rr=rr, pk=pk,
                        star=star, ok=True))
    return out


def detect_fvgs(o, h, l, c, atr):
    """Displacement wala FVG -> retest -> reaction -> 1%+ move before SL."""
    out, N = [], len(c)
    for i in range(2, N):
        a = atr[i - 1]
        min_gap = max(FVG_ATR * a, c[i] * MIN_FVG_PCT / 100)
        if l[i] > h[i - 2] and l[i] - h[i - 2] >= min_gap \
                and c[i - 1] - o[i - 1] >= DISP_ATR * a:
            top, bot = l[i], h[i - 2]
            m = find_retest("bull", top, bot, i + 1, h, l, c, N)
            if m is None:
                continue
            stop = min(bot, l[m])
            ok, mv, rr, pk = trade_outcome("bull", top, stop, m + 1, h, l)
            if ok:
                out.append(dict(kind="FVG", dir="bull", anchor=m, formed=i, x0=i - 2, x1=pk,
                                top=top, bottom=bot, entry=top, stop=stop, move=mv, rr=rr,
                                pk=pk, ok=True))
        elif h[i] < l[i - 2] and l[i - 2] - h[i] >= min_gap \
                and o[i - 1] - c[i - 1] >= DISP_ATR * a:
            top, bot = l[i - 2], h[i]
            m = find_retest("bear", top, bot, i + 1, h, l, c, N)
            if m is None:
                continue
            stop = max(top, h[m])
            ok, mv, rr, pk = trade_outcome("bear", bot, stop, m + 1, h, l)
            if ok:
                out.append(dict(kind="FVG", dir="bear", anchor=m, formed=i, x0=i - 2, x1=pk,
                                top=top, bottom=bot, entry=bot, stop=stop, move=mv, rr=rr,
                                pk=pk, ok=True))
    return out


# ----------------------------- CHARTS ---------------------------------
COL = {
    ("OB", "bull"): "#00c853", ("OB", "bear"): "#ff1744",
    ("FVG", "bull"): "#2979ff", ("FVG", "bear"): "#ff9100",
}
LIQ_COL = "#ab47bc"


def make_chart(df, tf, symbol, sigs, lo, hi, kinds, day):
    sub = df.iloc[lo:hi + 1]
    x = np.arange(lo, hi + 1)
    rng = max(sub.high.max() - sub.low.min(), 1e-9)
    fig, ax = plt.subplots(figsize=(16, 9))

    cols = ["#26a69a" if c >= o else "#ef5350" for o, c in zip(sub.open, sub.close)]
    ax.vlines(x, sub.low, sub.high, colors=cols, linewidth=0.8, zorder=3)
    body = np.maximum((sub.close - sub.open).abs().values, rng * 1e-4)
    ax.bar(x, body, bottom=np.minimum(sub.open, sub.close).values, width=0.7,
           color=cols, zorder=3)

    handles = []
    if "ema" in kinds:
        e100 = df.close.ewm(span=100, adjust=False).mean()
        e200 = df.close.ewm(span=200, adjust=False).mean()
        ax.plot(x, e100.iloc[lo:hi + 1], color="#fbc02d", lw=1.4, zorder=4)
        ax.plot(x, e200.iloc[lo:hi + 1], color="#7e57c2", lw=1.4, zorder=4)
        handles += [plt.Line2D([0], [0], color="#fbc02d", lw=2, label="EMA 100"),
                    plt.Line2D([0], [0], color="#7e57c2", lw=2, label="EMA 200")]

    for s in sigs:
        a, bull = s["anchor"], s["dir"] == "bull"
        arrow = "↑" if bull else "↓"
        if s["kind"] in ("OB", "FVG"):
            col = COL[(s["kind"], s["dir"])]
            star = s.get("star")
            x0, x1 = max(s["x0"], lo), min(s["x1"], hi)
            if x1 >= x0:
                ax.add_patch(Rectangle(
                    (x0 - 0.5, s["bottom"]), x1 - x0 + 1,
                    max(s["top"] - s["bottom"], rng * 1e-4),
                    facecolor=col, alpha=0.28, edgecolor="#ffd600" if star else col,
                    linewidth=2.2 if star else 0.8, zorder=2))
            tag = (f"{s['kind']}{'*' if star else ''} {arrow} "
                   f"+{s['move']:.1f}% | RR {s['rr']:.1f}")
            ax.text(max(s["x0"], lo), s["top"] + rng * 0.004, tag, fontsize=7, color=col,
                    va="bottom", zorder=5)
        else:
            col = LIQ_COL
            ax.hlines(s["top"], max(s["x0"], lo), a, colors=col, linestyles="--",
                      linewidth=1.1, zorder=2)
            tag = f"Sweep {arrow} +{s['move']:.1f}% | RR {s['rr']:.1f}"
            ytxt = df.low[a] - rng * 0.05 if bull else df.high[a] + rng * 0.03
            ax.text(a, ytxt, tag, fontsize=7, color=col, ha="center", zorder=6)

        # entry marker (retest / sweep candle) + SL + move line
        ym = df.low[a] - rng * 0.015 if bull else df.high[a] + rng * 0.015
        ax.scatter([a], [ym], marker="^" if bull else "v", s=60, color=col, zorder=6)
        end = min(s["pk"], hi)
        ax.hlines(s["stop"], a, max(end, a + 1), colors="#d50000", linestyles=":",
                  linewidth=1.0, zorder=2)
        if s["pk"] <= hi:
            peak = df.high[s["pk"]] if bull else df.low[s["pk"]]
            ax.plot([a, s["pk"]], [s["entry"], peak], color=col, ls=":", lw=1.2, zorder=4)

    if "ob" in kinds:
        handles += [Patch(color=COL[("OB", "bull")], alpha=0.4, label="Bullish OB"),
                    Patch(color=COL[("OB", "bear")], alpha=0.4, label="Bearish OB"),
                    Patch(facecolor="none", edgecolor="#ffd600", lw=2, label="OB + Sweep (*)")]
    if "fvg" in kinds:
        handles += [Patch(color=COL[("FVG", "bull")], alpha=0.4, label="Bullish FVG"),
                    Patch(color=COL[("FVG", "bear")], alpha=0.4, label="Bearish FVG")]
    if "liq" in kinds:
        handles.append(plt.Line2D([0], [0], color=LIQ_COL, ls="--", marker="v",
                                  label="Liquidity sweep"))
    if sigs:
        handles.append(plt.Line2D([0], [0], color="#d50000", ls=":", label="Stop loss"))

    ax.set_ylim(sub.low.min() - rng * 0.08, sub.high.max() + rng * 0.08)
    ax.set_xlim(lo - 1, hi + 2)
    step = max(1, len(x) // 12)
    ax.set_xticks(x[::step])
    ax.set_xticklabels(sub.t.iloc[::step].dt.strftime("%d-%m %H:%M"), rotation=30, fontsize=8)
    ax.grid(alpha=0.15)
    ax.set_title(f"{symbol}  {tf}  |  {day:%d-%m-%Y} (IST)  |  {len(sigs)} working setups "
                 f"(>= {MOVE_PCT}% move)", fontsize=13, weight="bold")
    if handles:
        ax.legend(handles=handles, loc="upper left", fontsize=8, ncol=2)
    fig.tight_layout()
    return fig


def summary_pages(pdf, rows, header):
    cols = ["TF", "Zone bana", "Entry time", "Type", "Dir", "Zone / Level",
            "Entry", "SL", "Move %", "RR"]
    per = 24
    chunks = [rows[i:i + per] for i in range(0, len(rows), per)] or [[]]
    for ch in chunks:
        fig = plt.figure(figsize=(16, 9))
        fig.text(0.03, 0.94, header, fontsize=14, weight="bold")
        ax = fig.add_axes([0.02, 0.03, 0.96, 0.86])
        ax.axis("off")
        if ch:
            tbl = ax.table(cellText=ch, colLabels=cols, loc="upper center", cellLoc="center")
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(8)
            tbl.scale(1, 1.5)
        else:
            ax.text(0.5, 0.6, f"Koi working setup nahi mila (>= {MOVE_PCT}% move wala)",
                    ha="center", fontsize=14)
        pdf.savefig(fig)
        plt.close(fig)


# ----------------------------- REPORT ---------------------------------
def _build_report(symbol, day, kinds):
    day_end = day + timedelta(days=1)
    table, figs, got_any = [], [], False
    for tf in TFS:
        ms, ctx = TF_MS[tf], CONTEXT[tf]
        s_ms = int(day.timestamp() * 1000) - (WARMUP + ctx) * ms
        e_ms = int(day_end.timestamp() * 1000) + (LOOKAHEAD + 1) * ms
        try:
            df = fetch_klines(symbol, tf, s_ms, e_ms)
        except Exception as e:
            log.warning("%s %s: %s", symbol, tf, e)
            continue
        disp_start = day - timedelta(milliseconds=ctx * ms)
        mask = ((df.t >= disp_start) & (df.t < day_end)).values
        if not mask.any():
            continue
        got_any = True
        idx = np.where(mask)[0]
        lo, hi = int(idx[0]), int(idx[-1])
        o, h, l, c = (df[k].values for k in ("open", "high", "low", "close"))
        atr = atr_arr(h, l, c)

        sweeps = detect_sweeps(o, h, l, c, atr)
        sigs = []
        if "liq" in kinds:
            sigs += [s for s in sweeps if s["ok"]]
        if "ob" in kinds:
            sigs += detect_obs(o, h, l, c, atr, sweeps)
        if "fvg" in kinds:
            sigs += detect_fvgs(o, h, l, c, atr)
        sigs = [s for s in sigs if lo <= s["anchor"] <= hi]
        sigs.sort(key=lambda s: s["anchor"])

        for s in sigs:
            name = {"OB": "Order Block", "FVG": "FVG", "LIQ": "Liq Sweep"}[s["kind"]]
            if s.get("star"):
                name += " (+Sweep)"
            zone = (f"swept {fmt(s['top'])}" if s["kind"] == "LIQ"
                    else f"{fmt(s['bottom'])} - {fmt(s['top'])}")
            table.append([tf, df.t[s["formed"]].strftime("%d-%m %H:%M"),
                          df.t[s["anchor"]].strftime("%d-%m %H:%M"), name,
                          "Bull" if s["dir"] == "bull" else "Bear", zone,
                          fmt(s["entry"]), fmt(s["stop"]), f"+{s['move']:.2f}",
                          f"{s['rr']:.1f}"])
        figs.append(make_chart(df, tf, symbol, sigs, lo, hi, kinds, day))

    if not got_any:
        raise RuntimeError("Is date ka data nahi mila (date check karo)")

    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        label = "+".join(k.upper() for k in kinds)
        summary_pages(pdf, table, f"{symbol} | {label} | {day:%d-%m-%Y} IST | "
                                  f"Working setups: move >= {MOVE_PCT}% before SL")
        for f in figs:
            pdf.savefig(f)
            plt.close(f)
    return buf.getvalue(), len(table)


def build_report(symbol, day, kinds):
    with LOCK:
        return _build_report(symbol, day, kinds)


# ============================ LIVE MODE ===============================
# Wahi rules (OB / FVG / Liq Sweep) - bas outcome filter nahi (future abhi aaya nahi).
#  OB / FVG : retest ka pehla touch (chalti candle me hi, close ka wait nahi)
#  Sweep    : sweep candle close hote hi
# Alert ke baad wahi trade_outcome rule: SL pehle = fail, 1% pehle = success.
LIVE_BARS = 400       # har TF ke itne candles (EMA/structure warmup ke liye)
POLL_SEC = 4          # live market check interval (seconds)
MAX_LIVE = 10         # ek chat me max live symbols
SUBS_FILE = "subs.json"
SUBS, SEEN, WARM, TRADES, EP_OK = {}, set(), set(), {}, {}


def load_subs():
    try:
        with open(SUBS_FILE) as f:
            SUBS.update(json.load(f))
    except Exception:
        pass


def save_subs():
    try:
        with open(SUBS_FILE, "w") as f:
            json.dump(SUBS, f)
    except Exception:
        log.exception("subs save failed")


def fetch_last(symbol, tf, limit=LIVE_BARS):
    """Latest candles (last wali abhi chal rahi / forming hoti hai)."""
    first = EP_OK.get(symbol, ENDPOINTS[0])
    last_err = None
    for url in [first] + [u for u in ENDPOINTS if u != first]:
        try:
            r = requests.get(url, params={"symbol": symbol, "interval": tf, "limit": limit},
                             timeout=8)
            data = r.json() if r.status_code == 200 else None
            if data:
                EP_OK[symbol] = url
                df = pd.DataFrame(data).iloc[:, :6]
                df.columns = ["ot", "open", "high", "low", "close", "vol"]
                for col in ["open", "high", "low", "close", "vol"]:
                    df[col] = df[col].astype(float)
                df["t"] = pd.to_datetime(df["ot"], unit="ms", utc=True).dt.tz_convert(IST)
                return df
            last_err = f"HTTP {r.status_code}"
        except Exception as e:
            last_err = e
    raise RuntimeError(f"{symbol} ka data nahi mila ({last_err})")


def _mk(symbol, tf, df, kind, d, anchor, formed, x0, top, bot, entry, stop, star, rules, key_i):
    bull = d == "bull"
    target = entry * (1 + MOVE_PCT / 100) if bull else entry * (1 - MOVE_PCT / 100)
    risk = abs(entry - stop)
    return dict(symbol=symbol, tf=tf, kind=kind, dir=d, anchor=int(anchor), formed=int(formed),
                x0=int(x0), top=top, bottom=bot, entry=entry, stop=stop, target=target,
                rr=abs(target - entry) / risk if risk > 0 else 0.0, star=star, rules=rules,
                price=float(df.close.iloc[-1]), t0=int(df.ot[anchor]),
                key=(symbol, tf, kind, d, int(df.ot[key_i])))


def live_signals(symbol, tf, df):
    """Same OB / FVG / Sweep rules, bas abhi ke naye entry moment par (outcome filter ke bina)."""
    o, h, l, c = (df[k].values for k in ("open", "high", "low", "close"))
    N = len(c)
    nc = N - 1  # last candle abhi chal rahi hai
    if nc < 60:
        return []
    atr = atr_arr(h, l, c)
    T = lambda i: df.t[i].strftime("%d-%m %H:%M")
    sweeps = detect_sweeps(o[:nc], h[:nc], l[:nc], c[:nc], atr[:nc])
    out = []

    # ---------------- Liquidity sweep (candle close par) ----------------
    for s in sweeps:
        i = s["anchor"]
        if i < nc - 2:
            continue
        bull = s["dir"] == "bull"
        rc = h[i] - l[i]
        wick = ((min(o[i], c[i]) - l[i]) if bull else (h[i] - max(o[i], c[i]))) / rc * 100
        ext = abs(l[i] - s["top"]) if bull else abs(h[i] - s["top"])
        side = "low" if bull else "high"
        rules = [
            f"Swing {side} {fmt(s['top'])} ({T(s['formed'])}) ko wick ne sweep kiya "
            f"({ext / atr[i]:.2f}x ATR paar, min {SWEEP_ATR})",
            f"Candle {T(i)} wapas andar close hui: close {fmt(c[i])}",
            f"Rejection wick = {wick:.0f}% of candle range (min {SWEEP_WICK * 100:.0f}%)",
            f"Entry = sweep candle close, SL = sweep wick ({fmt(s['stop'])})",
        ]
        out.append(_mk(symbol, tf, df, "LIQ", s["dir"], i, s["formed"], s["formed"],
                       s["top"], s["top"], c[i], s["stop"], False, rules, i))

    # ---------------- Order Block (retest touch par) ----------------
    def sweep_conf(m, d):
        return any(s["dir"] == d and m - 2 <= s["anchor"] <= m for s in sweeps)

    sh, sl = swing_points(h[:nc], l[:nc])
    seen = set()
    for d, pts in (("bull", sh), ("bear", sl)):
        bull = d == "bull"
        for j in pts:
            lvl = h[j] if bull else l[j]
            st, end = j + SWING_N + 1, min(nc, j + 120)
            if st >= end:
                continue
            seg = c[st:end]
            idx = np.where(seg > lvl)[0] if bull else np.where(seg < lvl)[0]
            if not len(idx):
                continue
            brk = st + idx[0]
            if bull:
                k = next((x for x in range(brk - 1, j, -1) if c[x] < o[x]), None)
            else:
                k = next((x for x in range(brk - 1, j, -1) if c[x] > o[x]), None)
            if k is None or (k, d) in seen:
                continue
            seen.add((k, d))
            top, bot, a = h[k], l[k], atr[k]
            if not (0 < top - bot <= MAX_ZONE_ATR * a):
                continue
            body = (c[k + 1:brk + 1] - o[k + 1:brk + 1]) if bull else (o[k + 1:brk + 1] - c[k + 1:brk + 1])
            if len(body) == 0 or body.max() < DISP_ATR * a:
                continue
            m = find_retest(d, top, bot, brk + 1, h, l, c, N)
            if m is None or m < N - 2:
                continue
            if bull:
                entry, stop = top, min(bot, l[m])
                wick_out = l[m] < bot
            else:
                entry, stop = bot, max(top, h[m])
                wick_out = h[m] > top
            star = wick_out or sweep_conf(m, d)
            rules = [
                f"Structure break: {T(brk)} candle swing {'high' if bull else 'low'} {fmt(lvl)} "
                f"({T(j)}) ke {'upar' if bull else 'neeche'} close hui",
                f"Order Block: {T(k)} ki {'bearish' if bull else 'bullish'} candle, zone "
                f"{fmt(bot)} - {fmt(top)} (size {(top - bot) / a:.1f}x ATR, max {MAX_ZONE_ATR})",
                f"Displacement: impulse body {body.max() / a:.1f}x ATR (min {DISP_ATR})",
                f"Retest: {T(m)} par price OB me wapas aaya, zone ke through close nahi hua",
            ]
            if star:
                rules.append("* Sweep confluence: " + (
                    "retest me OB ke bahar wick gayi" if wick_out
                    else "retest ke aas-paas liquidity sweep candle bani"))
            rules.append(f"Entry = OB {'top' if bull else 'bottom'}, SL = {fmt(stop)} "
                         f"(OB {'bottom' if bull else 'top'} / retest wick)")
            out.append(_mk(symbol, tf, df, "OB", d, m, k, k, top, bot, entry, stop, star, rules, k))

    # ---------------- FVG (retest touch par) ----------------
    for i in range(max(2, N - 63), nc):
        a = atr[i - 1]
        min_gap = max(FVG_ATR * a, c[i] * MIN_FVG_PCT / 100)
        if l[i] > h[i - 2] and l[i] - h[i - 2] >= min_gap and c[i - 1] - o[i - 1] >= DISP_ATR * a:
            d, top, bot, body = "bull", l[i], h[i - 2], c[i - 1] - o[i - 1]
        elif h[i] < l[i - 2] and l[i - 2] - h[i] >= min_gap and o[i - 1] - c[i - 1] >= DISP_ATR * a:
            d, top, bot, body = "bear", l[i - 2], h[i], o[i - 1] - c[i - 1]
        else:
            continue
        m = find_retest(d, top, bot, i + 1, h, l, c, N)
        if m is None or m < N - 2:
            continue
        if d == "bull":
            entry, stop = top, min(bot, l[m])
        else:
            entry, stop = bot, max(top, h[m])
        rules = [
            f"FVG gap {fmt(bot)} - {fmt(top)} ({T(i - 2)} se {T(i)}), size {top - bot:.4g} "
            f"(min {FVG_ATR}x ATR / {MIN_FVG_PCT}%)",
            f"Displacement: beech wali candle body {body / a:.1f}x ATR (min {DISP_ATR})",
            f"Retest: {T(m)} par price gap me wapas aaya, gap ke through close nahi hua",
            f"Entry = gap {'top' if d == 'bull' else 'bottom'}, SL = {fmt(stop)} (gap ka doosra end / retest wick)",
        ]
        out.append(_mk(symbol, tf, df, "FVG", d, m, i, i - 2, top, bot, entry, stop, False, rules, i))
    return out


def signal_text(s):
    buy = s["dir"] == "bull"
    name = {"OB": "Order Block", "FVG": "FVG", "LIQ": "Liq Sweep"}[s["kind"]]
    if s["star"]:
        name += " (+Sweep *)"
    lines = [
        f"🚨 NEW TRADE | {s['symbol']} | {s['tf']}",
        f"{'🟢 BUY / LONG' if buy else '🔴 SELL / SHORT'}  -  {name}",
        "",
        f"Entry: {fmt(s['entry'])}",
        f"SL: {fmt(s['stop'])}",
        f"Target (+{MOVE_PCT:g}%): {fmt(s['target'])}  (RR {s['rr']:.1f})",
        f"Abhi price: {fmt(s['price'])}",
        "",
        "Rule jo follow hua:",
    ]
    lines += [f"✅ {r}" for r in s["rules"]]
    lines += ["", f"⏱ {datetime.now(IST):%d-%m %H:%M:%S} IST"]
    return "\n".join(lines)


def live_pdf(df, s):
    buy = s["dir"] == "bull"
    col = "#00c853" if buy else "#ff1744"
    N = len(df)
    lo = max(0, N - 110)
    sub = df.iloc[lo:]
    x = np.arange(lo, N)
    ymin = min(sub.low.min(), s["stop"], s["target"])
    ymax = max(sub.high.max(), s["stop"], s["target"])
    rng = max(ymax - ymin, 1e-9)
    a = s["anchor"]
    with LOCK:
        fig = plt.figure(figsize=(16, 10))
        ax = fig.add_axes([0.05, 0.31, 0.92, 0.62])
        cc = ["#26a69a" if cl >= op else "#ef5350" for op, cl in zip(sub.open, sub.close)]
        ax.vlines(x, sub.low, sub.high, colors=cc, linewidth=0.8, zorder=3)
        body = np.maximum((sub.close - sub.open).abs().values, rng * 1e-4)
        ax.bar(x, body, bottom=np.minimum(sub.open, sub.close).values, width=0.7,
               color=cc, zorder=3)
        if s["kind"] == "LIQ":
            ax.hlines(s["top"], max(s["x0"], lo), a, colors=LIQ_COL, linestyles="--",
                      linewidth=1.3, zorder=2)
        else:
            zc = COL[(s["kind"], s["dir"])]
            x0 = max(s["x0"], lo)
            ax.add_patch(Rectangle((x0 - 0.5, s["bottom"]), N - x0 + 1.5,
                                   max(s["top"] - s["bottom"], rng * 1e-4),
                                   facecolor=zc, alpha=0.28,
                                   edgecolor="#ffd600" if s["star"] else zc,
                                   linewidth=2.2 if s["star"] else 0.8, zorder=2))
        end = N + 4
        for val, cl, ls, lab in ((s["entry"], col, "-", "Entry"),
                                 (s["stop"], "#d50000", ":", "SL"),
                                 (s["target"], "#00c853", "--", f"Target +{MOVE_PCT:g}%")):
            ax.hlines(val, a, end, colors=cl, linestyles=ls, linewidth=1.3, zorder=4)
            ax.text(end + 0.3, val, f"{lab} {fmt(val)}", fontsize=8, color=cl, va="center")
        ax.scatter([a], [s["entry"]], marker="^" if buy else "v", s=90, color=col, zorder=6)
        ax.set_xlim(lo - 1, N + 14)
        ax.set_ylim(ymin - rng * 0.06, ymax + rng * 0.06)
        step = max(1, len(x) // 12)
        ax.set_xticks(x[::step])
        ax.set_xticklabels(sub.t.iloc[::step].dt.strftime("%d-%m %H:%M"), rotation=30, fontsize=8)
        ax.grid(alpha=0.15)
        name = {"OB": "Order Block", "FVG": "FVG", "LIQ": "Liq Sweep"}[s["kind"]]
        ax.set_title(f"{s['symbol']}  {s['tf']}  |  {'BUY' if buy else 'SELL'}  {name}"
                     f"{' *' if s['star'] else ''}  |  {datetime.now(IST):%d-%m-%Y %H:%M} IST",
                     fontsize=13, weight="bold")
        y = 0.24
        fig.text(0.05, y, "Rule jo follow hua:", fontsize=12, weight="bold")
        for r in s["rules"]:
            y -= 0.034
            fig.text(0.06, y, "- " + r, fontsize=10)
        buf = io.BytesIO()
        with PdfPages(buf) as pdf:
            pdf.savefig(fig)
        plt.close(fig)
    return buf.getvalue()


def check_open(dfs):
    """Alert wale trades ka result: SL pehle = fail, target pehle = success (same trade_outcome rule)."""
    ev, now_ms = [], int(time.time() * 1000)
    for key, t in list(TRADES.items()):
        if now_ms - t["t0"] > (LOOKAHEAD + 5) * TF_MS[t["tf"]]:
            del TRADES[key]
            continue
        df = dfs.get((t["symbol"], t["tf"]))
        if df is None:
            continue
        after = df[df.ot > t["t0"]]
        bull, res = t["dir"] == "bull", None
        for n, (hh, ll) in enumerate(zip(after.high.values, after.low.values)):
            if n >= LOOKAHEAD:
                res = "time"
                break
            if bull:
                if ll <= t["stop"]:
                    res = "sl"
                    break
                if hh >= t["target"]:
                    res = "tp"
                    break
            else:
                if hh >= t["stop"]:
                    res = "sl"
                    break
                if ll <= t["target"]:
                    res = "tp"
                    break
        if not res:
            continue
        del TRADES[key]
        head = f"{t['symbol']} {t['tf']} {'BUY' if bull else 'SELL'} ({t['kind']})"
        txt = {"tp": f"✅ TARGET HIT +{MOVE_PCT:g}% | {head}\nEntry {fmt(t['entry'])} → {fmt(t['target'])}",
               "sl": f"❌ SL HIT | {head}\nEntry {fmt(t['entry'])} → SL {fmt(t['stop'])}",
               "time": f"⏱ {LOOKAHEAD} candles me {MOVE_PCT:g}% move nahi aaya | {head}"}[res]
        ev.append(dict(symbol=t["symbol"], text=txt))
    return ev


def scan_all(symbols):
    jobs = [(s, tf) for s in symbols for tf in TFS]

    def get(j):
        try:
            return j, fetch_last(*j)
        except Exception as e:
            log.warning("%s %s: %s", j[0], j[1], e)
            return j, None

    dfs = {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        for j, df in ex.map(get, jobs):
            if df is not None:
                dfs[j] = df
    events = []
    for (sym, tf), df in dfs.items():
        first = (sym, tf) not in WARM  # pehli baar: purane setups ko seen mark karo, alert nahi
        WARM.add((sym, tf))
        for s in live_signals(sym, tf, df):
            if s["key"] in SEEN:
                continue
            SEEN.add(s["key"])
            if first:
                continue
            TRADES[s["key"]] = s
            events.append(dict(symbol=sym, text=signal_text(s), pdf=live_pdf(df, s),
                               filename=f"{sym}_{tf}_{s['kind']}_{datetime.now(IST):%d%m_%H%M%S}.pdf"))
    events += check_open(dfs)
    return events


async def live_loop(app):
    await asyncio.sleep(2)
    log.info("Live scanner started")
    while True:
        t0 = time.time()
        try:
            watch = sorted({s for v in SUBS.values() for s in v})
            if watch:
                for ev in await asyncio.to_thread(scan_all, watch):
                    for chat in [ch for ch, v in SUBS.items() if ev["symbol"] in v]:
                        try:
                            await app.bot.send_message(int(chat), ev["text"])
                            if ev.get("pdf"):
                                await app.bot.send_document(
                                    int(chat), io.BytesIO(ev["pdf"]), filename=ev["filename"],
                                    caption=f"{ev['symbol']} chart + rules")
                        except Exception:
                            log.exception("send failed")
        except Exception:
            log.exception("live loop")
        await asyncio.sleep(max(1.0, POLL_SEC - (time.time() - t0)))


# ----------------------------- TELEGRAM -------------------------------
HELP = (
    "📊 *Signal Chart Bot v2*\n\n"
    "`/ob btc 23-11-2026` → Working Order Blocks\n"
    "`/fvg eth 23-11-2026` → FVG\n"
    "`/liq sol 23-11-2026` → Liquidity sweeps\n"
    "`/ema xauusdt 23-11-2026` → EMA 100/200\n"
    "`/all btc 23-11-2026` → Sab kuch\n\n"
    "Date IST me (DD-MM-YYYY), na do to aaj ki.\n"
    f"Multiple symbols: `/ob btc,eth,sol 23-11-2026` (max {MAX_SYMBOLS})\n\n"
    f"Sirf wahi setups jo rule follow karke SL se pehle {MOVE_PCT}%+ chale.\n"
    "TF: 5m, 15m, 1h. Output: PDF.\n\n"
    "🔴 *Live alerts*\n"
    f"`/live btc,eth,xau` → naya trade bante hi alert + PDF (max {MAX_LIVE})\n"
    "`/live` → list, `/stoplive` → band"
)


def make_handler(kinds):
    async def handler(update, context):
        msg = update.effective_message
        symbols, day = parse_args(context.args or [])
        if day is None:
            await msg.reply_text("❌ Date galat hai. Format: DD-MM-YYYY (jaise 23-11-2026)")
            return
        if not symbols:
            await msg.reply_text("❌ Symbol do. Jaise: /ob btc 23-11-2026")
            return
        if day > datetime.now(IST):
            await msg.reply_text("❌ Future ki date ka data nahi hota.")
            return
        for sym in symbols[:MAX_SYMBOLS]:
            status = await msg.reply_text(f"⏳ {sym} ({day:%d-%m-%Y} IST) ka chart ban raha hai...")
            try:
                pdf_bytes, n = await asyncio.to_thread(build_report, sym, day, kinds)
                await msg.reply_document(
                    document=io.BytesIO(pdf_bytes),
                    filename=f"{sym}_{day:%d-%m-%Y}_{'-'.join(kinds)}.pdf",
                    caption=f"{sym} | {day:%d-%m-%Y} IST | {n} working setups (>= {MOVE_PCT}% move)",
                )
            except Exception as e:
                log.exception("report failed")
                await msg.reply_text(f"❌ {sym}: {e}")
            finally:
                try:
                    await status.delete()
                except Exception:
                    pass

    return handler


async def start(update, context):
    await update.effective_message.reply_text(HELP, parse_mode="Markdown")


def parse_syms(args):
    return [norm_symbol(t) for t in re.split(r"[,/\s]+", " ".join(args)) if t]


async def live_cmd(update, context):
    msg, chat = update.effective_message, str(update.effective_chat.id)
    syms = parse_syms(context.args or [])
    cur = SUBS.setdefault(chat, [])
    if not syms:
        await msg.reply_text("🔴 Live: " + (", ".join(cur) if cur else "off") +
                             "\nOn karne ke liye: /live btc,eth,xau")
        return
    bad = []
    for sym in syms:
        if sym in cur:
            continue
        try:
            await asyncio.to_thread(fetch_last, sym, "5m", 5)
        except Exception:
            bad.append(sym)
            continue
        if len(cur) >= MAX_LIVE:
            await msg.reply_text(f"❌ Max {MAX_LIVE} symbols.")
            break
        cur.append(sym)
    save_subs()
    txt = "✅ Live ON: " + (", ".join(cur) or "-")
    if bad:
        txt += "\n❌ Symbol nahi mila: " + ", ".join(bad)
    await msg.reply_text(txt + "\nNaya trade bante hi alert + PDF aayega.")


async def stoplive_cmd(update, context):
    msg, chat = update.effective_message, str(update.effective_chat.id)
    syms = parse_syms(context.args or [])
    if syms:
        SUBS[chat] = [s for s in SUBS.get(chat, []) if s not in syms]
    else:
        SUBS[chat] = []
    save_subs()
    await msg.reply_text("🛑 Live: " + (", ".join(SUBS[chat]) or "off"))


async def post_init(app):
    asyncio.create_task(live_loop(app))


def main():
    from telegram.ext import Application, CommandHandler

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if BOT_TOKEN.startswith("PASTE"):
        raise SystemExit("BOT_TOKEN set karo (env var ya file ke upar).")
    load_subs()
    app = Application.builder().token(BOT_TOKEN).post_init(post_init).build()
    app.add_handler(CommandHandler(["start", "help"], start))
    app.add_handler(CommandHandler("ob", make_handler(["ob"])))
    app.add_handler(CommandHandler("fvg", make_handler(["fvg"])))
    app.add_handler(CommandHandler(["liq", "liquidity"], make_handler(["liq"])))
    app.add_handler(CommandHandler("ema", make_handler(["ema"])))
    app.add_handler(CommandHandler("all", make_handler(["ob", "fvg", "liq", "ema"])))
    app.add_handler(CommandHandler("live", live_cmd))
    app.add_handler(CommandHandler("stoplive", stoplive_cmd))
    log.info("Bot started")
    app.run_polling()


if __name__ == "__main__":
    main()
