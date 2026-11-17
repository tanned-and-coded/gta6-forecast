"""Forecast a game launch day's hourly Twitch viewers, hours 2..24.

Entry for make.fun's gta6-twitch-ccu-forecast-01 (program/v1, Python
3.13, standard library only, no network). Run once per case: reads every
*.in file in $MAKEFUN_INPUT_DIR (else /input) in name order and writes
<id>.out to $MAKEFUN_OUTPUT_DIR (else /output) with five quantiles
(0.05, 0.25, 0.5, 0.75, 0.95) of each hour's concurrent viewers.

Model (tuned by leave-one-out cross-validation over the practice cases'
comparables; see README.md and dev/):

  For every comparable launch, take its launch-day hourly viewers,
  log-interpolate any gaps, and form log-ratios L_h = ln(v_h / v_1).
  The forecast for hour h is p1 * exp(mu_h + z_tau * sd_h), where mu_h
  and sd_h are the weighted mean and standard deviation of the pool's
  L_h, and z_tau are the empirical quantiles of the pool's standardized
  residuals pooled over all hours (pooled-z). Pool weights are Gaussian
  in log hour-1 viewers (bandwidth 1.2), so launches of similar scale
  dominate. Three pool adjustments, all driven by the case's own text
  and data:

  - Rolling-midnight target (its notes say so): global-unlock and
    late-surge comparables leave the pool; their shapes cannot occur.
  - Global-unlock target with a parsed unlock hour u: aligned copies
    of global/surge curves, time-shifted so their unlock hour (parsed
    or, failing that, the hour their curve reaches 60% of its log climb
    to peak) lands on u (and u +/- 3 at half weight), are mixed in at
    35% of the pool's weight.
  - Cold target (small hour-1 print, unlock pattern unknown): the same
    shifted copies are added over a grid of candidate unlock hours, so
    unlock timing is integrated out instead of guessed.

  Any failure or shortage of data falls back to the house baseline's
  formula (flat at hour 1, normal spread 0.15 * sqrt(h - 1)); a case
  never goes unwritten or invalid.
"""

import json
import math
import os
import re
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z_NORMAL = [-1.645, -0.674, 0.0, 0.674, 1.645]

BW_V1 = 1.2                    # Gaussian bandwidth on log hour-1 viewers
COLD_LOGV1 = math.log(20000)   # below this an unknown target may be pre-unlock
MIX = 0.35                     # pool weight share of shifted copies
UGRID = (5, 9, 13, 17, 21)     # candidate unlock hours for cold targets
MIN_POOL = 5                   # below this, use the baseline formula

TS_RE = re.compile(r"T(\d{2}):(\d{2})(?::\d{2})?Z")
TS_RANGE_RE = re.compile(r"T(\d{2}):(\d{2})-\d{2}:\d{2}Z")
GLOBAL_PHRASES = ("single global", "simultaneous global", "single simultaneous",
                  "global unlock", "simultaneous across", "one global")


def ts(s):
    return datetime.fromisoformat(str(s).replace("Z", "+00:00")).timestamp()


def parse_unlock(text):
    """Classify free unlock text: ('rolling', None) | ('global', hour) |
    ('global', None) | ('unknown', None). Hour is in day hours (1 = the
    day's first hour). Consoles at midnight local dominate a launch's
    Twitch curve even when a PC port unlocks at one global time."""
    t = (text or "").lower()
    if "midnight local" in t or "rolling" in t:
        return ("rolling", None)
    if any(p in t for p in GLOBAL_PHRASES):
        m = TS_RE.search(text or "") or TS_RANGE_RE.search(text or "")
        if m:
            u = 1 + int(m.group(1)) + int(m.group(2)) / 60.0
            return ("global", u) if u > 2.5 else ("rolling", None)
        return ("global", None)
    return ("unknown", None)


def bucket_hours(points, day_start):
    start = ts(day_start)
    buckets = {}
    for p in points or []:
        try:
            v = p.get("viewers")
            if v is None:
                continue
            off = int((ts(p["from"]) - start) // 3600)
        except Exception:
            continue
        buckets.setdefault(off, []).append(float(v))
    return {o: sum(vs) / len(vs) for o, vs in buckets.items()}


def interp_log(buckets, off):
    if off in buckets:
        return math.log(max(1.0, buckets[off]))
    lo = max((o for o in buckets if o < off), default=None)
    hi = min((o for o in buckets if o > off), default=None)
    if lo is None or hi is None:
        return None
    a, b = math.log(max(1.0, buckets[lo])), math.log(max(1.0, buckets[hi]))
    return a + (b - a) * (off - lo) / (hi - lo)


class Comp:
    """A comparable launch: launch-day log curve, ratios, unlock class,
    and its empirical surge hour (first hour reaching 60% of the log
    climb from hour 1 to peak, when that climb exceeds 1.5)."""

    def __init__(self, raw):
        self.buckets = bucket_hours(raw.get("hourly"), raw.get("day_start"))
        self.logv = {}
        for h in range(1, 25):
            v = interp_log(self.buckets, h - 1)
            if v is not None:
                self.logv[h] = v
        self.kind, self.u = parse_unlock(raw.get("unlock") or "")
        self.u_emp = None
        if 1 in self.logv and len(self.logv) >= 20:
            peak = max(self.logv.values())
            climb = peak - self.logv[1]
            if climb >= 1.5:
                for h in range(2, 25):
                    if h in self.logv and self.logv[h] >= self.logv[1] + 0.6 * climb:
                        self.u_emp = float(h)
                        break

    @property
    def usable(self):
        return 1 in self.logv and len(self.logv) >= 20

    @property
    def logv1(self):
        return self.logv[1]

    def ratios(self):
        base = self.logv[1]
        return {h: self.logv[h] - base for h in range(2, 25) if h in self.logv}

    def shifted(self, shift):
        out = {}
        base = self.logv[1]
        for h in range(2, 25):
            src = h - shift
            lo, hi = int(math.floor(src)), int(math.ceil(src))
            if lo in self.logv and hi in self.logv:
                frac = src - lo
                out[h] = self.logv[lo] * (1 - frac) + self.logv[hi] * frac - base
            elif src in self.logv:
                out[h] = self.logv[src] - base
        return out


def quantile(sorted_vals, tau):
    n = len(sorted_vals)
    if n == 1:
        return sorted_vals[0]
    pos = tau * (n - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, n - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


def round2(x):
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if float(value).is_integer() else value


def baseline_quantiles(p1, hours):
    p = max(1.0, p1)
    return {h: [p * math.exp(z * 0.15 * math.sqrt(h - 1)) for z in Z_NORMAL] for h in hours}


def pooled_z(curves, ws, logp1):
    mu, sd = {}, {}
    for h in range(2, 25):
        pairs = [(r[h], w) for r, w in zip(curves, ws) if h in r]
        if not pairs:
            mu[h], sd[h] = 0.0, 0.15 * math.sqrt(h - 1)
            continue
        sw = sum(w for _, w in pairs)
        m = sum(v * w for v, w in pairs) / sw
        var = sum(w * (v - m) ** 2 for v, w in pairs) / sw
        mu[h], sd[h] = m, max(math.sqrt(max(var, 0.0)), 0.05)
    resid = []
    for r in curves:
        for h, v in r.items():
            resid.append((v - mu[h]) / sd[h])
    resid.sort()
    zs = [quantile(resid, t) for t in LEVELS] if resid else Z_NORMAL
    return {h: [math.exp(logp1 + mu[h] + z * sd[h]) for z in zs] for h in range(2, 25)}


def model_quantiles(case, p1):
    """The comparable-pool model; raises on any data problem so the
    caller can fall back to the baseline."""
    comps = []
    for raw in case.get("comparables") or []:
        try:
            c = Comp(raw)
        except Exception:
            continue
        if c.usable:
            comps.append(c)
    if len(comps) < MIN_POOL:
        raise ValueError("too few usable comparables")
    logp1 = math.log(max(1.0, p1))
    logv1_t = logp1
    kind_t, u_t = parse_unlock(case.get("notes") or "")

    def w_v1(c):
        d = c.logv1 - logv1_t
        return math.exp(-(d * d) / (2 * BW_V1 * BW_V1))

    if kind_t == "rolling":
        pool = [c for c in comps
                if c.kind != "global" and not (c.kind == "unknown" and c.u_emp is not None)]
        if len(pool) < MIN_POOL:
            pool = comps
    else:
        pool = comps
    curves = [c.ratios() for c in pool]
    ws = [w_v1(c) for c in pool]

    extra_curves, extra_ws = [], []
    if kind_t == "global" and u_t:
        for c in comps:
            if c.kind == "global" and c.u:
                base_u = c.u
            elif c.u_emp is not None:
                base_u = c.u_emp
            else:
                continue
            for du, mw in ((0, 1.0), (-3, 0.5), (3, 0.5)):
                extra_curves.append(c.shifted(u_t + du - base_u))
                extra_ws.append(w_v1(c) * mw)
    elif kind_t != "rolling" and logv1_t < COLD_LOGV1:
        for c in comps:
            if c.u_emp is None:
                continue
            for u_c in UGRID:
                extra_curves.append(c.shifted(u_c - c.u_emp))
                extra_ws.append(w_v1(c) / len(UGRID))
    if extra_curves:
        base_total, extra_total = sum(ws), sum(extra_ws)
        if extra_total > 0:
            scale = (MIX / (1 - MIX)) * base_total / extra_total
            curves += extra_curves
            ws += [w * scale for w in extra_ws]
    return pooled_z(curves, ws, logp1)


def clean_row(vals):
    """Clamp to (0, 1e9], force non-decreasing, round like the starter."""
    out = []
    prev = 0.0
    for v in vals:
        if not math.isfinite(v):
            v = prev if prev > 0 else 1.0
        v = min(max(v, 0.01), 1e9)
        v = max(v, prev)
        out.append(round2(v))
        prev = out[-1]
    return out


def forecast_case(case):
    known = case.get("known") or []
    if isinstance(known, dict):
        known = [known]
    first = next((k for k in known if k.get("hour") == 1), known[0] if known else None)
    p1 = 1.0
    if first is not None:
        try:
            p1 = float(first.get("viewers") or 1.0)
        except Exception:
            p1 = 1.0
    hours = []
    for row in case.get("forecast") or []:
        try:
            hours.append(int(row["hour"]))
        except Exception:
            continue
    if not hours:
        hours = list(range(2, 25))
    # The contest's output shape is fixed: five quantiles at LEVELS for
    # each of hours 2..24, whatever the case file claims.
    try:
        quant = model_quantiles(case, p1)
    except Exception:
        quant = baseline_quantiles(p1, hours)
    rows = []
    for h in sorted(set(hours)):
        vals = quant.get(h) or baseline_quantiles(p1, [h])[h]
        rows.append({"hour": h, "q": clean_row([float(v) for v in vals])})
    return {"hours": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        try:
            case = json.loads(path.read_text(encoding="utf-8"))
            forecast = forecast_case(case)
        except Exception:
            # A malformed case file still gets a valid, flat forecast.
            forecast = {"hours": [{"hour": h, "q": [1, 1, 1, 1, 1]} for h in range(2, 25)]}
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
