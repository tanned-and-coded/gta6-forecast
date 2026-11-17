"""Shared data extraction and scoring for local validation.

A comparable's hourly points are bucketed by whole-hour offset from its
day_start. Launch-day hour h (1..24) is offset h-1. Missing offsets are
filled by linear interpolation in log space between the nearest observed
offsets (feature extraction only; scoring always uses raw observed hours).
"""
import json
import math
from datetime import datetime
from pathlib import Path

KIT = Path.home() / "workspace/makefun/kits-gta/gta6-twitch-ccu-forecast-01-kit"
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]


def ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def load_case(name):
    return json.loads((KIT / "input" / f"{name}.in").read_text())


def bucket_hours(points, day_start):
    """{offset_hours_from_day_start: mean viewers} from raw points."""
    start = ts(day_start)
    buckets = {}
    for p in points:
        if p.get("viewers") is None:
            continue
        off = int((ts(p["from"]) - start) // 3600)
        buckets.setdefault(off, []).append(float(p["viewers"]))
    return {o: sum(v) / len(v) for o, v in buckets.items()}


def interp_log(buckets, off):
    """Log viewers at integer offset, log-linear between observed offsets."""
    if off in buckets:
        return math.log(max(1.0, buckets[off]))
    lo = max((o for o in buckets if o < off), default=None)
    hi = min((o for o in buckets if o > off), default=None)
    if lo is None or hi is None:
        return None
    a, b = math.log(max(1.0, buckets[lo])), math.log(max(1.0, buckets[hi]))
    return a + (b - a) * (off - lo) / (hi - lo)


class Comp:
    """One comparable launch with extracted launch-day curve and features."""

    def __init__(self, raw):
        self.name = raw["name"]
        self.platforms = raw.get("platforms") or ""
        self.unlock = raw.get("unlock") or ""
        self.buckets = bucket_hours(raw["hourly"], raw["day_start"])
        # Interpolated log viewers for launch-day hours 1..24.
        self.logv = {}
        for h in range(1, 25):
            v = interp_log(self.buckets, h - 1)
            if v is not None:
                self.logv[h] = v
        # Raw observed launch-day hours, for scoring pseudo-targets.
        self.raw = {h: self.buckets[h - 1] for h in range(1, 25) if (h - 1) in self.buckets}
        # Pre-launch level: median log viewers over offsets -24..-1.
        pre = [interp_log(self.buckets, o) for o in range(-24, 0)]
        pre = [x for x in pre if x is not None]
        self.pre_log = sorted(pre)[len(pre) // 2] if pre else None

    @property
    def usable(self):
        return 1 in self.logv and len(self.logv) >= 20

    def ratios(self):
        """L_h = ln(v_h / v_1) for h=2..24 from interpolated curve."""
        if not self.usable:
            return {}
        base = self.logv[1]
        return {h: self.logv[h] - base for h in range(2, 25) if h in self.logv}


def load_comps():
    case = load_case("p00")
    comps = [Comp(c) for c in case["comparables"]]
    return [c for c in comps if c.usable]


def pinball(y, q, tau):
    d = math.log(max(1.0, y)) - math.log(q)
    return 2.0 * max(tau * d, (tau - 1.0) * d)


def score_forecast(raw_hours, quantiles):
    """Mean pinball over scored hours x levels.

    raw_hours: {h: actual viewers}; quantiles: {h: [q05..q95]}.
    Returns (mean_loss, n_items).
    """
    total, n = 0.0, 0
    for h, y in raw_hours.items():
        if h < 2 or h not in quantiles:
            continue
        for tau, q in zip(LEVELS, quantiles[h]):
            total += pinball(y, q, tau)
            n += 1
    return (total / n if n else float("nan")), n


def baseline_quantiles(p1):
    p = max(1.0, p1)
    return {h: [p * math.exp(z * 0.15 * math.sqrt(h - 1)) for z in Z] for h in range(2, 25)}
