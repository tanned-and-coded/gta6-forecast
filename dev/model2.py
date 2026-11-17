"""Model v2: unlock-aware pools, time-shift alignment for global unlocks,
feature-similarity weights, pooled standardized quantiles.

Unlock parsing works on free text (a comparable's `unlock` field, or the
target case's `notes`): returns ('rolling', None), ('global', u) with u
the unlock hour in day hours (1 = day_start), or ('unknown', None).
"""
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import LEVELS, Z, load_comps, score_forecast
from loo import quantile, wmean_sd

TS_RE = re.compile(r"T(\d{2}):(\d{2})(?::\d{2})?Z")
TS_RANGE_RE = re.compile(r"T(\d{2}):(\d{2})-\d{2}:\d{2}Z")
GLOBAL_PHRASES = ("single global", "simultaneous global", "single simultaneous",
                  "global unlock", "simultaneous across", "one global")


def _unlock_hour(text):
    m = TS_RE.search(text or "") or TS_RANGE_RE.search(text or "")
    if not m:
        return None
    return 1 + int(m.group(1)) + int(m.group(2)) / 60.0


def parse_unlock(text, day_start_ts=None):
    t = (text or "").lower()
    # Consoles at midnight local dominate a launch's Twitch curve even when
    # a PC port unlocks at one global time.
    if "midnight local" in t or "rolling" in t:
        return ("rolling", None)
    if any(p in t for p in GLOBAL_PHRASES):
        u = _unlock_hour(text or "")
        if u is not None:
            return ("global", u) if u > 2.5 else ("rolling", None)
        return ("global", None)
    return ("unknown", None)


class CompV2:
    def __init__(self, comp):
        self.c = comp
        self.kind, self.u = parse_unlock(comp.unlock)
        self.logv1 = comp.logv[1]
        self.lift = comp.logv[1] - comp.pre_log if comp.pre_log is not None else None

    def shifted_ratios(self, shift):
        """Log-ratio curve with time shifted by `shift` hours (target - comp
        unlock offset): value at target hour h is comp's ln v at hour
        h - shift, minus comp's ln v_1."""
        out = {}
        base = self.c.logv[1]
        for h in range(2, 25):
            src = h - shift
            lo, hi = int(math.floor(src)), int(math.ceil(src))
            if lo in self.c.logv and hi in self.c.logv:
                frac = src - lo
                v = self.c.logv[lo] * (1 - frac) + self.c.logv[hi] * frac
                out[h] = v - base
            elif src in self.c.logv:
                out[h] = self.c.logv[src] - base
        return out


def feat_weights(pool, target, bw_v1=None, bw_lift=None):
    ws = []
    for c in pool:
        w = 1.0
        if bw_v1:
            d = c.logv1 - target["logv1"]
            w *= math.exp(-(d * d) / (2 * bw_v1 * bw_v1))
        if bw_lift and c.lift is not None and target["lift"] is not None:
            d = c.lift - target["lift"]
            w *= math.exp(-(d * d) / (2 * bw_lift * bw_lift))
        ws.append(w)
    return ws


def pooled_quantiles(curves, ws, logp1, z_mode="pooled"):
    """curves: list of {h: L_h}; returns {h: [quantiles in viewers]}."""
    mu, sd = {}, {}
    for h in range(2, 25):
        pairs = [(r[h], w) for r, w in zip(curves, ws) if h in r]
        if not pairs:
            mu[h], sd[h] = 0.0, 0.15 * math.sqrt(h - 1)
            continue
        mu[h], sd[h] = wmean_sd(pairs)
        if sd[h] < 0.05:
            sd[h] = 0.05
    if z_mode == "pooled":
        resid = []
        for r, w in zip(curves, ws):
            for h, v in r.items():
                resid.append((v - mu[h]) / sd[h])
        resid.sort()
        zs = [quantile(resid, t) for t in LEVELS]
    else:
        zs = Z
    return {h: [math.exp(logp1 + mu[h] + z * sd[h]) for z in zs] for h in range(2, 25)}


def model_v2(pool_comps, target, pool_mode="auto", bw_v1=None, bw_lift=None,
             z_mode="pooled"):
    """pool_comps: [CompV2]; target: {logv1, lift, kind, u}."""
    kind = target["kind"]
    if pool_mode == "auto":
        if kind == "rolling":
            sel = [c for c in pool_comps if c.kind != "global"]
        elif kind == "global":
            sel = [c for c in pool_comps if c.kind == "global"]
            if len(sel) < 3:
                sel = pool_comps
        else:
            sel = pool_comps
    elif pool_mode == "all":
        sel = pool_comps
    elif pool_mode == "noglobal_for_rolling":
        sel = [c for c in pool_comps if c.kind != "global"] if kind == "rolling" else pool_comps
    else:
        sel = pool_comps
    curves, kept = [], []
    for c in sel:
        shift = 0
        if kind == "global" and target["u"] and c.kind == "global" and c.u:
            shift = int(round(target["u"] - c.u))
        curves.append(c.shifted_ratios(shift))
        kept.append(c)
    ws = feat_weights(kept, target, bw_v1, bw_lift)
    return pooled_quantiles(curves, ws, target["logp1"], z_mode)


def run(comps, label, **kw):
    v2 = [CompV2(c) for c in comps]
    total, items = 0.0, 0
    per = {}
    for t, tv in zip(comps, v2):
        pool = [c for c in v2 if c is not tv]
        target = {"logp1": math.log(max(1.0, t.raw[1])), "lift": tv.lift,
                  "kind": tv.kind, "u": tv.u, "logv1": tv.logv1}
        q = model_v2(pool, target, **kw)
        loss, n = score_forecast(t.raw, q)
        per[t.name] = loss
        total += loss * n
        items += n
    print(f"{label:44s} LOO={total / items:.4f}")
    return total / items, per


def main():
    comps = load_comps()
    v2 = [CompV2(c) for c in comps]
    from collections import Counter
    print("unlock classes:", Counter(c.kind for c in v2),
          "| global u:", {c.c.name[:24]: c.u for c in v2 if c.kind == "global"})
    run(comps, "v2 auto pools pooled-z")
    run(comps, "v2 all + bw_v1=1.5", pool_mode="all", bw_v1=1.5)
    run(comps, "v2 all + bw_v1=1.2", pool_mode="all", bw_v1=1.2)
    run(comps, "v2 all + bw_v1=2.0", pool_mode="all", bw_v1=2.0)
    run(comps, "v2 auto + bw_v1=1.5", bw_v1=1.5)
    run(comps, "v2 noglobal_for_rolling + bw_v1=1.5", pool_mode="noglobal_for_rolling", bw_v1=1.5)
    run(comps, "v2 all + bw_v1=1.5 + bw_lift=1.2", pool_mode="all", bw_v1=1.5, bw_lift=1.2)
    _, per = run(comps, "v2 all + bw_v1=1.5 (per-case)", pool_mode="all", bw_v1=1.5)
    for name, loss in sorted(per.items(), key=lambda kv: -kv[1])[:10]:
        print(f"    {name[:44]:46s} {loss:.3f}")


if __name__ == "__main__":
    main()
