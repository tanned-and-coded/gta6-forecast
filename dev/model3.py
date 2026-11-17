"""Model v3: hybrid pools.

- Global-unlock target (unlock time parsed from text): pool = other
  global-unlock comps plus 'surge' comps (unknown class whose own curve
  jumps by > e^1.5 from hour 1 to peak), every curve time-shifted so its
  unlock hour (text-parsed, else empirical: first hour reaching 60% of
  the log climb to peak) aligns with the target's.
- Any other target: all comps, unshifted.
Weights: Gaussian in log hour-1 viewers (bw_v1). Quantiles: pooled
standardized residuals (pooled-z) over the weighted pool.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import LEVELS, load_comps, score_forecast
from loo import quantile, wmean_sd
from model2 import CompV2, feat_weights


def empirical_u(comp):
    lv = comp.logv
    if 1 not in lv:
        return None
    peak = max(lv.values())
    climb = peak - lv[1]
    if climb < 1.5:
        return None
    for h in range(2, 25):
        if h in lv and lv[h] >= lv[1] + 0.6 * climb:
            return float(h)
    return None


def shifted(curve_logv, base_logv1, shift):
    out = {}
    for h in range(2, 25):
        src = h - shift
        lo, hi = int(math.floor(src)), int(math.ceil(src))
        if lo in curve_logv and hi in curve_logv:
            frac = src - lo
            out[h] = curve_logv[lo] * (1 - frac) + curve_logv[hi] * frac - base_logv1
        elif src in curve_logv:
            out[h] = curve_logv[src] - base_logv1
    return out


def pooled_z(curves, ws, logp1):
    mu, sd = {}, {}
    for h in range(2, 25):
        pairs = [(r[h], w) for r, w in zip(curves, ws) if h in r]
        if not pairs:
            mu[h], sd[h] = 0.0, 0.15 * math.sqrt(h - 1)
            continue
        mu[h], sd[h] = wmean_sd(pairs)
        sd[h] = max(sd[h], 0.05)
    resid = []
    for r in curves:
        for h, v in r.items():
            resid.append((v - mu[h]) / sd[h])
    resid.sort()
    zs = [quantile(resid, t) for t in LEVELS]
    return {h: [math.exp(logp1 + mu[h] + z * sd[h]) for z in zs] for h in range(2, 25)}


def model_v3(pool, target, bw_v1=1.2, blend=0.0):
    """pool/target are CompV2-based dicts (see run())."""
    if target["kind"] == "global" and target["u"]:
        sel = []
        for c in pool:
            if c["kind"] == "global" and c["u"]:
                sel.append((c, target["u"] - c["u"]))
            elif c["kind"] == "unknown" and c["u_emp"] is not None:
                sel.append((c, target["u"] - c["u_emp"]))
        if len(sel) >= 3:
            curves = [shifted(c["logv"], c["logv1"], s) for c, s in sel]
            ws = feat_weights([c["asv2"] for c, _ in sel], target, bw_v1=bw_v1)
            q_aligned = pooled_z(curves, ws, target["logp1"])
            if blend <= 0:
                return q_aligned
            curves_all = [c["ratios"] for c in pool]
            ws_all = feat_weights([c["asv2"] for c in pool], target, bw_v1=bw_v1)
            q_all = pooled_z(curves_all, ws_all, target["logp1"])
            return {h: [math.exp((1 - blend) * math.log(a) + blend * math.log(b))
                        for a, b in zip(q_aligned[h], q_all[h])] for h in range(2, 25)}
    curves = [c["ratios"] for c in pool]
    ws = feat_weights([c["asv2"] for c in pool], target, bw_v1=bw_v1)
    return pooled_z(curves, ws, target["logp1"])


def wrap(comps):
    out = []
    for c in comps:
        v = CompV2(c)
        out.append({"c": c, "asv2": v, "kind": v.kind, "u": v.u,
                    "u_emp": empirical_u(c), "logv": c.logv, "logv1": v.logv1,
                    "ratios": c.ratios(), "lift": v.lift})
    return out


def run(comps, label, **kw):
    W = wrap(comps)
    total, items = 0.0, 0
    per, by_kind = {}, {}
    for t in W:
        pool = [c for c in W if c is not t]
        target = {"logp1": math.log(max(1.0, t["c"].raw[1])), "lift": t["lift"],
                  "kind": t["kind"], "u": t["u"], "logv1": t["logv1"]}
        q = model_v3(pool, target, **kw)
        loss, n = score_forecast(t["c"].raw, q)
        per[t["c"].name] = loss
        by_kind.setdefault(t["kind"], []).append(loss)
        total += loss * n
        items += n
    kinds = " ".join(f"{k}={sum(v)/len(v):.3f}(n{len(v)})" for k, v in sorted(by_kind.items()))
    print(f"{label:40s} LOO={total / items:.4f}  [{kinds}]")
    return total / items, per


def main():
    comps = load_comps()
    run(comps, "v3 hybrid bw=1.2")
    run(comps, "v3 hybrid bw=1.0", bw_v1=1.0)
    run(comps, "v3 hybrid bw=1.5", bw_v1=1.5)
    run(comps, "v3 hybrid bw=1.2 blend=0.3", blend=0.3)
    _, per = run(comps, "v3 hybrid bw=1.2 (per-case)", bw_v1=1.2)
    for name, loss in sorted(per.items(), key=lambda kv: -kv[1])[:10]:
        print(f"    {name[:44]:46s} {loss:.3f}")


if __name__ == "__main__":
    main()
