"""Model v4: unified weighted curve pool.

Base: every pool comparable's unshifted log-ratio curve, weighted by a
Gaussian in log hour-1 viewers. Additions:
- global-unlock target with parsed unlock hour u: aligned copies of
  global/surge curves shifted to u (and u +/- 3), mixed with the base
  pool at weight `mix`.
- 'cold' target (log v1 below threshold, unlock not rolling): surge
  curves also shifted to a grid of candidate unlock hours, so unlock
  timing is a latent variable integrated out of the pool.
Quantiles come from pooled standardized residuals over the final pool.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import LEVELS, load_comps, score_forecast
from loo import quantile, wmean_sd
from model2 import CompV2
from model3 import empirical_u, shifted


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


def w_v1(c, target, bw):
    d = c["logv1"] - target["logv1"]
    return math.exp(-(d * d) / (2 * bw * bw))


def model_v4(pool, target, bw_v1=1.2, cold_logv1=math.log(30000), mix=0.5,
             ugrid=(5, 9, 13, 17, 21)):
    curves, ws = [], []
    for c in pool:
        curves.append(c["ratios"])
        ws.append(w_v1(c, target, bw_v1))
    extra_curves, extra_ws = [], []
    surge = [c for c in pool if c["u_emp"] is not None]
    if target["kind"] == "global" and target["u"]:
        for c in pool:
            if c["kind"] == "global" and c["u"]:
                base_u = c["u"]
            elif c["u_emp"] is not None:
                base_u = c["u_emp"]
            else:
                continue
            for du, mw in ((0, 1.0), (-3, 0.5), (3, 0.5)):
                extra_curves.append(shifted(c["logv"], c["logv1"], target["u"] + du - base_u))
                extra_ws.append(w_v1(c, target, bw_v1) * mw)
        if extra_curves:
            base_total = sum(ws)
            extra_total = sum(extra_ws)
            scale = (mix / (1 - mix)) * base_total / extra_total if extra_total > 0 else 0.0
            curves += extra_curves
            ws += [w * scale for w in extra_ws]
    elif target["kind"] != "rolling" and target["logv1"] < cold_logv1:
        for c in surge:
            for u_t in ugrid:
                extra_curves.append(shifted(c["logv"], c["logv1"], u_t - c["u_emp"]))
                extra_ws.append(w_v1(c, target, bw_v1) / len(ugrid))
        if extra_curves:
            base_total = sum(ws)
            extra_total = sum(extra_ws)
            scale = (mix / (1 - mix)) * base_total / extra_total if extra_total > 0 else 0.0
            curves += extra_curves
            ws += [w * scale for w in extra_ws]
    return pooled_z(curves, ws, target["logp1"])


def wrap(comps):
    out = []
    for c in comps:
        v = CompV2(c)
        out.append({"c": c, "kind": v.kind, "u": v.u, "u_emp": empirical_u(c),
                    "logv": c.logv, "logv1": v.logv1, "ratios": c.ratios(),
                    "lift": v.lift})
    return out


def run(comps, label, **kw):
    W = wrap(comps)
    total, items = 0.0, 0
    per, by_kind = {}, {}
    for t in W:
        pool = [c for c in W if c is not t]
        target = {"logp1": math.log(max(1.0, t["c"].raw[1])), "lift": t["lift"],
                  "kind": t["kind"], "u": t["u"], "logv1": t["logv1"]}
        q = model_v4(pool, target, **kw)
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
    run(comps, "v4 default mix=0.5")
    run(comps, "v4 mix=0.35", mix=0.35)
    run(comps, "v4 mix=0.65", mix=0.65)
    run(comps, "v4 cold=20000", cold_logv1=math.log(20000))
    run(comps, "v4 cold=50000", cold_logv1=math.log(50000))
    _, per = run(comps, "v4 default (per-case)")
    for name, loss in sorted(per.items(), key=lambda kv: -kv[1])[:10]:
        print(f"    {name[:44]:46s} {loss:.3f}")


if __name__ == "__main__":
    main()
