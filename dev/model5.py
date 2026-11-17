"""Final tuning pass on the v4 design: rolling-pool exclusion, sd scale,
cold threshold, and a pre-peak (last 48 h) weight dimension."""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import common
from common import load_comps, score_forecast
from model2 import CompV2
from model3 import empirical_u
from model4 import pooled_z, shifted, w_v1


def pre_peak48(comp):
    vals = [common.interp_log(comp.buckets, o) for o in range(-48, 0)]
    vals = [v for v in vals if v is not None]
    return max(vals) if vals else None


def wrap(comps):
    out = []
    for c in comps:
        v = CompV2(c)
        out.append({"c": c, "kind": v.kind, "u": v.u, "u_emp": empirical_u(c),
                    "logv": c.logv, "logv1": v.logv1, "ratios": c.ratios(),
                    "lift": v.lift, "pre_peak": pre_peak48(c)})
    return out


def model(pool, target, bw_v1=1.2, cold_logv1=math.log(20000), mix=0.35,
          ugrid=(5, 9, 13, 17, 21), rolling_excl_global=False, sd_scale=1.0,
          bw_peak=None):
    def wgt(c):
        w = w_v1(c, target, bw_v1)
        if bw_peak and c["pre_peak"] is not None and target["pre_peak"] is not None:
            d = c["pre_peak"] - target["pre_peak"]
            w *= math.exp(-(d * d) / (2 * bw_peak * bw_peak))
        return w

    base = [c for c in pool if not (rolling_excl_global and target["kind"] == "rolling"
                                    and c["kind"] == "global")]
    curves = [c["ratios"] for c in base]
    ws = [wgt(c) for c in base]
    extra_curves, extra_ws = [], []
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
                extra_ws.append(wgt(c) * mw)
    elif target["kind"] != "rolling" and target["logv1"] < cold_logv1:
        for c in pool:
            if c["u_emp"] is None:
                continue
            for u_t in ugrid:
                extra_curves.append(shifted(c["logv"], c["logv1"], u_t - c["u_emp"]))
                extra_ws.append(wgt(c) / len(ugrid))
    if extra_curves:
        base_total, extra_total = sum(ws), sum(extra_ws)
        scale = (mix / (1 - mix)) * base_total / extra_total if extra_total > 0 else 0.0
        curves += extra_curves
        ws += [w * scale for w in extra_ws]
    q = pooled_z(curves, ws, target["logp1"])
    if sd_scale != 1.0:
        lp = target["logp1"]
        q = {h: [math.exp(lp + (math.log(x) - lp) * sd_scale) for x in row]
             for h, row in q.items()}
    return q


def run(comps, label, **kw):
    W = wrap(comps)
    total, items = 0.0, 0
    by_kind = {}
    for t in W:
        pool = [c for c in W if c is not t]
        target = {"logp1": math.log(max(1.0, t["c"].raw[1])), "lift": t["lift"],
                  "kind": t["kind"], "u": t["u"], "logv1": t["logv1"],
                  "pre_peak": t["pre_peak"]}
        q = model(pool, target, **kw)
        loss, n = score_forecast(t["c"].raw, q)
        by_kind.setdefault(t["kind"], []).append(loss)
        total += loss * n
        items += n
    kinds = " ".join(f"{k}={sum(v)/len(v):.3f}(n{len(v)})" for k, v in sorted(by_kind.items()))
    print(f"{label:44s} LOO={total / items:.4f}  [{kinds}]")
    return total / items


def main():
    comps = load_comps()
    run(comps, "v5 base (mix.35 cold20k)")
    run(comps, "v5 rolling_excl_global", rolling_excl_global=True)
    run(comps, "v5 sd_scale=0.92", sd_scale=0.92)
    run(comps, "v5 sd_scale=1.08", sd_scale=1.08)
    run(comps, "v5 bw_peak=1.5", bw_peak=1.5)
    run(comps, "v5 cold=15000", cold_logv1=math.log(15000))
    run(comps, "v5 cold=25000", cold_logv1=math.log(25000))
    run(comps, "v5 mix=0.5", mix=0.5)


if __name__ == "__main__":
    main()
