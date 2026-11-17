"""Leave-one-out cross-validation over the pooled comparables.

Each usable comparable in turn is the pseudo-target: its launch-day hour-1
(raw bucket) is the known print, the other launches form its pool. Models
return {hour: [q05..q95]}; scoring is the contest's mean pinball loss on
the log scale against the pseudo-target's raw observed hours 2..24.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import LEVELS, Z, baseline_quantiles, load_comps, score_forecast


def quantile(sorted_vals, tau):
    """Type-7 empirical quantile of pre-sorted values."""
    n = len(sorted_vals)
    if n == 1:
        return sorted_vals[0]
    pos = tau * (n - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, n - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


def wmean_sd(pairs):
    """pairs: [(value, weight)] -> (weighted mean, weighted sd)."""
    sw = sum(w for _, w in pairs)
    if sw <= 0:
        return 0.0, 0.0
    mu = sum(v * w for v, w in pairs) / sw
    var = sum(w * (v - mu) ** 2 for v, w in pairs) / sw
    return mu, math.sqrt(max(var, 0.0))


def smooth(series, win=1):
    """series: {h: x}; centered moving average with half-width win."""
    out = {}
    for h in series:
        vals = [series[k] for k in range(h - win, h + win + 1) if k in series]
        out[h] = sum(vals) / len(vals)
    return out


class Model:
    """Per-hour normal location/scale on pool log-ratios, with options.

    weight: 'uniform' or ('lift', bandwidth) — weight pool comps by how
    close their hour-1 lift over their pre-launch level is to the target's.
    mu_shrink / sd_shrink: pseudo-count pulling mu to 0 / sd to the house
    baseline's sd at that hour. sd_floor: added in quadrature.
    smooth_win: moving-average half-width for mu and sd across hours.
    z_mode: 'normal' uses fixed normal quantiles; 'pooled' uses empirical
    quantiles of pooled standardized residuals.
    """

    def __init__(self, weight="uniform", mu_shrink=0.0, sd_shrink=0.0,
                 sd_floor=0.0, smooth_win=0, z_mode="normal", mu_scale=1.0,
                 sd_scale=1.0):
        self.weight = weight
        self.mu_shrink = mu_shrink
        self.sd_shrink = sd_shrink
        self.sd_floor = sd_floor
        self.smooth_win = smooth_win
        self.z_mode = z_mode
        self.mu_scale = mu_scale
        self.sd_scale = sd_scale

    def weights(self, pool, target):
        if self.weight == "uniform":
            return [1.0] * len(pool)
        _, bw = self.weight
        t_lift = target["lift"]
        out = []
        for c in pool:
            if c.pre_log is None or t_lift is None:
                out.append(1.0)
            else:
                d = (c.logv[1] - c.pre_log) - t_lift
                out.append(math.exp(-(d * d) / (2 * bw * bw)))
        return out

    def fit_quantiles(self, pool, target):
        ws = self.weights(pool, target)
        ratios = [c.ratios() for c in pool]
        mu, sd = {}, {}
        for h in range(2, 25):
            pairs = [(r[h], w) for r, w in zip(ratios, ws) if h in r]
            if not pairs:
                mu[h], sd[h] = 0.0, 0.15 * math.sqrt(h - 1)
                continue
            m, s = wmean_sd(pairs)
            n = sum(w for _, w in pairs)
            if self.mu_shrink:
                m *= n / (n + self.mu_shrink)
            if self.sd_shrink:
                sbase = 0.15 * math.sqrt(h - 1)
                s = math.sqrt((n * s * s + self.sd_shrink * sbase * sbase) / (n + self.sd_shrink))
            mu[h], sd[h] = m, s
        if self.smooth_win:
            mu = smooth(mu, self.smooth_win)
            sd = smooth(sd, self.smooth_win)
        if self.z_mode == "pooled":
            resid = []
            for r, w in zip(ratios, ws):
                for h, v in r.items():
                    if sd[h] > 1e-9:
                        resid.append((v - mu[h]) / sd[h])
            resid.sort()
            zs = [quantile(resid, t) for t in LEVELS] if resid else Z
        else:
            zs = Z
        logp1 = target["logp1"]
        out = {}
        for h in range(2, 25):
            s = math.sqrt((sd[h] * self.sd_scale) ** 2 + self.sd_floor ** 2)
            m = mu[h] * self.mu_scale
            out[h] = [math.exp(logp1 + m + z * s) for z in zs]
        return out


def empirical_model(pool, target, weight="uniform"):
    """Direct per-hour weighted empirical quantiles of pool log-ratios."""
    m = Model(weight=weight)
    ws = m.weights(pool, target)
    ratios = [c.ratios() for c in pool]
    logp1 = target["logp1"]
    out = {}
    for h in range(2, 25):
        vals = sorted(r[h] for r in ratios if h in r)
        if not vals:
            out[h] = [math.exp(logp1)] * 5
            continue
        # Weighted empirical quantile via sorted pool (uniform within hour).
        out[h] = [math.exp(logp1 + quantile(vals, t)) for t in LEVELS]
    return out


def run_loo(comps, model_fn, label):
    total, items, per_case = 0.0, 0, {}
    for t in comps:
        pool = [c for c in comps if c is not t]
        target = {"logp1": math.log(max(1.0, t.raw[1])),
                  "lift": (t.logv[1] - t.pre_log) if t.pre_log is not None else None}
        q = model_fn(pool, target)
        loss, n = score_forecast(t.raw, q)
        per_case[t.name] = loss
        total += loss * n
        items += n
    mean = total / items
    print(f"{label:44s} LOO={mean:.4f}  (items={items})")
    return mean, per_case


def main():
    comps = load_comps()
    print(f"{len(comps)} usable comparables")
    results = {}
    results["baseline"], base_pc = run_loo(comps, lambda pool, t: baseline_quantiles(math.exp(t["logp1"])), "house baseline")
    variants = [
        ("normal uniform", Model()),
        ("normal uniform smooth1", Model(smooth_win=1)),
        ("normal mu_shrink4", Model(mu_shrink=4)),
        ("normal sd_shrink4", Model(sd_shrink=4)),
        ("normal sd_floor0.10", Model(sd_floor=0.10)),
        ("normal pooled-z", Model(z_mode="pooled")),
        ("normal pooled-z smooth1", Model(z_mode="pooled", smooth_win=1)),
        ("empirical uniform", None),
        ("lift bw=1.0", Model(weight=("lift", 1.0))),
        ("lift bw=0.6", Model(weight=("lift", 0.6))),
        ("lift bw=0.6 pooled-z", Model(weight=("lift", 0.6), z_mode="pooled")),
        ("lift bw=0.6 smooth1", Model(weight=("lift", 0.6), smooth_win=1)),
        ("lift bw=0.4", Model(weight=("lift", 0.4))),
    ]
    for label, model in variants:
        if model is None:
            fn = empirical_model
        else:
            fn = lambda pool, t, m=model: m.fit_quantiles(pool, t)
        results[label], _ = run_loo(comps, fn, label)
    print("\nworst baseline cases:")
    for name, loss in sorted(base_pc.items(), key=lambda kv: -kv[1])[:8]:
        print(f"  {name[:44]:46s} {loss:.3f}")


if __name__ == "__main__":
    main()
