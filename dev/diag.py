"""Diagnostics: per-comparable lift, peak ratio, and loss split by regime."""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import baseline_quantiles, load_comps, score_forecast
from loo import Model

comps = load_comps()
rows = []
for c in comps:
    lift = c.logv[1] - c.pre_log if c.pre_log is not None else float("nan")
    peak_ratio = max(c.logv.values()) - c.logv[1]
    rows.append((c.name, c.raw[1], lift, peak_ratio))
rows.sort(key=lambda r: r[2])
print(f"{'name':46s} {'v1':>9s} {'lift':>6s} {'ln(peak/v1)':>11s}")
for name, v1, lift, pr in rows:
    print(f"{name[:44]:46s} {v1:9.0f} {lift:6.2f} {pr:11.2f}")

model = Model(z_mode="pooled")
print("\npooled-z per-case loss by lift tercile:")
terciles = [[], [], []]
for t in comps:
    pool = [c for c in comps if c is not t]
    target = {"logp1": math.log(max(1.0, t.raw[1])),
              "lift": (t.logv[1] - t.pre_log) if t.pre_log is not None else None}
    q = model.fit_quantiles(pool, target)
    loss, _ = score_forecast(t.raw, q)
    lift = target["lift"] if target["lift"] is not None else 0.0
    idx = 0 if lift < 0.8 else (1 if lift < 2.2 else 2)
    terciles[idx].append((t.name, lift, loss))
for i, group in enumerate(terciles):
    if group:
        mean = sum(g[2] for g in group) / len(group)
        print(f" tercile {i}: n={len(group)} mean loss={mean:.3f}")
        for name, lift, loss in sorted(group, key=lambda g: -g[2])[:4]:
            print(f"    {name[:40]:42s} lift={lift:5.2f} loss={loss:.3f}")
