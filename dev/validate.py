"""Validate the shipped main.py.

1. Leave-one-out over the pooled comparables, driving main.py's own
   forecast_case with synthesized cases (the pseudo-target's raw
   comparable dict becomes `notes`-carried unlock text plus its hour-1
   print as `known`; the other launches are its `comparables`). The
   house baseline formula is scored identically.
2. End-to-end runs of main.py on the practice inputs p00..p03: output
   shape checks, byte-identical reruns, and wall time per case.
"""
import importlib.util
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import LEVELS, baseline_quantiles, load_case, score_forecast
import common

REPO = Path(__file__).resolve().parent.parent
KIT = common.KIT

spec = importlib.util.spec_from_file_location("entry_main", REPO / "main.py")
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)


def pseudo_case(target_raw, pool_raws):
    return {
        "levels": LEVELS,
        "known": [{"hour": 1, "viewers": target_raw_v1(target_raw)}],
        "forecast": [{"hour": h} for h in range(2, 25)],
        "notes": target_raw.get("unlock") or "",
        "comparables": pool_raws,
    }


def target_raw_v1(raw):
    buckets = common.bucket_hours(raw["hourly"], raw["day_start"])
    return buckets.get(0, 1.0)


def raw_hours(raw):
    buckets = common.bucket_hours(raw["hourly"], raw["day_start"])
    return {h: buckets[h - 1] for h in range(2, 25) if (h - 1) in buckets}


def loo():
    case = load_case("p00")
    raws = []
    for raw in case["comparables"]:
        c = common.Comp(raw)
        if c.usable:
            raws.append(raw)
    print(f"LOO over {len(raws)} comparables (shipped main.py vs house baseline)")
    tot_m, tot_b, items = 0.0, 0.0, 0
    per = []
    for i, t in enumerate(raws):
        pool = [r for j, r in enumerate(raws) if j != i]
        fc = entry.forecast_case(pseudo_case(t, pool))
        q = {row["hour"]: row["q"] for row in fc["hours"]}
        truth = raw_hours(t)
        lm, n = score_forecast(truth, q)
        lb, _ = score_forecast(truth, baseline_quantiles(target_raw_v1(t)))
        per.append((t["name"], lm, lb))
        tot_m += lm * n
        tot_b += lb * n
        items += n
    print(f"  baseline LOO = {tot_b / items:.4f}")
    print(f"  model    LOO = {tot_m / items:.4f}")
    per.sort(key=lambda r: -r[1])
    print("  worst model cases:", ", ".join(f"{n[:28]} {l:.2f}" for n, l, _ in per[:5]))
    return tot_m / items, tot_b / items


def check_output(path):
    data = json.loads(path.read_text())
    assert set(data.keys()) == {"hours"}, data.keys()
    rows = data["hours"]
    assert len(rows) == 23, len(rows)
    assert sorted(r["hour"] for r in rows) == list(range(2, 25))
    for r in rows:
        assert set(r.keys()) == {"hour", "q"}
        q = r["q"]
        assert len(q) == 5
        assert all(isinstance(x, (int, float)) and math.isfinite(x) and 0 < x <= 1e9 for x in q), q
        assert all(q[i] <= q[i + 1] for i in range(4)), q
    assert path.stat().st_size <= 65536
    return True


def end_to_end():
    print("end-to-end on practice cases:")
    for name in ("p00", "p01", "p02", "p03"):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            (td / "in").mkdir()
            (td / "out").mkdir()
            (td / "in" / f"{name}.in").write_bytes((KIT / "input" / f"{name}.in").read_bytes())
            env = {**os.environ, "MAKEFUN_INPUT_DIR": str(td / "in"),
                   "MAKEFUN_OUTPUT_DIR": str(td / "out")}
            t0 = time.perf_counter()
            subprocess.run([sys.executable, str(REPO / "main.py")], env=env, check=True)
            dt = time.perf_counter() - t0
            first = (td / "out" / f"{name}.out").read_bytes()
            subprocess.run([sys.executable, str(REPO / "main.py")], env=env, check=True)
            second = (td / "out" / f"{name}.out").read_bytes()
            check_output(td / "out" / f"{name}.out")
            print(f"  {name}: valid shape, deterministic={first == second}, "
                  f"runtime={dt:.2f}s, bytes={len(first)}")


if __name__ == "__main__":
    loo()
    end_to_end()
