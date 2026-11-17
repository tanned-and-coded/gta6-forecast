"""Edge cases: main.py must always emit a valid forecast and never crash."""
import importlib.util
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("entry_main", REPO / "main.py")
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)


def valid(fc):
    rows = fc["hours"]
    assert len(rows) == 23 and sorted(r["hour"] for r in rows) == list(range(2, 25))
    for r in rows:
        q = r["q"]
        assert len(q) == 5 and all(math.isfinite(x) and 0 < x <= 1e9 for x in q)
        assert all(q[i] <= q[i + 1] for i in range(4))
    return True


base = {"levels": [0.05, 0.25, 0.5, 0.75, 0.95],
        "known": [{"hour": 1, "viewers": 50000}],
        "forecast": [{"hour": h} for h in range(2, 25)],
        "notes": "", "comparables": []}

cases = {
    "no comparables": dict(base),
    "hour-1 viewers 0": {**base, "known": [{"hour": 1, "viewers": 0}]},
    "hour-1 viewers null": {**base, "known": [{"hour": 1, "viewers": None}]},
    "known missing": {k: v for k, v in base.items() if k != "known"},
    "known as dict": {**base, "known": {"hour": 1, "viewers": 1234}},
    "forecast missing": {k: v for k, v in base.items() if k != "forecast"},
    "levels missing": {k: v for k, v in base.items() if k != "levels"},
    "levels unusual": {**base, "levels": [0.1, 0.5, 0.9]},
    "comparables null": {**base, "comparables": None},
    "comparable empty hourly": {**base, "comparables": [
        {"name": "x", "day_start": "2026-01-01T00:00:00Z", "hourly": [],
         "unlock": "", "platforms": ""}]},
    "comparable sparse": {**base, "comparables": [
        {"name": "x", "day_start": "2026-01-01T00:00:00Z",
         "hourly": [{"from": "2026-01-01T00:00:00Z", "viewers": 100, "channels": 3}],
         "unlock": "Single global unlock 2026-01-01T14:00Z", "platforms": "PC"}]},
    "comparable bad rows": {**base, "comparables": [
        {"name": "x", "day_start": "not-a-date", "hourly": [{"from": "junk", "viewers": "NaN"}],
         "unlock": None, "platforms": None}, None, 42]},
    "huge hour-1": {**base, "known": [{"hour": 1, "viewers": 10**12}]},
    "tiny hour-1": {**base, "known": [{"hour": 1, "viewers": 0.4}]},
}
for label, case in cases.items():
    fc = entry.forecast_case(case)
    assert valid(fc), label
    print(f"  ok: {label}")

# Whole-program level: a malformed .in file still yields a valid .out.
with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    (td / "in").mkdir()
    (td / "out").mkdir()
    (td / "in" / "bad.in").write_text("{not json", encoding="utf-8")
    (td / "in" / "empty.in").write_text("{}", encoding="utf-8")
    env = {**os.environ, "MAKEFUN_INPUT_DIR": str(td / "in"),
           "MAKEFUN_OUTPUT_DIR": str(td / "out")}
    subprocess.run([sys.executable, str(REPO / "main.py")], env=env, check=True)
    for stem in ("bad", "empty"):
        fc = json.loads((td / "out" / f"{stem}.out").read_text())
        assert valid(fc), stem
    print("  ok: malformed input files")
print("all edge cases pass")
