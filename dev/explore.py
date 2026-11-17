"""Explore the practice-case comparables: launch-day coverage and shapes."""
import json
import math
from datetime import datetime, timezone
from pathlib import Path

KIT = Path.home() / "workspace/makefun/kits-gta/gta6-twitch-ccu-forecast-01-kit"


def ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def load_case(name):
    return json.loads((KIT / "input" / f"{name}.in").read_text())


def launch_day_points(comp):
    """Return {hour 1..24: viewers} from raw points whose 'from' is in that hour."""
    start = ts(comp["day_start"])
    out = {}
    for p in comp["hourly"]:
        t = ts(p["from"])
        h = int((t - start) // 3600) + 1
        if 1 <= h <= 24 and p["viewers"] is not None:
            out.setdefault(h, []).append(p["viewers"])
    return {h: sum(v) / len(v) for h, v in out.items()}


def main():
    case = load_case("p00")
    comps = case["comparables"]
    print(f"{len(comps)} comparables in p00")
    names = [c["name"] for c in comps]
    # Consistency across cases: same comparable set?
    for other in ("p01", "p02", "p03"):
        names2 = [c["name"] for c in load_case(other)["comparables"]]
        print(other, "same set:", sorted(names) == sorted(names2))
    for c in comps:
        day = launch_day_points(c)
        cov = len(day)
        v1 = day.get(1)
        peak = max(day.values()) if day else None
        print(f"{c['name'][:44]:46s} day_start={c['day_start'][:10]} pts={len(c['hourly']):4d} "
              f"launch-hours={cov:2d} v1={v1} peak={peak} platforms={c['platforms']}")


if __name__ == "__main__":
    main()
