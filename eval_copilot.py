"""Benchmark Nemotron variants as the in-flight copilot, on scenes from a real recorded run.

Replays every logged scene summary through each model and measures what matters for a 1 Hz
control loop: latency, valid-JSON rate, how often the safety guard has to override, and
agreement with the largest model.

    python eval_copilot.py web\static\videos\demo4_copilot_log.json
    python eval_copilot.py <log> --models nvidia/a nvidia/b --limit 40
"""
import argparse
import json
import re
import statistics
import time

from terrain_nav import nebius, safety
from terrain_nav.copilot import SYSTEM, _parse


def _params_b(model_id: str) -> float:
    """Total parameter count from ids like ...-120b-a12b; 0 if absent."""
    sizes = [float(x) for x in re.findall(r"(\d+(?:\.\d+)?)b(?![a-z])", model_id.lower())]
    return max(sizes, default=0.0)


def run(model: str, scenes: list, mission) -> dict:
    out = {"model": model, "lat": [], "valid": 0, "overrides": 0, "actions": [], "tokens": 0}
    for sc in scenes:
        t0 = time.perf_counter()
        try:
            r = nebius.chat(model, [{"role": "system", "content": SYSTEM},
                                    {"role": "user", "content": json.dumps(dict(sc, mission=mission))}], 600)
            out["lat"].append((time.perf_counter() - t0) * 1000)
            out["tokens"] += getattr(r.usage, "total_tokens", 0) or 0
            d = _parse(r.choices[0].message.content or "")
            out["valid"] += 1
            final, why = safety.check(d.action, sc, mission)
            out["overrides"] += bool(why)
            out["actions"].append(final)
        except Exception:
            out["actions"].append(None)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--models", nargs="*", help="default: every Nemotron on your Token Factory account")
    ap.add_argument("--limit", type=int, default=30)
    args = ap.parse_args()

    data = json.load(open(args.log, encoding="utf-8"))
    scenes = [e["scene"] for e in data["log"]][: args.limit]
    mission = data.get("mission")
    mission = mission and {k: mission.get(k) for k in ("objective", "avoid_terrain", "max_risk")}
    models = args.models or [m for m in nebius.available() if "nemotron" in m.lower()]
    print(f"{len(scenes)} scenes x {len(models)} models")

    results = []
    for m in models:
        print(f"  {m} ...", flush=True)
        results.append(run(m, scenes, mission))
    ref = max(results, key=lambda r: _params_b(r["model"]))["actions"]

    rows = ["| model | p50 ms | p95 ms | valid JSON | guard overrides | agrees w/ largest | tokens/call |",
            "|---|---|---|---|---|---|---|"]
    for r in results:
        lat = sorted(r["lat"]) or [0]
        agree = sum(a == b for a, b in zip(r["actions"], ref) if a and b) / max(len(scenes), 1)
        rows.append(f"| {r['model']} | {statistics.median(lat):.0f} | {lat[int(len(lat) * .95)]:.0f} | "
                    f"{r['valid'] / len(scenes):.0%} | {r['overrides']} | {agree:.0%} | "
                    f"{r['tokens'] / max(r['valid'], 1):.0f} |")
    table = "\n".join(rows)
    print("\n" + table)
    out = args.log.replace("_copilot_log.json", "") + "_model_eval.md"
    with open(out, "w", encoding="utf-8") as f:
        f.write(f"# Nemotron copilot benchmark\n\n{len(scenes)} real scenes from `{args.log}`.\n\n{table}\n")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
