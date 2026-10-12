"""Nemotron flight copilot, served by Nebius Token Factory.

Three phases, three uses of Nemotron:
  plan    terrain_nav/mission.py: reasoning-tier Nemotron + Tavily writes the mission brief
  fly     this file: fast-tier Nemotron reads a ~1 KB scene summary once per second and picks
          the next manoeuvre; terrain_nav/safety.py can veto it with hard rules
  debrief this file: reasoning-tier Nemotron turns the decision log into a mission report

The local stack (CLIP, Depth Anything, YOLO-World, planner) produces the scene summary, so no
pixels leave the vehicle. With NEBIUS_API_KEY the decisions come from Nemotron; without it a local
rule-based copilot (planner + obstacles) stands in, so the pipeline and the Monad recorder still run.

Every logged decision is also streamed to the FlightRecorder contract on Monad (terrain_nav/monad.py).
"""
import json
import math
import os
import threading
import time
from collections import Counter
from dataclasses import dataclass, field

import config
from terrain_nav import monad, nebius, safety

ACTIONS = ["PROCEED", "SLOW", "VEER_LEFT", "VEER_RIGHT", "CLIMB", "HOLD", "RETURN"]
MEMORY_STEPS = 3                    # recent decisions sent back so Nemotron sees trends

SYSTEM = f"""You are the onboard navigation copilot of a small FPV drone / ground rover.
You never see the image. A perception stack running on the vehicle gives you a JSON scene summary:
- terrain: fraction of the drivable band per class, with traversal cost (inf = impassable)
- obstacles: label, distance in metres, bearing in degrees (negative = left), detector confidence
- planner: whether a safe path exists and which way it bends
- state: altitude above ground (m), ground speed (m/s), climb rate (m/s), pitch/roll (deg)
- recent: your last few decisions, oldest first (use them to notice trends, avoid flip-flopping)
- mission: the operator's objective and constraints, if any
Decide the next manoeuvre. Be conservative near obstacles closer than 5 m and over water or mud.
Low-confidence detections (conf < 0.3) far away matter less than close confident ones.
Answer with ONLY a JSON object, no prose, no code fences:
{{"action": one of {ACTIONS}, "risk": number 0..1, "reason": "<= 12 words", "callout": "<= 6 words, spoken to the pilot"}}"""

REPORT_SYSTEM = """You are a field robotics analyst. You get the mission brief (if any) and a time-ordered
log of a drone/rover run: per-step scene summaries, the copilot's decisions, and any safety-guard
overrides. Write a mission report in Markdown with these sections: Summary (3 sentences, say whether
the objective was met), Terrain encountered, Hazards and how they were handled, Copilot performance
(were any decisions wrong, too cautious, or overridden by the guard? cite timestamps), Recommendations
for the next run. Be specific, cite times like t=12.4s, and do not invent data that is not in the log."""


@dataclass
class Decision:
    action: str = "STANDBY"
    risk: float = 0.0
    reason: str = ""
    callout: str = ""
    proposed: str = ""              # Nemotron's action before the safety guard
    override: str = ""              # why the guard changed it ('' = accepted)
    latency_ms: float = 0.0
    t: float = 0.0                  # flight time the decision was made for


@dataclass
class Copilot:
    mission: dict | None = None
    enabled: bool = False
    status: str = ""
    model: str = ""
    last: Decision = field(default_factory=Decision)
    log: list = field(default_factory=list)
    calls: int = 0
    tokens: int = 0

    def __post_init__(self):
        self._busy = threading.Lock()
        self.chain = monad.Recorder(ACTIONS)
        self.local = not os.environ.get("NEBIUS_API_KEY")
        self.model = "local-rules" if self.local else nebius.model("fast")
        self.enabled = True
        self.status = "READY"

    def reset(self):
        self.last, self.log, self.calls, self.tokens = Decision(), [], 0, 0

    @property
    def overrides(self) -> int:
        return sum(1 for e in self.log if e["decision"]["override"])

    # --- in-flight decisions ---

    def maybe_decide(self, scene: dict, t: float, blocking: bool):
        """Ask Nemotron for a decision if one is due. Live mode runs it on a thread so the
        video never waits on the network; recording runs it inline so decisions line up."""
        if not self.enabled or t - self.last.t < config.COPILOT_EVERY_S and self.calls:
            return
        if blocking:
            self._decide(scene, t)
        elif self._busy.acquire(blocking=False):
            threading.Thread(target=self._decide, args=(scene, t, True), daemon=True).start()

    def _decide(self, scene: dict, t: float, release: bool = False):
        try:
            msg = dict(scene, recent=[{"t": e["t"], "action": e["decision"]["action"]}
                                      for e in self.log[-MEMORY_STEPS:]])
            if self.mission:
                msg["mission"] = {k: self.mission.get(k) for k in
                                  ("objective", "avoid_terrain", "max_risk", "return_conditions")}
            t0 = time.perf_counter()
            d = _local_decide(scene) if self.local else                 _parse(self._chat(self.model, SYSTEM, json.dumps(msg), max_tokens=600))
            d.latency_ms, d.t = (time.perf_counter() - t0) * 1000, t
            self.calls += self.local
            d.proposed = d.action
            d.action, d.override = safety.check(d.action, scene, self.mission)
            ceiling = (self.mission or {}).get("max_risk")
            if not d.override and ceiling and d.risk > ceiling and d.action in safety.UNSAFE_FORWARD:
                d.action, d.override = "HOLD", f"risk {d.risk:.2f} over ceiling {ceiling}"
            if d.override:
                d.callout = d.override[:40]
            self.last = d
            self.status = "ONLINE"
            self.log.append({"t": round(t, 1), "scene": scene, "decision": {
                "action": d.action, "proposed": d.proposed, "override": d.override,
                "risk": round(d.risk, 3), "reason": d.reason, "latency_ms": round(d.latency_ms)}})
            self.chain.log(self.log[-1])
        except Exception as e:                   # network or quota: keep flying on local planner
            self.status = f"ERROR {type(e).__name__}"[:22].upper()
            self.last.t = t
        finally:
            if release:
                self._busy.release()

    def _chat(self, model: str, system: str, user: str, max_tokens: int) -> str:
        r = nebius.chat(model, [{"role": "system", "content": system}, {"role": "user", "content": user}],
                        max_tokens)
        self.calls += 1
        self.tokens += getattr(r.usage, "total_tokens", 0) or 0
        return r.choices[0].message.content or ""

    # --- after the flight ---

    def stats(self) -> dict:
        lat = sorted(e["decision"]["latency_ms"] for e in self.log)
        return {"model": self.model, "decisions": len(self.log), "overrides": self.overrides,
                "tokens": self.tokens, "actions": dict(Counter(e["decision"]["action"] for e in self.log)),
                "latency_ms_p50": lat[len(lat) // 2] if lat else None,
                "latency_ms_p95": lat[int(len(lat) * 0.95)] if lat else None,
                "chain_confirmed": self.chain.confirmed, "chain_failed": self.chain.failed}

    def mission_report(self) -> str | None:
        if not self.enabled or not self.log:
            return None
        with self._busy:                          # let any in-flight call finish first
            pass
        s = self.stats()
        if self.local:
            return _local_report(self.mission, s, self.log)
        payload = {"mission": self.mission, "stats": s, "log": self.log[-config.REPORT_MAX_STEPS:]}
        model = nebius.model("reason")
        report = nebius.strip_think(self._chat(model, REPORT_SYSTEM, json.dumps(payload), max_tokens=6000))
        return (f"{report}\n\n---\n*Copilot {s['model']}: {s['decisions']} decisions, "
                f"p50 {s['latency_ms_p50']} ms, {s['overrides']} safety overrides. "
                f"Report by {model} on Nebius Token Factory.*\n")


def _local_decide(scene: dict) -> Decision:
    """Rule-based stand-in when no LLM key is set: follow the planner, slow for close obstacles."""
    obs = [o for o in scene.get("obstacles", []) if o["conf"] >= 0.3]
    near = min(obs, key=lambda o: o["dist_m"]) if obs else None
    pl = scene.get("planner", {})
    if not pl.get("path_found"):
        return Decision("HOLD", 0.8, "no traversable path in view", "hold position")
    if near and near["dist_m"] < 5:
        side = "VEER_RIGHT" if near["bearing_deg"] < 0 else "VEER_LEFT"
        return Decision(side, 0.6, f"{near['label']} {near['dist_m']}m at {near['bearing_deg']} deg", "obstacle, veering")
    bend = pl.get("bends", "straight")
    if bend != "straight":
        return Decision(f"VEER_{bend.upper()}", 0.3, f"planner path bends {bend}", f"bearing {bend}")
    return Decision("PROCEED", 0.15, "clear path straight ahead", "clear ahead")


def _local_report(mission: dict | None, s: dict, log: list) -> str:
    ovr = [e for e in log if e["decision"]["override"]]
    lines = [f"# Mission report (local copilot)", "",
             f"Objective: {(mission or {}).get('objective', 'free flight')}", "",
             f"- {s['decisions']} decisions, {s['overrides']} safety-guard overrides",
             f"- actions: " + ", ".join(f"{k} x{v}" for k, v in s["actions"].items()), "",
             "## Guard overrides"]
    lines += [f"- t={e['t']}s: {e['decision']['proposed']} -> {e['decision']['action']} ({e['decision']['override']})"
              for e in ovr] or ["- none"]
    return "\n".join(lines) + "\n"


def _parse(text: str) -> Decision:
    text = nebius.strip_think(text)
    s, e = text.find("{"), text.rfind("}")
    obj = json.loads(text[s:e + 1])
    action = str(obj.get("action", "HOLD")).upper().replace(" ", "_")
    return Decision(action=action if action in ACTIONS else "HOLD",
                    risk=float(min(max(obj.get("risk", 0.5), 0.0), 1.0)),
                    reason=str(obj.get("reason", ""))[:90],
                    callout=str(obj.get("callout", ""))[:40])


def scene_summary(labels, idx, terrain_cost, boxes, path, drone_state: dict) -> dict:
    """Compact, model-readable description of one frame (no pixels leave the vehicle)."""
    band = idx[config.PLAN_FIRST_ROW:]
    counts = {labels[i]: int((band == i).sum()) for i in set(band.ravel().tolist())}
    total = max(sum(counts.values()), 1)
    costs = {labels[i]: terrain_cost[config.PLAN_FIRST_ROW:][band == i].max() for i in set(band.ravel().tolist())}
    terrain = {k: {"frac": round(v / total, 2), "cost": "inf" if math.isinf(costs[k]) else round(float(costs[k]), 1)}
               for k, v in sorted(counts.items(), key=lambda kv: -kv[1])}

    w = config.FRAME_SIZE[0]
    f = config.focal_px(w)
    obstacles = []
    for b in sorted(boxes, key=lambda b: b.distance_m)[:6]:
        x1, _, x2, _ = b.obstacle.box
        bearing = math.degrees(math.atan(((x1 + x2) / 2 - w / 2) / f))
        obstacles.append({"label": b.obstacle.label, "dist_m": round(b.distance_m, 1),
                          "bearing_deg": round(bearing), "conf": round(b.obstacle.conf, 2)})

    if path is None:
        planner = {"path_found": False}
    else:
        cols = idx.shape[1]
        bend = (path[-1][1] - (cols - 1) / 2) / cols
        planner = {"path_found": True, "bends": "left" if bend < -0.15 else "right" if bend > 0.15 else "straight",
                   "length_cells": len(path)}

    s = drone_state
    state = {"alt_m": round(float(s["alt"]), 1), "speed_mps": round(float(s["speed"]), 1),
             "climb_mps": round(float(s["climb"]), 1), "pitch_deg": round(s["pitch"]), "roll_deg": round(s["roll"])}
    return {"terrain": terrain, "obstacles": obstacles, "planner": planner, "state": state}
