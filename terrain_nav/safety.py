"""Deterministic safety guard between Nemotron and the vehicle.

The language model is good at weighing context, but a physical system needs hard limits that
never depend on a model's output. Every decision passes through `check()`; if a rule fires,
the action is replaced and the override is logged, shown on the HUD and counted in the debrief.
"""
import config

UNSAFE_FORWARD = {"PROCEED", "SLOW"}


def check(action: str, scene: dict, mission: dict | None = None) -> tuple[str, str]:
    """Return (final action, override reason or '')."""
    obs = scene.get("obstacles", [])
    ahead = [o for o in obs if abs(o["bearing_deg"]) <= config.GUARD_CONE_DEG]
    close = [o for o in ahead if o["dist_m"] < config.GUARD_STOP_DIST_M]
    st = scene.get("state", {})

    if close and action in UNSAFE_FORWARD:
        o = min(close, key=lambda o: o["dist_m"])
        return "HOLD", f"{o['label']} {o['dist_m']}m dead ahead"
    if not scene.get("planner", {}).get("path_found", True) and action in UNSAFE_FORWARD:
        return "HOLD", "planner found no safe path"
    if st.get("alt_m", 99) < config.GUARD_MIN_ALT_M and st.get("climb_mps", 0) < config.GUARD_DESCENT_MPS \
            and action != "CLIMB":
        return "CLIMB", f"low ({st['alt_m']}m) and sinking"
    for o in close:                               # veering into a close obstacle
        if (action == "VEER_LEFT" and o["bearing_deg"] < 0) or (action == "VEER_RIGHT" and o["bearing_deg"] > 0):
            return "HOLD", f"veer blocked by {o['label']}"

    if mission:
        avoid = set(mission.get("avoid_terrain", []))
        ter = scene.get("terrain", {})
        bad = [k for k in avoid if ter.get(k, {}).get("frac", 0) >= 0.5]
        if bad and action == "PROCEED":
            return "SLOW", f"mission avoids {bad[0]}"
    return action, ""
