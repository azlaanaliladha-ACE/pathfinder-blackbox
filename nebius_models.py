"""List the Nemotron models your Nebius Token Factory key can call, show which one each tier
resolves to, and ping them. Also checks the Tavily key.

    python nebius_models.py
"""
import os

from terrain_nav import nebius

print("Nemotron models on your Token Factory account:")
for i in nebius.available():
    if "nemotron" in i.lower():
        print("  ", i)

for tier in ("fast", "reason"):
    m = nebius.model(tier)
    try:
        r = nebius.chat(m, [{"role": "user", "content": "Reply with the single word: ready"}], 300)
        print(f"\n{tier:6s} -> {m}: {nebius.strip_think(r.choices[0].message.content)[:40]!r}")
    except Exception as e:
        print(f"\n{tier:6s} -> {m}: FAILED {e}  (set COPILOT_MODEL / REPORT_MODEL)")

if os.environ.get("TAVILY_API_KEY"):
    from terrain_nav.mission import tavily_search
    print(f"\nTavily: {len(tavily_search('Nebius Token Factory'))} results")
else:
    print("\nTavily: TAVILY_API_KEY not set (needed for plan_mission.py)")
