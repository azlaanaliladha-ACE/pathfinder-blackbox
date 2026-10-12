"""Preflight mission agent: Nemotron (reasoning tier) + Tavily web search.

Given an operator goal in plain language and a site, Nemotron researches the site with Tavily
(terrain, weather, water, drone rules), then writes a mission brief. The brief has a
machine-readable part (terrain to avoid, risk ceiling, return conditions) that the in-flight
copilot and the safety guard enforce, and a human-readable briefing with cited sources.
"""
import json
import os
from datetime import date

import config
from terrain_nav import nebius

TERRAIN_CLASSES = [k for k in config.TERRAIN if k not in ("sky", "indoor")]

SYSTEM = f"""You are the mission planner for a small camera drone / ground rover used for field survey
and search-and-rescue. Today is {date.today().isoformat()}.
Research the site with the web_search tool before planning (terrain, water bodies, current weather and
wind, local drone regulations or no-fly zones). Use 2-4 focused searches. Then answer with ONLY a JSON
object, no prose and no code fences:
{{"objective": "<one line>",
  "avoid_terrain": [subset of {TERRAIN_CLASSES}],
  "max_risk": number 0..1 (the copilot holds above this),
  "min_clearance_m": number,
  "return_conditions": ["<short condition>", ...],
  "hazards": [{{"hazard": "<short>", "source": "<url from your searches>"}}],
  "go_no_go": "GO" | "CAUTION" | "NO-GO",
  "briefing": "<markdown, 120-220 words, for the pilot, cite sources inline as [n]>"}}"""

TOOLS = [{"type": "function", "function": {
    "name": "web_search",
    "description": "Search the live web (Tavily). Returns titles, urls and content snippets.",
    "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}}]


def tavily_search(query: str) -> list[dict]:
    from tavily import TavilyClient
    res = TavilyClient(api_key=os.environ["TAVILY_API_KEY"]).search(
        query, max_results=config.TAVILY_MAX_RESULTS, search_depth="advanced")
    return [{"title": r["title"], "url": r["url"], "content": r["content"][:600]} for r in res["results"]]


def plan_mission(goal: str, site: str, log=print) -> dict:
    model = nebius.model("reason")
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Site: {site}\nOperator goal: {goal}"}]
    sources = []
    try:
        for _ in range(config.PLANNER_MAX_STEPS):
            r = nebius.chat(model, msgs, 4000, tools=TOOLS, tool_choice="auto")
            m = r.choices[0].message
            if not m.tool_calls:
                break
            msgs.append({"role": "assistant", "content": m.content or "",
                         "tool_calls": [tc.model_dump() for tc in m.tool_calls]})
            for tc in m.tool_calls:
                q = json.loads(tc.function.arguments).get("query", site)
                log(f"  Nemotron -> web_search({q!r})")
                hits = tavily_search(q)
                sources += hits
                msgs.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(hits)})
        else:                                 # out of search rounds: force the final answer
            msgs.append({"role": "user", "content": "Stop searching. Answer with the JSON now."})
            m = nebius.chat(model, msgs, 4000).choices[0].message
        text = m.content
    except Exception as e:                    # endpoint without tool calling: search first, then plan
        log(f"  tool calling unavailable ({type(e).__name__}); running fixed searches")
        for q in (f"{site} terrain water hazards", f"{site} weather wind today",
                  f"drone flying rules {site}"):
            log(f"  web_search({q!r})")
            sources += tavily_search(q)
        msgs = [msgs[0], {"role": "user", "content": f"Site: {site}\nOperator goal: {goal}\n"
                          f"Search results (already gathered, do not call tools):\n{json.dumps(sources)}"}]
        text = nebius.chat(model, msgs, 4000).choices[0].message.content

    text = nebius.strip_think(text)
    brief = json.loads(text[text.find("{"):text.rfind("}") + 1])
    brief["avoid_terrain"] = [t for t in brief.get("avoid_terrain", []) if t in TERRAIN_CLASSES]
    brief.update(goal=goal, site=site, model=model, searches=len(sources),
                 sources=list(dict.fromkeys(s["url"] for s in sources)))
    return brief


def brief_markdown(b: dict) -> str:
    lines = [f"# Mission brief: {b['site']}", "",
             f"**Goal:** {b['goal']}  ", f"**Objective:** {b.get('objective', '')}  ",
             f"**Decision:** {b.get('go_no_go', '?')}  ", f"**Planner:** {b['model']} + Tavily ({b['searches']} results)", "",
             b.get("briefing", ""), "", "## Constraints enforced in flight", "",
             f"- avoid terrain: {', '.join(b['avoid_terrain']) or 'none'}",
             f"- copilot risk ceiling: {b.get('max_risk', 0.7)}",
             f"- min clearance: {b.get('min_clearance_m', config.GUARD_STOP_DIST_M)} m"]
    lines += [f"- return if: {c}" for c in b.get("return_conditions", [])]
    lines += ["", "## Hazards", ""] + [f"- {h['hazard']} ({h.get('source', '')})" for h in b.get("hazards", [])]
    lines += ["", "## Sources", ""] + [f"{i + 1}. {u}" for i, u in enumerate(b["sources"])]
    return "\n".join(lines)


def load(path: str | None) -> dict | None:
    if not path:
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)
