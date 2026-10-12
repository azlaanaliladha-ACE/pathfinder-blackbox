"""Preflight: Nemotron researches the site with Tavily and writes the mission brief.

    python plan_mission.py --site "Fjadrargljufur canyon, Iceland" --goal "survey the river bed, stay clear of water"
    python record.py --source videos\aerial.webm --name demo4 --mission missions\fjadrargljufur.json
"""
import argparse
import json
import os
import re

from terrain_nav.mission import brief_markdown, plan_mission


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", required=True, help="place name, as specific as possible")
    ap.add_argument("--goal", required=True, help="what the operator wants, in plain language")
    ap.add_argument("--name", help="output name (default: from the site)")
    args = ap.parse_args()

    print(f"Planning mission for {args.site} ...")
    brief = plan_mission(args.goal, args.site)
    name = args.name or re.sub(r"[^a-z0-9]+", "_", args.site.lower()).strip("_")[:40]
    os.makedirs("missions", exist_ok=True)
    with open(f"missions/{name}.json", "w", encoding="utf-8") as f:
        json.dump(brief, f, indent=1)
    with open(f"missions/{name}.md", "w", encoding="utf-8") as f:
        f.write(brief_markdown(brief))
    print(f"\n{brief.get('go_no_go')}: {brief.get('objective')}")
    print(f"avoid {brief['avoid_terrain']}, risk ceiling {brief.get('max_risk')}, {len(brief['sources'])} sources")
    print(f"Saved missions/{name}.json and missions/{name}.md")


if __name__ == "__main__":
    main()
