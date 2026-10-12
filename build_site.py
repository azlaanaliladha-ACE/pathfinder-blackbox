"""Copy one recorded flight into site/ (static, deployable anywhere, verifies itself against Monad).

    python build_site.py demo4
"""
import json
import shutil
import sys
import os

SRC = os.path.join("web", "static", "videos")
OUT = "site"

name = sys.argv[1] if len(sys.argv) > 1 else "demo4"
data = json.load(open(os.path.join(SRC, f"{name}_copilot_log.json"), encoding="utf-8"))
if not data.get("chain"):
    raise SystemExit("Record the flight with the Monad recorder enabled first (record.py with .env set).")
os.makedirs(OUT, exist_ok=True)
for ext in (".mp4", "_map3d.png", "_report.md"):
    if os.path.exists(os.path.join(SRC, name + ext)):
        shutil.copy(os.path.join(SRC, name + ext), os.path.join(OUT, "flight" + ext))
abi = json.load(open(os.path.join("contracts", "FlightRecorder.json")))["abi"]
with open(os.path.join(OUT, "flight.json"), "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False)
with open(os.path.join(OUT, "abi.json"), "w") as f:
    json.dump(abi, f)
print(f"site/ ready: mission #{data['chain']['mission_id']} on chain {data['chain']['chain_id']}")
