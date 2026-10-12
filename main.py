"""Desktop viewer.

    python main.py                       # laptop webcam
    python main.py --source clip.mp4     # local video
    python main.py --source <YouTube URL>
"""
import argparse
import os
import time

import cv2

import config

from terrain_nav.mission import load as load_mission
from terrain_nav.pipeline import Pipeline
from terrain_nav.sources import open_source


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", help="video file or YouTube URL (default: webcam)")
    ap.add_argument("--mission", help="mission brief JSON from plan_mission.py (copilot + guard enforce it)")
    ap.add_argument("--scene", choices=list(config.SCENES), default="outdoor")
    args = ap.parse_args()
    config.use_scene(args.scene)

    pipe, cap = Pipeline(mission=load_mission(args.mission)), open_source(args.source)
    pipe.copilot.chain.start(f"live-{int(time.time())}", pipe.copilot.mission)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        out = pipe.process(frame)
        cv2.imshow("Pathfinder", out)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        if key in (ord("s"), ord("S")):
            os.makedirs("screenshots", exist_ok=True)
            cv2.imwrite(f"screenshots/frame_{pipe.frame_no}.png", out)
    cap.release()
    cv2.destroyAllWindows()

    report = pipe.copilot.mission_report()
    if report:
        os.makedirs("reports", exist_ok=True)
        path = f"reports/mission_{int(time.time())}.md"
        with open(path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"Mission report: {path}")
    chain = pipe.copilot.chain.end(report)
    if chain:
        print(f"Monad mission #{chain['mission_id']}: {chain['confirmed']} decisions on-chain, head {chain['head']}")


if __name__ == "__main__":
    main()
