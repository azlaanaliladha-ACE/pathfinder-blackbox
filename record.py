"""Run a video through the pipeline and save an annotated, browser-playable MP4.

    python record.py --source <file or YouTube URL> --name demo1 [--seconds 30]
"""
import argparse
import json
import os

import cv2
import imageio_ffmpeg
import subprocess

import config
from terrain_nav.orbit3d import Orbit
from terrain_nav.mission import load as load_mission
from terrain_nav.pipeline import Pipeline
from terrain_nav.sources import open_source

OUT_DIR = os.path.join("web", "static", "videos")
END_CARD_S = 3.0                                  # seconds the final top-down map is shown
ORBIT_S = 12.0                                    # seconds for one full 3D orbit at the end


def record(pipe: Pipeline, source: str, name: str, seconds: float):
    cap = open_source(source)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    stride = max(1, round(src_fps / 30))          # cap output at ~30 fps
    fps = src_fps / stride
    max_frames = int(fps * seconds)
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"{name}.mp4")

    # H.264 via ffmpeg so every browser can play it (OpenCV's mp4v is not web-safe)
    w, h = config.OUTPUT_SIZE
    ffmpeg = subprocess.Popen(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
         "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", f"{fps}", "-i", "-",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", out_path],
        stdin=subprocess.PIPE)

    pipe.reset(fps)
    pipe.copilot.chain.start(name, pipe.copilot.mission)       # open the mission on Monad
    print(f"Monad flight recorder: {pipe.copilot.chain.status}")
    n = blocked = 0
    while n < max_frames:
        for _ in range(stride - 1):
            cap.grab()                            # skip frames without decoding
        ok, frame = cap.read()
        if not ok:
            break
        ffmpeg.stdin.write(pipe.process(frame).tobytes())
        blocked += pipe.last_path is None
        n += 1
        if n % 50 == 0:
            print(f"  {name}: {n}/{max_frames} frames, {pipe.fps:.1f} FPS")
    cap.release()

    # ending: the whole map top-down, then a full 3D orbit of the terrain and flight path
    final = pipe.world.render(config.OUTPUT_SIZE, title="Full terrain map: whole flight")
    for _ in range(int(fps * END_CARD_S)):
        ffmpeg.stdin.write(final.tobytes())
    cv2.imwrite(os.path.join(OUT_DIR, f"{name}_map.png"), final)
    orbit = Orbit(pipe.world, config.OUTPUT_SIZE)
    n_orbit = int(fps * ORBIT_S)
    for i in range(n_orbit):
        view = orbit.frame(i / n_orbit, "3D terrain map: whole flight")
        ffmpeg.stdin.write(view.tobytes())
        if i == n_orbit // 8:
            cv2.imwrite(os.path.join(OUT_DIR, f"{name}_map3d.png"), view)
    ffmpeg.stdin.close()
    ffmpeg.wait()
    print(f"Saved {out_path} ({n} frames, no path found in {blocked / max(n, 1):.0%})")

    # mission report from the decision log, then close the mission on Monad with the report's hash
    cp = pipe.copilot
    if cp.log:
        print("Writing mission report ...")
        report = cp.mission_report()
        if report:
            path = os.path.join(OUT_DIR, f"{name}_report.md")
            with open(path, "w", encoding="utf-8", newline="\n") as f:
                f.write(report)
            print(f"Saved {path}  ({cp.calls} copilot calls, {cp.tokens} tokens)")
        chain = cp.chain.end(report)
        if chain:
            ok = chain["head"] == chain["on_chain_head"]
            print(f"Monad mission #{chain['mission_id']}: {chain['confirmed']} decisions confirmed, "
                  f"p50 {chain['confirm_ms_p50']} ms, head {'MATCHES' if ok else 'DIFFERS FROM'} chain")
        with open(os.path.join(OUT_DIR, f"{name}_copilot_log.json"), "w", encoding="utf-8") as f:
            json.dump({"mission": cp.mission, "stats": cp.stats(), "chain": chain, "log": cp.log}, f, indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--seconds", type=float, default=30)
    ap.add_argument("--mission", help="mission brief JSON from plan_mission.py (copilot + guard enforce it)")
    ap.add_argument("--scene", choices=list(config.SCENES), default="outdoor",
                    help="indoor uses the indoor depth model (needed for corridors and rooms)")
    args = ap.parse_args()
    config.use_scene(args.scene)
    record(Pipeline(live=False, mission=load_mission(args.mission)), args.source, args.name, args.seconds)


if __name__ == "__main__":
    main()
