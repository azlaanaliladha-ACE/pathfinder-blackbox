"""Local demo site.

    python -m web.app      ->  http://localhost:5000
"""
import json
import os
import threading

import cv2
import markdown
from flask import Flask, Response, jsonify, render_template, request

from terrain_nav.pipeline import Pipeline
from terrain_nav.sources import open_source

HERE = os.path.dirname(__file__)
VIDEO_DIR = os.path.join(HERE, "static", "videos")

DEMOS = [  # file name (without .mp4) -> caption shown on the page
    ("demo4", "FPV drone flyover"),
]

app = Flask(__name__)
_pipe: Pipeline | None = None
_lock = threading.Lock()


def pipeline() -> Pipeline:
    global _pipe
    with _lock:
        if _pipe is None:
            _pipe = Pipeline()
    return _pipe


def _debrief(name: str) -> dict | None:
    """Mission brief, Nemotron decision timeline and report saved by record.py for one run."""
    log_path = os.path.join(VIDEO_DIR, f"{name}_copilot_log.json")
    if not os.path.exists(log_path):
        return None
    with open(log_path, encoding="utf-8") as f:
        data = json.load(f)
    rep_path = os.path.join(VIDEO_DIR, f"{name}_report.md")
    report = open(rep_path, encoding="utf-8").read() if os.path.exists(rep_path) else ""
    mission = data.get("mission")
    return {"stats": data.get("stats", {}), "mission": mission, "log": data["log"], "chain": data.get("chain"),
            "brief_html": markdown.markdown(_brief_md(mission)) if mission else "",
            "report_html": markdown.markdown(report)}


def _brief_md(mission: dict) -> str:
    from terrain_nav.mission import brief_markdown
    return brief_markdown(mission)


@app.route("/")
def index():
    demos = [(n, cap) for n, cap in DEMOS if os.path.exists(os.path.join(VIDEO_DIR, f"{n}.mp4"))]
    return render_template("index.html", demos=demos, debrief=_debrief(demos[0][0]) if demos else None)


@app.route("/api/plan", methods=["POST"])
def api_plan():
    """Preflight from the browser: Nemotron + Tavily mission brief."""
    from terrain_nav.mission import brief_markdown, plan_mission
    body = request.get_json(force=True)
    try:
        brief = plan_mission(body["goal"], body["site"], log=lambda *_: None)
    except Exception as e:
        return jsonify(error=f"{type(e).__name__}: {e}"), 502
    return jsonify(brief=brief, html=markdown.markdown(brief_markdown(brief)))


def _webcam_frames():
    pipe, cap = pipeline(), open_source(None)
    pipe.reset()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            with _lock:
                out = pipe.process(frame)
            _, jpg = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 80])
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n"
    finally:
        cap.release()


@app.route("/live")
def live():
    return Response(_webcam_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, threaded=True)
