"""Video sources: laptop webcam, a local video file, or a YouTube URL."""
import os
import subprocess
import sys

import cv2

import config


def download_youtube(url: str, out_dir: str = "videos") -> str:
    """Download a YouTube clip (<=720p) once and return its local path."""
    os.makedirs(out_dir, exist_ok=True)
    cmd = [sys.executable, "-m", "yt_dlp",
           "-f", "bv*[height<=720][ext=mp4]/bv*[height<=720]/best",   # video only: audio unused
           "-o", os.path.join(out_dir, "%(id)s.%(ext)s"),
           "--print", "after_move:filepath", "--no-simulate", url]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return out.stdout.strip().splitlines()[-1]


def open_source(source: str | None) -> cv2.VideoCapture:
    """None -> webcam, http(s) URL -> YouTube download, anything else -> file path."""
    if source is None:
        cap = cv2.VideoCapture(config.CAMERA_INDEX, cv2.CAP_DSHOW)
    else:
        path = download_youtube(source) if source.startswith("http") else source
        cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source or 'webcam'}")
    return cap
