"""Manual fallback: the operator clicks the shuttle in a few frames; the same fit then runs.

Click 1: the shuttle at contact (the last frame before it flies off the racket).
Clicks 2-7: the shuttle in the following frames (4-6 clicks).
Keys: d / . next frame   a / , previous   f / s jump 10   z undo   Enter finish   Esc cancel
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

import video

WIN = "Booth - click the shuttle"
MAX_W = 1600


def collect_clicks(path: Path, start_frame: int = 0) -> list[tuple[int, float, float]]:
    """Show the clip, return [(frame, u, v)] in full-resolution pixels, sorted by frame."""
    cache, scale = [], 1.0
    for _, fr in video.frames(path, gray=False):  # keep display-size JPEGs in memory only
        scale = min(1.0, MAX_W / fr.shape[1])
        small = cv2.resize(fr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else fr
        cache.append(cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 92])[1])
    if not cache:
        return []
    clicks: dict[int, tuple[float, float]] = {}
    state = {"i": int(np.clip(start_frame, 0, len(cache) - 1))}

    def on_mouse(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicks[state["i"]] = (x / scale, y / scale)
            state["i"] = min(state["i"] + 1, len(cache) - 1)

    cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WIN, on_mouse)
    try:
        while True:
            i = state["i"]
            img = cv2.imdecode(cache[i], cv2.IMREAD_COLOR)
            for f, (u, v) in clicks.items():
                c = (0, 0, 255) if f == i else (0, 200, 255)
                cv2.circle(img, (int(u * scale), int(v * scale)), 6, c, 2)
            n = len(clicks)
            msg = "Click the shuttle AT CONTACT" if n == 0 else f"{n} clicks - click the shuttle in the next frames (4-6)"
            for k, line in enumerate([f"frame {i + 1}/{len(cache)}   {msg}",
                                      "d/. next  a/, back  f/s +-10  z undo  Enter done  Esc cancel"]):
                cv2.putText(img, line, (12, 30 + 28 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
            cv2.imshow(WIN, img)
            key = cv2.waitKey(30) & 0xFF
            if key in (ord("d"), ord(".")):
                state["i"] = min(i + 1, len(cache) - 1)
            elif key in (ord("a"), ord(",")):
                state["i"] = max(i - 1, 0)
            elif key == ord("f"):
                state["i"] = min(i + 10, len(cache) - 1)
            elif key == ord("s"):
                state["i"] = max(i - 10, 0)
            elif key == ord("z") and clicks:
                last = max(clicks)
                del clicks[last]
                state["i"] = last
            elif key in (13, 10):
                break
            elif key == 27:
                clicks.clear()
                break
    finally:
        cv2.destroyWindow(WIN)
        cv2.waitKey(1)
    return [(f, u, v) for f, (u, v) in sorted(clicks.items())]


def save_clicks(path: Path, clicks) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([{"frame": f, "u": u, "v": v} for f, u, v in clicks], indent=1))


def load_clicks(path: Path) -> list[tuple[int, float, float]]:
    return sorted((int(c["frame"]), float(c["u"]), float(c["v"])) for c in json.loads(Path(path).read_text()))
