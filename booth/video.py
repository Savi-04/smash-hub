"""Video input: real frame timestamps (ffprobe), slow-motion re-timing check, frames, audio."""
from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

HIGH_SPEED_FPS = 100.0  # timestamps faster than this are treated as real capture time
TOOL_TIMEOUT = 120  # seconds; a malformed file must not hang ffprobe/ffmpeg forever


def _need(tool: str) -> str:
    path = shutil.which(tool)
    if not path:
        raise RuntimeError(f"'{tool}' not found. Install FFmpeg first: brew install ffmpeg")
    return path


@dataclass
class ClipInfo:
    path: Path
    width: int
    height: int
    timestamps: np.ndarray  # presentation time of every video frame, seconds
    has_audio: bool
    audio_start: float
    tags: dict = field(default_factory=dict)
    timing_note: str = ""
    timing_ok: bool = True
    retimed: bool = False

    @property
    def fps(self) -> float:
        return 1.0 / float(np.median(np.diff(self.timestamps)))


def probe(path: Path) -> ClipInfo:
    """Read stream info and the real per-frame timestamps (from packets, sorted into display order)."""
    ffprobe = _need("ffprobe")
    meta = json.loads(subprocess.run(
        [ffprobe, "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
        capture_output=True, text=True, check=True, timeout=TOOL_TIMEOUT).stdout)
    vs = next(s for s in meta["streams"] if s["codec_type"] == "video")
    aud = [s for s in meta["streams"] if s["codec_type"] == "audio"]
    out = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries", "packet=pts_time",
         "-of", "csv=p=0", str(path)], capture_output=True, text=True, check=True, timeout=TOOL_TIMEOUT).stdout
    ts = np.sort(np.array([float(x) for x in out.split() if x.strip() not in ("", "N/A")]))
    tags = {**meta.get("format", {}).get("tags", {}), **vs.get("tags", {})}
    info = ClipInfo(Path(path), int(vs["width"]), int(vs["height"]), ts, bool(aud),
                    float(aud[0].get("start_time", 0.0)) if aud else 0.0, tags)
    check_timing(info)
    return info


def check_timing(info: ClipInfo) -> None:
    """Flag clips whose timestamps are not real capture time.

    An iPhone slo-mo original keeps ~240 fps timestamps. A clip that was rendered/exported
    as slow motion plays the 240 fps frames at ~30 fps, so its timestamps are ~8x too slow.
    We can't always tell which one we got, so: >= 100 fps -> trusted; mixed frame rates ->
    re-timed segments; < 100 fps -> needs --capture-fps from the operator.
    """
    ts = info.timestamps
    if len(ts) < 3:
        info.timing_ok, info.timing_note = False, "clip has fewer than 3 frames"
        return
    dt = np.diff(ts)
    med = float(np.median(dt))
    fps = 1.0 / med
    slow = dt > 2.5 * med  # gaps: dropped frames, or a re-timed (stretched) part
    fast = dt < 0.4 * med
    notes = [f"timestamps: {fps:.1f} fps median, {len(ts)} frames"]
    if fast.sum() > 2 or slow.sum() > max(3, 0.05 * len(dt)):
        info.retimed = True
        notes.append(f"mixed frame intervals ({fast.sum()} short, {slow.sum()} long): parts of this clip were re-timed")
    elif slow.any():
        notes.append(f"{int(slow.sum())} dropped-frame gap(s); real timestamps handle them")
    if fps < HIGH_SPEED_FPS:
        info.retimed = True  # can't trust these as capture time
        notes.append("below 100 fps: either a normal-speed video or a slow-motion export with stretched time")
    if info.retimed:
        info.timing_ok = False
        notes.append("send the original slo-mo clip (AirDrop: Options > All Photos Data) or pass --capture-fps 240")
    info.timing_note = "; ".join(notes)


def override_capture_fps(info: ClipInfo, fps: float) -> None:
    """Operator states the capture rate: use evenly spaced timestamps at that rate."""
    info.timestamps = info.timestamps[0] + np.arange(len(info.timestamps)) / fps
    info.timing_ok, info.retimed = True, False
    info.timing_note = f"timestamps replaced by operator setting: {fps:g} fps capture (assumes no dropped frames)"


def frames(path: Path, gray: bool = True):
    """Yield (index, frame) in display order."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video {path}")
    i = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield i, (cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if gray else frame)
            i += 1
    finally:
        cap.release()


def read_audio(path: Path, rate: int = 48000) -> np.ndarray:
    ffmpeg = _need("ffmpeg")
    raw = subprocess.run([ffmpeg, "-v", "error", "-i", str(path), "-map", "0:a:0", "-ac", "1", "-ar", str(rate),
                          "-f", "f32le", "-"], capture_output=True, check=True, timeout=TOOL_TIMEOUT).stdout
    return np.frombuffer(raw, dtype=np.float32)


def audio_onset(info: ClipInfo, t_lo: float, t_hi: float, delay: float, rate: int = 48000) -> float | None:
    """Time (video clock) of a sharp hit sound in [t_lo, t_hi] after removing `delay`.

    Returns None if there is no clear spike (noisy hall, no audio).
    """
    if not info.has_audio:
        return None
    a = read_audio(info.path, rate)
    if len(a) < rate // 10:
        return None
    env = np.abs(np.diff(a, prepend=a[0]))
    k = max(1, rate // 2000)  # 0.5 ms smoothing
    env = np.convolve(env, np.ones(k) / k, mode="same")
    i0 = max(0, int((t_lo + delay - info.audio_start) * rate))
    i1 = min(len(env), int((t_hi + delay - info.audio_start) * rate) + 1)
    if i1 - i0 < 2:
        return None
    peak = i0 + int(np.argmax(env[i0:i1]))
    ctx = env[max(0, peak - rate // 10): peak + rate // 10]
    if env[peak] < 10 * (np.median(ctx) + 1e-9):
        return None
    onset = peak
    while onset > i0 and env[onset - 1] > 0.5 * env[peak]:
        onset -= 1
    return info.audio_start + onset / rate - delay
