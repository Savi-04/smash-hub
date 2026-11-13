"""Smash speed from one side-on 240 fps clip over the calibration mat (beta, no ML).

  python speed.py CLIP                 analyse; result goes to the leaderboard's pending list
  python speed.py CLIP --check         setup check only: mat found? camera distance? frame rate?
  python speed.py CLIP --manual        skip automatic tracking, click the shuttle yourself
  python speed.py CLIP --clicks F.json re-run the manual fit from saved clicks (no window)

If automatic tracking fails, the click window opens so the booth always gets a result.
"""
from __future__ import annotations

import argparse
import json
import secrets
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

import geometry
import manual
import physics
import track
import video
from geometry import CameraPose, detect_markers, estimate_pose

HERE = Path(__file__).parent
DATA = HERE / "data"
LAST_POSE = DATA / "last_pose.json"
MAX_CACHE_BYTES = 700e6  # keep decoded frames in memory below this, else decode twice


class BoothError(Exception):
    pass


def find_pose(sampled: list[tuple[int, np.ndarray]], size, use_saved: bool = True):
    best = None
    for _, g in sampled[:15]:
        m = detect_markers(g)
        if len(m) >= 2:
            p = estimate_pose(m, size)
            if p and (best is None or (p.n_markers, -p.reproj_px) > (best.n_markers, -best.reproj_px)):
                best = p
        if best and best.n_markers == 4:
            break
    if best:
        best.save(LAST_POSE)
        return best, "mat"
    if use_saved and LAST_POSE.exists():
        p = CameraPose.from_json(json.loads(LAST_POSE.read_text()))
        if tuple(p.image_size) == tuple(size):
            return p, "saved (mat not visible in this clip; using the last good setup)"
    raise BoothError("calibration mat not found (need 2+ markers). Check the setup with --check.")


def fit_points(info, pose, frames_idx, xz, contact, scale_rel, L, limits, audio=False, av_offset=0.0):
    """Shared by auto and manual: contact = (frame, xz) of the shuttle at contact, or None."""
    ts = info.timestamps
    frames_idx, xz = list(frames_idx), np.asarray(xz, float)
    notes = []
    # The track may start with the shuttle still at contact (it hasn't left yet): treat that point as the contact.
    if contact is None and len(xz) > 3:
        v01 = np.linalg.norm(xz[1] - xz[0]) / (ts[frames_idx[1]] - ts[frames_idx[0]])
        v12 = np.linalg.norm(xz[2] - xz[1]) / (ts[frames_idx[2]] - ts[frames_idx[1]])
        if v01 < 0.5 * v12:
            contact = (frames_idx[0], xz[0])
    if contact is not None:
        while len(xz) > 1 and np.linalg.norm(xz[0] - contact[1]) < 0.03:
            contact = (frames_idx[0], xz[0])
            frames_idx, xz = frames_idx[1:], xz[1:]
    t = ts[np.asarray(frames_idx)]
    if contact is not None:
        fc, pc = contact
        s = physics.track_coordinate(xz, origin=pc)
        r = physics.fit_speed(t, s, L, impact_known=True, t0_bracket=(ts[fc], ts[frames_idx[0]]),
                              scale_rel=scale_rel, limits=limits)
        imp = pc
    else:
        f1 = frames_idx[0]
        lo, hi = ts[max(f1 - 1, 0)], ts[f1]
        imp = xz[0]
        if audio:
            delay = np.linalg.norm(pose.centre - [imp[0], 0, imp[1]]) / 343.0 + av_offset
            ta = video.audio_onset(info, lo - 2 / info.fps, hi + 2 / info.fps, delay)
            if ta is not None and lo - 1 / info.fps <= ta <= hi:
                lo, hi = ta - 3e-4, ta + 3e-4
                notes.append(f"impact time from the hit sound ({ta * 1000:.1f} ms)")
        s = physics.track_coordinate(xz)
        r = physics.fit_speed(t, s, L, impact_known=False, t0_bracket=(lo, hi), scale_rel=scale_rel, limits=limits)
    if audio and contact is not None and r.ok:
        delay = np.linalg.norm(pose.centre - [imp[0], 0, imp[1]]) / 343.0 + av_offset
        ta = video.audio_onset(info, r.t0 - 3 / info.fps, r.t0 + 3 / info.fps, delay)
        if ta is not None:
            notes.append(f"hit sound vs video impact: {(ta - r.t0) * 1000:+.1f} ms")
    r.samples = (t, s)  # for the result chart
    return r, notes


def _curve(result, n=40) -> dict | None:
    """Measured points + fitted speed curve, relative to the impact, for the app's chart."""
    if result is None or not result.ok or not hasattr(result, "samples"):
        return None
    t, s = result.samples
    v0, L = result.v0_kmh / physics.KMH, result.drag_length
    tt = np.asarray(t)[result.used] - result.t0
    ss = np.asarray(s)[result.used]
    if result.mode == "bracket":  # s has an unknown offset: line it up with the fitted curve
        ss = ss - np.mean(ss - physics.distance(tt, v0, L))
    grid = np.linspace(0, max(float(tt.max()), 1e-3), n)
    return {"t_ms": [round(float(x) * 1000, 2) for x in tt], "d_m": [round(float(x), 3) for x in ss],
            "fit_t_ms": [round(float(x) * 1000, 2) for x in grid],
            "fit_d_m": [round(float(x), 3) for x in physics.distance(grid, v0, L)],
            "fit_kmh": [round(float(x) * physics.KMH, 1) for x in physics.speed_at(grid, v0, L)]}


def analyse(clip: Path, *, shuttle="feather", drag_length=None, capture_fps=None, force_manual=False,
            clicks_file=None, clicks=None, allow_window=True, audio=False, av_offset_ms=0.0, depth_tol=0.15,
            check_only=False, debug_image=False, mode="mat", distance_m=None, stabilise=True,
            progress=None) -> dict:
    """mode="mat": Verified (printed mat in view). mode="estimate": no mat, `distance_m` from phone to hitting line.

    clicks: [(frame, u, v)] in original pixels (contact first) instead of automatic tracking.
    progress: optional callback(stage: str, fraction: float) for the app.
    """
    say = progress or (lambda *_: None)
    t_start = time.time()
    if mode not in ("mat", "estimate"):
        raise BoothError(f"unknown mode {mode}")
    if mode == "estimate" and not (distance_m and 1.0 <= distance_m <= 12.0):
        raise BoothError("no-mat estimate needs the phone-to-hitting-line distance (1-12 m)")
    say("Reading the video", 0.05)
    info = video.probe(clip)
    if capture_fps:
        video.override_capture_fps(info, capture_fps)
    if not info.timing_ok:
        raise BoothError(info.timing_note)
    size = (info.width, info.height)
    n = len(info.timestamps)
    keep_all = n * info.width * info.height <= MAX_CACHE_BYTES
    step = max(1, n // 40)
    stab = track.Stabiliser() if stabilise else None
    shifts, all_frames, sampled = [], [], []
    for i, g in video.frames(clip):
        d = stab.shift(g) if stab else (0.0, 0.0)
        shifts.append(d)
        if stab:
            g = stab.apply(g, d)
        if keep_all:
            all_frames.append(g)
        if i % step == 0:
            sampled.append((i, g))
        if i % 50 == 0:
            say("Reading the video", 0.05 + 0.35 * min(i / max(n, 1), 1))
    n = min(n, len(shifts))
    info.timestamps = info.timestamps[:n]
    shake_px = float(np.max(np.linalg.norm(np.array(shifts), axis=1))) if shifts else 0.0
    if mode == "mat":
        say("Finding the mat", 0.42)
        pose, pose_src = find_pose(sampled, size)
        lens = "mat"
    else:
        K = None
        if LAST_POSE.exists():
            saved = CameraPose.from_json(json.loads(LAST_POSE.read_text()))
            if tuple(saved.image_size) == tuple(size):
                K = saved.K
        pose = geometry.pose_from_distance(size, distance_m, K=K)
        pose_src = "distance entered"
        lens = "calibrated" if K is not None else "nominal"
    res = {
        "clip": clip.name, "fps": round(info.fps, 1), "timing": info.timing_note, "pose_source": pose_src,
        "mode": mode, "camera_distance_m": round(pose.plane_distance, 2), "camera_height_m": round(float(pose.centre[2]), 2),
        "markers": pose.n_markers, "reproj_px": None if mode == "estimate" else round(pose.reproj_px, 2),
        "px_per_m": round(pose.px_per_m(), 1), "shake_px": round(shake_px, 1),
    }
    if check_only:
        problems = []
        if info.fps < 200:
            problems.append(f"frame rate {info.fps:.0f} fps (want 240)")
        if mode == "mat" and pose.n_markers < 4:
            problems.append(f"only {pose.n_markers}/4 mat markers visible")
        if not 2.5 <= pose.plane_distance <= 5.0:
            problems.append(f"phone {pose.plane_distance:.1f} m from the hitting line (want 3-4 m)")
        if shake_px > 40:
            problems.append(f"camera moved {shake_px:.0f} px: use a stand")
        res["status"] = "setup ok" if not problems and pose_src in ("mat", "distance entered") else "fix setup"
        res["problems"] = problems
        return res

    L = drag_length or physics.DRAG_LENGTH_M[shuttle]
    if mode == "mat":
        scale_rel = depth_tol / pose.plane_distance
        limits, limits_manual = physics.FitLimits(), physics.FitLimits(min_points=4)
    else:  # distance guessed within ~12 %; lens ~6 % (nominal) or ~4 % (from an earlier mat clip)
        scale_rel = float(np.hypot(0.12, 0.06 if lens == "nominal" else 0.04))
        scale_rel = float(np.hypot(scale_rel, depth_tol / pose.plane_distance))
        limits = physics.FitLimits(max_rel_margin=0.25)
        limits_manual = physics.FitLimits(min_points=4, max_rel_margin=0.25)
    result, notes, method, tr = None, [], "auto", None
    if not force_manual and not clicks_file and not clicks:
        say("Finding the shuttle", 0.5)
        bg = track.background([g for _, g in sampled])
        z_range = (0.15, 3.6) if mode == "mat" else (-1.5, 4.5)
        roi = track.plane_roi(pose, z_range=z_range)
        prm = track.adapt(track.DetectParams(), [g for _, g in sampled], bg)
        ppm = pose.px_per_m()
        if keep_all:
            src = all_frames
        else:
            src = (track.Stabiliser.apply(g, shifts[i]) if stab else g for i, g in video.frames(clip))
        cands = []
        for i, g in enumerate(src):
            if i >= n:
                break
            cands.append(track.candidates(g, bg, roi, pose, prm, ppm))
            if i % 50 == 0:
                say("Finding the shuttle", 0.5 + 0.3 * i / max(n, 1))
        say("Tracking the flight", 0.82)
        tr = track.find_track(cands, info.timestamps)
        if tr is None:
            notes.append("automatic tracking: no smash track found")
        else:
            say("Fitting the flight", 0.9)
            contact = (tr.impact_frame, tr.impact_xz) if tr.impact_xz is not None else None
            result, more = fit_points(info, pose, tr.frames, tr.xz, contact, scale_rel, L, limits,
                                      audio, av_offset_ms / 1000)
            notes += more
            if not result.ok:
                notes.append(f"automatic fit refused: {result.reason}")
    if result is None or not result.ok:
        if clicks_file:
            clicks = manual.load_clicks(clicks_file)
        elif not clicks and allow_window:
            print("Automatic tracking didn't give a result: opening the click window.", file=sys.stderr)
            start = (tr.frames[0] - 2) if tr else 0
            clicks = manual.collect_clicks(clip, start)
            if clicks:
                manual.save_clicks(DATA / "clicks" / f"{clip.stem}.json", clicks)
        if clicks:
            method = "manual"
            clicks = sorted(clicks)
            if len(clicks) < 5:
                raise BoothError(f"need the contact tap + at least 4 more (got {len(clicks)})")
            fidx = [int(c[0]) for c in clicks]
            if max(fidx) >= n:
                raise BoothError("a tap is outside the video")
            uv = np.array([[c[1] - shifts[f][0], c[2] - shifts[f][1]] for f, c in zip(fidx, clicks)])
            xz = pose.to_plane(uv)
            result, more = fit_points(info, pose, fidx[1:], xz[1:], (fidx[0], xz[0]), scale_rel, L,
                                      limits_manual, audio, av_offset_ms / 1000)
            notes += more
    if result is None:
        raise BoothError("; ".join(notes) or "no result")
    if debug_image and tr is not None and keep_all:
        _debug_image(all_frames[tr.frames[0]], tr, clip)
    result.drag_length = L
    say("Done", 1.0)
    res.update({
        "status": "ok" if result.ok else "refused", "method": method, "speed_kmh": _r(result.v0_kmh),
        "margin_kmh": _r(result.margin_kmh), "reason": result.reason, "points": result.n_points,
        "fit_mode": result.mode, "rms_cm": _r(result.rms_m * 100 if result.rms_m is not None else None, 1),
        "shuttle": shuttle, "drag_length_m": L, "provisional_drag": shuttle in physics.PROVISIONAL_SHUTTLES and not drag_length,
        "lens": lens, "beta": True, "notes": notes, "seconds": round(time.time() - t_start, 1),
        "suggested_frame": int(tr.frames[0] - 2) if tr else None, "frames": n,
        "curve": _curve(result) if result.ok else None,
    })
    return res


def _r(x, nd=0):
    return None if x is None else round(float(x), nd)


def _debug_image(gray, tr, clip):
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for u, v in tr.uv:
        cv2.circle(img, (int(u), int(v)), 8, (0, 0, 255), 2)
    out = HERE / "out" / "debug" / f"{clip.stem}.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), img)
    print(f"debug image (contains the player; delete after checking): {out}", file=sys.stderr)


def add_pending(res: dict) -> Path:
    """Hand the result to the leaderboard (the operator adds the nickname + consent there)."""
    rid = datetime.now().strftime("%Y%m%d-%H%M%S-") + secrets.token_hex(3)
    entry = {"id": rid, "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
             **{k: res[k] for k in ("speed_kmh", "margin_kmh", "method", "shuttle", "provisional_drag", "clip")}}
    d = DATA / "pending"
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / f".{rid}.tmp"
    tmp.write_text(json.dumps(entry))
    tmp.replace(d / f"{rid}.json")
    return d / f"{rid}.json"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clip", type=Path)
    ap.add_argument("--check", action="store_true", help="setup check only")
    ap.add_argument("--shuttle", choices=sorted(physics.DRAG_LENGTH_M), default="feather")
    ap.add_argument("--drag-length", type=float, help="override the drag length L in metres")
    ap.add_argument("--capture-fps", type=float, help="real capture rate, if the clip's timestamps were re-timed")
    ap.add_argument("--manual", action="store_true", help="click the shuttle yourself")
    ap.add_argument("--clicks", type=Path, help="saved clicks JSON (manual fit without a window)")
    ap.add_argument("--no-window", action="store_true", help="never open the click window")
    ap.add_argument("--audio", action="store_true", help="refine the impact time from the hit sound")
    ap.add_argument("--av-offset-ms", type=float, default=0.0, help="phone audio/video offset (clap test)")
    ap.add_argument("--depth-tol", type=float, default=0.15, help="how far (m) the shuttle may be off the hitting line")
    ap.add_argument("--no-pending", action="store_true", help="don't send the result to the leaderboard")
    ap.add_argument("--debug-image", action="store_true", help="save a frame with the track drawn (contains the player)")
    ap.add_argument("--json", action="store_true", help="print the full result as JSON")
    ap.add_argument("--no-mat", type=float, metavar="METRES", dest="distance",
                    help="no mat in view: estimate from this phone-to-hitting-line distance (wider margin)")
    a = ap.parse_args()
    try:
        res = analyse(a.clip, shuttle=a.shuttle, drag_length=a.drag_length, capture_fps=a.capture_fps,
                      force_manual=a.manual, clicks_file=a.clicks, allow_window=not a.no_window, audio=a.audio,
                      av_offset_ms=a.av_offset_ms, depth_tol=a.depth_tol, check_only=a.check,
                      debug_image=a.debug_image, mode="estimate" if a.distance else "mat",
                      distance_m=a.distance)
    except BoothError as e:
        print(f"NO RESULT: {e}")
        return 2
    if a.json:
        print(json.dumps(res, indent=1))
    if a.check:
        print(f"{res['status'].upper()}: {res['markers']}/4 markers ({res['pose_source']}), phone {res['camera_distance_m']} m "
              f"from the line at {res['camera_height_m']} m high, {res['fps']} fps, {res['px_per_m']} px/m")
        for p in res["problems"]:
            print(f"  - {p}")
        return 0 if res["status"] == "setup ok" else 1
    for note in res["notes"]:
        print(f"  note: {note}")
    if res["status"] != "ok":
        print(f"NO RESULT ({res['method']}): {res['reason']}")
        return 2
    flag = " (nylon drag not yet measured)" if res["provisional_drag"] else ""
    print(f"SPEED (beta): {res['speed_kmh']:.0f} ± {res['margin_kmh']:.0f} km/h  [{res['method']}, "
          f"{res['points']} points, {res['seconds']} s]{flag}")
    if not a.no_pending:
        add_pending(res)
        print("  -> sent to the leaderboard: add the nickname on the operator page")
    return 0


if __name__ == "__main__":
    sys.exit(main())
