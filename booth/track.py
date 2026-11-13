"""Shuttle detection (bright blob vs. background, no ML) and the post-impact track."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from geometry import CameraPose

MIN_SPEED, MAX_SPEED = 11.0, 140.0  # m/s (40-500 km/h) for the outgoing smash


@dataclass
class DetectParams:
    diff_thresh: int = 35  # brighter than the background by this much (raised automatically in noisy clips)
    bright_thresh: int = 90  # and at least this bright (white shuttle; dim halls still pass)
    max_candidates: int = 40


class Stabiliser:
    """Undo small camera shake (hand-held or wobbly stand) by shifting every frame onto the first one."""

    def __init__(self, scale: float = 0.25):
        self.scale = scale
        self.ref = None
        self.window = None

    def _small(self, gray):
        s = cv2.resize(gray, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA).astype(np.float32)
        if self.window is None:
            self.window = cv2.createHanningWindow(s.shape[::-1], cv2.CV_32F)
        return s

    def shift(self, gray) -> tuple[float, float]:
        s = self._small(gray)
        if self.ref is None:
            self.ref = s
            return 0.0, 0.0
        (dx, dy), _ = cv2.phaseCorrelate(self.ref.copy(), s.copy(), self.window)  # it can modify its inputs
        if abs(dx) > 0.1 * s.shape[1] or abs(dy) > 0.1 * s.shape[0]:
            return 0.0, 0.0
        # Accept the shift only if it really lines the frame up better (a moving shuttle or player can fool it).
        M = np.float32([[1, 0, -dx], [0, 1, -dy]])
        moved = cv2.warpAffine(s, M, s.shape[::-1], borderMode=cv2.BORDER_REPLICATE)
        if np.mean(np.abs(moved - self.ref)) > 0.9 * np.mean(np.abs(s - self.ref)):
            return 0.0, 0.0
        return dx / self.scale, dy / self.scale

    @staticmethod
    def apply(gray, dxdy):
        dx, dy = dxdy
        if abs(dx) < 0.25 and abs(dy) < 0.25:
            return gray
        M = np.float32([[1, 0, -dx], [0, 1, -dy]])
        return cv2.warpAffine(gray, M, gray.shape[::-1], flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def adapt(prm: DetectParams, gray_frames: list[np.ndarray], bg: np.ndarray) -> DetectParams:
    """Raise the difference threshold if the clip is noisy (grain in dim light, compression)."""
    d = np.concatenate([(f.astype(np.int16) - bg)[::7, ::7].ravel() for f in gray_frames[:10]])
    sigma = 1.4826 * float(np.median(np.abs(d - np.median(d))))
    return DetectParams(max(prm.diff_thresh, int(6 * sigma)), prm.bright_thresh, prm.max_candidates)


def background(gray_frames: list[np.ndarray]) -> np.ndarray:
    """Per-pixel median of frames spread over the clip: removes the moving shuttle and player."""
    return np.median(np.stack(gray_frames), axis=0).astype(np.uint8)


def plane_roi(pose: CameraPose, z_range=(0.15, 3.6), x_range=(-2.5, 7.0)) -> np.ndarray:
    """Pixels whose ray meets the hitting plane at a sensible height (excludes floor and mat)."""
    w, h = pose.image_size
    step = 4
    us, vs = np.meshgrid(np.arange(0, w, step), np.arange(0, h, step))
    with np.errstate(divide="ignore", invalid="ignore"):
        p = pose.to_plane(np.stack([us.ravel(), vs.ravel()], axis=1))
    rays = np.linalg.inv(pose.K) @ np.vstack([us.ravel(), vs.ravel(), np.ones(us.size)])
    in_front = (pose.R.T @ rays)[1] * -pose.centre[1] > 0
    ok = in_front & (p[:, 1] > z_range[0]) & (p[:, 1] < z_range[1]) & (p[:, 0] > x_range[0]) & (p[:, 0] < x_range[1])
    small = ok.reshape(us.shape).astype(np.uint8) * 255
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST) > 0


def candidates(gray, bg, roi, pose: CameraPose, prm: DetectParams, ppm: float) -> np.ndarray:
    """Bright moving blobs in one frame -> array of (X, Z, u, v, area) in plane metres / pixels."""
    diff = cv2.subtract(gray, bg)
    mask = ((diff > prm.diff_thresh) & (gray > prm.bright_thresh) & roi).astype(np.uint8)
    n, _, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return np.zeros((0, 5))
    area = stats[1:, cv2.CC_STAT_AREA]
    max_area = (0.45 * ppm) * (0.10 * ppm)
    keep = np.flatnonzero((area >= 3) & (area <= max_area))
    if len(keep) > prm.max_candidates:
        keep = keep[np.argsort(-area[keep])[: prm.max_candidates]]
    uv = cents[1:][keep]
    xz = pose.to_plane(uv)
    return np.column_stack([xz, uv, area[keep]])


@dataclass
class Track:
    frames: list[int]
    xz: np.ndarray  # N x 2, metres on the hitting plane
    uv: np.ndarray  # N x 2, pixels
    impact_frame: int | None = None  # frame where the shuttle is seen at contact (before it flies)
    impact_xz: np.ndarray | None = None


def _extend(cands, ts, f, p, v, L=4.6, max_gap=2):
    """Follow a seed forward with a drag-aware constant-direction prediction."""
    frames, pts = [f], [p]
    last_f, last_p, vel = f, p, v
    misses = 0
    g = f + 1
    while g < len(cands) and misses <= max_gap:
        dt = ts[g] - ts[last_f]
        speed = np.linalg.norm(vel)
        step = speed * dt
        pred = last_p + vel / speed * L * np.log1p(speed * dt / L)
        c = cands[g]
        if len(c):
            dist = np.linalg.norm(c[:, :2] - pred, axis=1)
            j = int(np.argmin(dist))
            if dist[j] < 0.05 + 0.15 * step:
                new = c[j, :2]
                nv = (new - last_p) / dt
                if np.dot(nv, vel) > 0.95 * np.linalg.norm(nv) * speed and MIN_SPEED * 0.5 < np.linalg.norm(nv) < MAX_SPEED:
                    frames.append(g)
                    pts.append(new)
                    vel = 0.5 * vel + 0.5 * nv if len(frames) > 2 else nv
                    last_f, last_p, misses = g, new, 0
                    g += 1
                    continue
        misses += 1
        g += 1
    return frames, pts


def _has_next(cands, ts, g, p, v, L=4.6):
    """Cheap check: is there a candidate where a third point of this seed should be?"""
    for h in (g + 1, g + 2):
        if h >= len(cands) or not len(cands[h]):
            continue
        dt = ts[h] - ts[g]
        sp = np.linalg.norm(v)
        pred = p + v / sp * L * np.log1p(sp * dt / L)
        if np.min(np.linalg.norm(cands[h][:, :2] - pred, axis=1)) < 0.08 + 0.2 * sp * dt:
            return True
    return False


def find_track(cands: list[np.ndarray], ts: np.ndarray, min_len: int = 6) -> Track | None:
    """Longest fast, straight-ish, not-rising track = the smash; plus the contact position before it."""
    best = None
    used: set[tuple[int, int]] = set()  # (frame, candidate) already part of a found track
    for f in range(len(cands) - 1):
        if not len(cands[f]):
            continue
        for ia, a in enumerate(cands[f]):
            if (f, ia) in used:
                continue
            for g in (f + 1, f + 2):
                if g >= len(cands) or not len(cands[g]):
                    continue
                dt = ts[g] - ts[f]
                vs = (cands[g][:, :2] - a[:2]) / dt
                sp = np.linalg.norm(vs, axis=1)
                ok = (sp > MIN_SPEED) & (sp < MAX_SPEED) & (vs[:, 1] <= 0.3 * sp)
                for v in vs[ok]:
                    if not _has_next(cands, ts, g, a[:2] + v * dt, v):
                        continue
                    fr, pts = _extend(cands, ts, f, a[:2], v)
                    if len(fr) >= min_len and (best is None or len(fr) > len(best[0])):
                        best = (fr, pts)
                        used |= {(fr_i, int(np.argmin(np.linalg.norm(cands[fr_i][:, :2] - p, axis=1))))
                                 for fr_i, p in zip(fr, pts)}
    if best is None:
        return None
    fr, pts = best
    xz = np.array(pts)
    uv = np.array([cands[g][np.argmin(np.linalg.norm(cands[g][:, :2] - p, axis=1)), 2:4] for g, p in zip(fr, pts)])
    tr = Track(fr, xz, uv)
    _find_contact(tr, cands)
    return tr


def _find_contact(tr: Track, cands) -> None:
    """Shuttle seen 1-2 frames before the fast track, just behind its start = contact position."""
    u = tr.xz[-1] - tr.xz[0]
    u = u / np.linalg.norm(u)
    first_step = np.linalg.norm(tr.xz[1] - tr.xz[0]) / max(tr.frames[1] - tr.frames[0], 1)
    for back in (1, 2):
        g = tr.frames[0] - back
        if g < 0 or not len(cands[g]):
            continue
        rel = cands[g][:, :2] - tr.xz[0]
        along = rel @ u
        perp = np.abs(rel @ np.array([-u[1], u[0]]))
        ok = (along < 0.02) & (along > -1.5 * first_step) & (perp < 0.06)
        if ok.any():
            j = np.flatnonzero(ok)[np.argmax(along[ok])]
            tr.impact_frame, tr.impact_xz = g, cands[g][j, :2].copy()
            return
