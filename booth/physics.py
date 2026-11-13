"""Shuttle slowdown model and speed fit (initial_plan.md §11.8).

Without gravity, quadratic drag gives  d(t) = L * ln(1 + v0 * t / L)  (L = drag length).
Over the 0.05-0.1 s we track after a smash, gravity changes the speed by < 1 %, so the
path is fitted with this model on the distance travelled along the (nearly straight) track.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares

# Drag length L in metres. Feather value from initial_plan.md §11.8.
# Nylon has not been measured yet: it uses the feather value and the result is flagged.
DRAG_LENGTH_M = {"feather": 4.6, "nylon": 4.6}
PROVISIONAL_SHUTTLES = {"nylon"}

KMH = 3.6


def distance(t, v0, L):
    """Distance travelled t seconds after impact."""
    return L * np.log1p(v0 * np.asarray(t, dtype=float) / L)


def speed_at(t, v0, L):
    return v0 / (1.0 + v0 * np.asarray(t, dtype=float) / L)


def v0_from_distance_time(d, T, L):
    """Starting speed from total distance d and flight time T: v0 = L (e^(d/L) - 1) / T."""
    return L * np.expm1(d / L) / T


@dataclass
class FitLimits:
    min_points: int = 6
    max_rms_m: float = 0.03  # residual RMS along the track
    max_rel_margin: float = 0.10  # refuse if the ± margin is wider than 10 %
    min_span_m: float = 0.6  # tracked distance must cover at least this much
    min_kmh: float = 50.0
    max_kmh: float = 480.0


@dataclass
class FitResult:
    ok: bool
    v0_kmh: float | None = None
    margin_kmh: float | None = None
    t0: float | None = None
    rms_m: float | None = None
    n_points: int = 0
    used: list[int] = field(default_factory=list)  # indices of the input points kept
    mode: str = ""  # "impact-point" or "bracket"
    reason: str = ""  # why it was refused (when ok is False)
    fit_sigma_kmh: float | None = None
    scale_rel: float = 0.0

    def summary(self) -> str:
        if not self.ok:
            return f"NO RESULT: {self.reason}"
        return f"{self.v0_kmh:.0f} ± {self.margin_kmh:.0f} km/h"


def track_coordinate(points: np.ndarray, origin: np.ndarray | None = None) -> np.ndarray:
    """Distance along the track for 2D points (metres), measured from `origin`.

    Projects onto the main direction (PCA) instead of summing step lengths, so pixel
    noise does not add up into extra distance.
    """
    pts = np.asarray(points, dtype=float)
    centre = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - centre)
    u = vt[0]
    if np.dot(pts[-1] - pts[0], u) < 0:
        u = -u
    ref = pts[0] if origin is None else np.asarray(origin, dtype=float)
    return (pts - ref) @ u


def _fit_impact(t, s, L, t0_guess, v_guess):
    """s measured from the impact point: s = L ln(1 + v0 (t - t0) / L). Params v0, t0."""

    def res(p):
        v0, t0 = p
        x = np.maximum(t - t0, -L / v0 * 0.999 + 1e-9)
        return distance(x, v0, L) - s

    r = least_squares(res, [v_guess, t0_guess], method="lm")
    return r


def _fit_bracket(t, s, L, t0, v_guess):
    """Impact point unknown, impact time fixed: s = s0 + L ln(1 + v0 (t - t0) / L). Params v0, s0."""

    def res(p):
        v0, s0 = p
        return s0 + distance(t - t0, v0, L) - s

    return least_squares(res, [v_guess, s[0] - v_guess * (t[0] - t0)], method="lm")


def _sigma(r, n_params):
    n = len(r.fun)
    dof = max(n - n_params, 1)
    s2 = float(r.fun @ r.fun) / dof
    try:
        cov = np.linalg.inv(r.jac.T @ r.jac) * s2
    except np.linalg.LinAlgError:
        return float("inf")
    return float(np.sqrt(max(cov[0, 0], 0.0)))


def fit_speed(
    times,
    s,
    L: float,
    *,
    impact_known: bool,
    t0_bracket: tuple[float, float],
    scale_rel: float = 0.0,
    limits: FitLimits | None = None,
) -> FitResult:
    """Fit v0 to track positions.

    times, s: timestamps (s) and distance along the track (m) of the shuttle after impact.
    impact_known: True if s is measured from the impact point (s = 0 at impact); then the
        impact time is fitted, and must fall inside t0_bracket.
    t0_bracket: (last time the shuttle was not yet flying, first time it was). If the impact
        point is unknown, v0 is fitted at both ends and the middle of this window and the
        spread is added to the margin.
    scale_rel: relative systematic scale uncertainty (e.g. shuttle off the hitting plane).
    Margin = sqrt((2 sigma_fit)^2 + (scale_rel v0)^2 + bracket spread^2).
    """
    lim = limits or FitLimits()
    t = np.asarray(times, dtype=float)
    s = np.asarray(s, dtype=float)
    idx = np.arange(len(t))
    if len(t) < lim.min_points:
        return FitResult(False, n_points=len(t), reason=f"only {len(t)} shuttle positions (need {lim.min_points})")
    order = np.argsort(t)
    t, s, idx = t[order], s[order], idx[order]
    span = s[-1] - s[0] if not impact_known else s[-1]
    if span < lim.min_span_m:
        return FitResult(False, n_points=len(t), reason=f"track too short ({span:.2f} m)")
    v_guess = max((s[-1] - s[0]) / max(t[-1] - t[0], 1e-6), 5.0) * 1.3
    lo, hi = t0_bracket
    mid = 0.5 * (lo + hi)

    def run(tt, ss):
        if impact_known:
            r = _fit_impact(tt, ss, L, mid, v_guess)
            return r, r.x[0], _sigma(r, 2), r.x[1], 0.0
        fits = [_fit_bracket(tt, ss, L, t0, v_guess) for t0 in (lo, mid, hi)]
        vs = [f.x[0] for f in fits]
        return fits[1], vs[1], _sigma(fits[1], 2), mid, (max(vs) - min(vs)) / 2

    keep = np.ones(len(t), bool)
    for _ in range(3):  # drop up to 2 outliers
        r, v0, sig, t0, spread = run(t[keep], s[keep])
        res = np.abs(r.fun)
        robust = 1.4826 * np.median(res) + 1e-4
        worst = int(np.argmax(res))
        if res[worst] > max(4 * robust, 0.02) and keep.sum() > lim.min_points:
            keep[np.flatnonzero(keep)[worst]] = False
            continue
        break

    rms = float(np.sqrt(np.mean(r.fun**2)))
    v_kmh = v0 * KMH
    scale_err = 0.0
    if scale_rel > 0:  # refit with distances scaled up/down: drag makes the effect larger than linear
        v_hi = run(t[keep], s[keep] * (1 + scale_rel))[1]
        v_lo = run(t[keep], s[keep] * (1 - scale_rel))[1]
        scale_err = max(abs(v_hi - v0), abs(v0 - v_lo))
    margin = KMH * float(np.sqrt((2 * sig) ** 2 + scale_err**2 + spread**2))
    out = FitResult(
        True, v_kmh, margin, float(t0), rms, int(keep.sum()), sorted(idx[keep].tolist()),
        "impact-point" if impact_known else "bracket", fit_sigma_kmh=sig * KMH, scale_rel=scale_rel,
    )
    frame = hi - lo
    if impact_known and not (lo - 0.5 * frame <= t0 <= hi + 0.5 * frame):
        out.ok, out.reason = False, f"fitted impact time is outside the impact frames ({t0 * 1000:.1f} ms)"
    elif rms > lim.max_rms_m:
        out.ok, out.reason = False, f"positions don't follow a smash path (rms {rms * 100:.1f} cm)"
    elif not np.isfinite(margin) or margin > lim.max_rel_margin * v_kmh:
        out.ok, out.reason = False, f"margin too wide (±{margin:.0f} km/h on {v_kmh:.0f})"
    elif not lim.min_kmh <= v_kmh <= lim.max_kmh:
        out.ok, out.reason = False, f"{v_kmh:.0f} km/h is outside the plausible range"
    return out
