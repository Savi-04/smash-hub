"""Known-answer tests for the slowdown model and fit."""
import numpy as np
import pytest

from physics import KMH, distance, fit_speed, track_coordinate, v0_from_distance_time

L = 4.6
FPS = 240.0


def test_worked_example_212_kmh():
    # initial_plan.md §11.8: d ≈ 9.2 m, T = 0.5 s, L = 4.6 m -> v0 ≈ 58.8 m/s ≈ 212 km/h
    v0 = v0_from_distance_time(9.2, 0.5, L)
    assert v0 == pytest.approx(58.8, abs=0.1)
    assert v0 * KMH == pytest.approx(212, abs=1)
    assert distance(0.5, v0, L) == pytest.approx(9.2, abs=1e-9)  # model and inverse agree


def synthetic_track(v0_kmh, rng, *, n=16, noise_m=0.005, drop=0, impact_known=True, angle_deg=-12):
    """2D positions in the hitting plane, sampled at 240 fps, with a sub-frame impact time."""
    v0 = v0_kmh / KMH
    t0 = rng.uniform(0, 1 / FPS)  # impact between frame 0 and frame 1
    frames = np.arange(1, n + 1)
    t = frames / FPS
    d = distance(t - t0, v0, L)
    a = np.radians(angle_deg)
    p_imp = np.array([0.2, 2.5])
    pts = p_imp + np.outer(d, [np.cos(a), np.sin(a)]) + rng.normal(0, noise_m, (n, 2))
    keep = np.sort(rng.choice(n, n - drop, replace=False))
    t, pts = t[keep], pts[keep]
    if impact_known:
        s = track_coordinate(pts, origin=p_imp + rng.normal(0, noise_m, 2))
    else:
        s = track_coordinate(pts)
    return t, s, (0.0, 1 / FPS)


@pytest.mark.parametrize("v0_kmh", [150, 220, 300, 380])
@pytest.mark.parametrize("seed", range(5))
def test_fit_recovers_known_speed(v0_kmh, seed):
    rng = np.random.default_rng(seed)
    t, s, bracket = synthetic_track(v0_kmh, rng, drop=3)
    r = fit_speed(t, s, L, impact_known=True, t0_bracket=bracket)
    assert r.ok, r.reason
    assert r.v0_kmh == pytest.approx(v0_kmh, rel=0.03)
    assert abs(r.v0_kmh - v0_kmh) <= r.margin_kmh  # truth inside the reported margin


@pytest.mark.parametrize("seed", range(5))
def test_fit_without_impact_point_widens_margin(seed):
    rng = np.random.default_rng(seed)
    t, s, bracket = synthetic_track(250, rng, impact_known=False, drop=2)
    r = fit_speed(t, s, L, impact_known=False, t0_bracket=bracket)
    assert r.ok, r.reason
    assert r.v0_kmh == pytest.approx(250, rel=0.05)
    assert abs(r.v0_kmh - 250) <= r.margin_kmh


def test_outlier_is_dropped():
    rng = np.random.default_rng(1)
    t, s, bracket = synthetic_track(250, rng)
    s[7] += 0.25  # e.g. a bright sock picked up instead of the shuttle
    r = fit_speed(t, s, L, impact_known=True, t0_bracket=bracket)
    assert r.ok and 7 not in r.used
    assert r.v0_kmh == pytest.approx(250, rel=0.03)


def test_refuses_noisy_track():
    rng = np.random.default_rng(2)
    t, s, bracket = synthetic_track(250, rng, noise_m=0.08)
    r = fit_speed(t, s, L, impact_known=True, t0_bracket=bracket)
    assert not r.ok and r.reason


def test_refuses_too_few_points():
    rng = np.random.default_rng(3)
    t, s, bracket = synthetic_track(250, rng, n=4)
    r = fit_speed(t, s, L, impact_known=True, t0_bracket=bracket)
    assert not r.ok and "positions" in r.reason


def test_scale_uncertainty_enters_margin():
    rng = np.random.default_rng(4)
    t, s, bracket = synthetic_track(250, rng)
    a = fit_speed(t, s, L, impact_known=True, t0_bracket=bracket)
    b = fit_speed(t, s, L, impact_known=True, t0_bracket=bracket, scale_rel=0.04)
    assert b.margin_kmh > a.margin_kmh >= 0
    assert b.margin_kmh >= 0.04 * b.v0_kmh
