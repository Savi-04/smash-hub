"""End-to-end: synthetic clips with known speeds through speed.analyse (auto + manual)."""
import json

import numpy as np
import pytest

import manual
import speed
import synth


@pytest.fixture(autouse=True)
def isolated_data(tmp_path, monkeypatch):
    monkeypatch.setattr(speed, "DATA", tmp_path / "data")
    monkeypatch.setattr(speed, "LAST_POSE", tmp_path / "data" / "last_pose.json")


@pytest.fixture(scope="module")
def clips(tmp_path_factory):
    d = tmp_path_factory.mktemp("clips")
    specs = {
        "base": dict(kmh=250, seed=0),
        "fast_gaps": dict(kmh=340, seed=1, angle=-16, hide=(27, 31, 32)),  # missed detections mid-track
        "slow": dict(kmh=160, seed=2, angle=-6),
        "late_contact": dict(kmh=321, seed=0, angle=-15, t_imp=0.10717),  # track starts on the contact frame
        "no_contact": dict(kmh=220, seed=3, hide=(23, 24)),  # racket hides the shuttle at contact
        "retimed": dict(kmh=250, seed=4, retime_fps=30),  # slow-motion export: 240 fps frames at 30 fps
        "hall": dict(kmh=230, seed=6, hall=True),  # no backdrop: light walls, ceiling lights, someone walking past
        "hall_shaky": dict(kmh=280, seed=7, hall=True, shake_px=2.0),  # hand-held phone
        "no_mat": dict(kmh=270, seed=5, hall=True, with_mat=False),
    }
    out = {}
    for name, kw in specs.items():
        path = d / f"{name}.mov"
        out[name] = (path, synth.render(path, **kw))
    return out


@pytest.mark.parametrize("name", ["base", "fast_gaps", "slow", "late_contact", "hall", "hall_shaky"])
def test_auto_speed_matches_truth(clips, name):
    path, truth = clips[name]
    r = speed.analyse(path, allow_window=False)
    assert r["status"] == "ok", r
    assert r["method"] == "auto" and r["fit_mode"] == "impact-point"
    assert r["speed_kmh"] == pytest.approx(truth["kmh"], rel=0.03)
    assert abs(r["speed_kmh"] - truth["kmh"]) <= r["margin_kmh"]
    assert r["seconds"] < 30


def test_contact_hidden_uses_bracket_and_audio(clips):
    path, truth = clips["no_contact"]
    r = speed.analyse(path, allow_window=False)
    assert r["status"] == "ok" and r["fit_mode"] == "bracket"
    assert abs(r["speed_kmh"] - truth["kmh"]) <= r["margin_kmh"]
    ra = speed.analyse(path, allow_window=False, audio=True)
    assert any("hit sound" in n for n in ra["notes"])
    assert ra["margin_kmh"] <= r["margin_kmh"]
    assert ra["speed_kmh"] == pytest.approx(truth["kmh"], rel=0.03)


def test_retimed_clip_is_refused_until_capture_fps_given(clips):
    path, truth = clips["retimed"]
    with pytest.raises(speed.BoothError, match="capture-fps"):
        speed.analyse(path, allow_window=False)
    r = speed.analyse(path, allow_window=False, capture_fps=240)
    assert r["status"] == "ok"
    assert r["speed_kmh"] == pytest.approx(truth["kmh"], rel=0.03)


def test_manual_clicks_give_same_answer(clips, tmp_path):
    path, truth = clips["base"]
    sc = synth.Scene()
    rng = np.random.default_rng(7)
    contact = int(truth["t_impact"] * truth["fps"])  # last frame before the shuttle flies
    clicks = []
    for f in range(contact, contact + 7):
        x, z = truth["shuttle_xz"][f]
        u, v = sc.project([[x, 0, z]])[0] + rng.normal(0, 1.0, 2)  # operator clicks within ~1 px
        clicks.append({"frame": f, "u": float(u), "v": float(v)})
    f = tmp_path / "clicks.json"
    f.write_text(json.dumps(clicks))
    assert len(manual.load_clicks(f)) == 7
    r = speed.analyse(path, clicks_file=f, allow_window=False)
    assert r["status"] == "ok" and r["method"] == "manual"
    assert r["speed_kmh"] == pytest.approx(truth["kmh"], rel=0.05)
    assert abs(r["speed_kmh"] - truth["kmh"]) <= r["margin_kmh"]


def test_mat_missing_uses_saved_pose(clips, tmp_path, monkeypatch):
    path, truth = clips["base"]
    speed.analyse(path, check_only=True)  # setup check saves the pose
    assert speed.LAST_POSE.exists()
    monkeypatch.setattr(speed, "detect_markers", lambda g: {})
    r = speed.analyse(path, allow_window=False)
    assert r["pose_source"].startswith("saved") and r["status"] == "ok"


def test_setup_check(clips):
    r = speed.analyse(clips["base"][0], check_only=True)
    assert r["status"] == "setup ok" and r["markers"] == 4
    assert r["camera_distance_m"] == pytest.approx(4.2, abs=0.25)


def test_pending_entry_has_no_media(clips):
    r = speed.analyse(clips["base"][0], allow_window=False)
    p = speed.add_pending(r)
    entry = json.loads(p.read_text())
    assert set(entry) == {"id", "created", "speed_kmh", "margin_kmh", "method", "shuttle", "provisional_drag", "clip"}


def test_shake_is_measured(clips):
    r = speed.analyse(clips["hall_shaky"][0], allow_window=False)
    assert r["shake_px"] > 5
    assert speed.analyse(clips["hall"][0], allow_window=False)["shake_px"] < 1


@pytest.mark.parametrize("distance", [3.8, 4.2, 4.6])  # true distance 4.2 m; guesses ±10 %
def test_no_mat_estimate_keeps_truth_inside_margin(clips, distance):
    path, truth = clips["no_mat"]
    r = speed.analyse(path, allow_window=False, mode="estimate", distance_m=distance)
    assert r["status"] == "ok" and r["mode"] == "estimate" and r["lens"] == "nominal"
    assert abs(r["speed_kmh"] - truth["kmh"]) <= r["margin_kmh"]
    assert r["margin_kmh"] > 0.1 * r["speed_kmh"]  # an estimate never claims mat accuracy


def test_no_mat_clip_in_mat_mode_is_refused(clips):
    with pytest.raises(speed.BoothError, match="mat not found"):
        speed.analyse(clips["no_mat"][0], allow_window=False)


def test_result_has_chart_curve(clips):
    r = speed.analyse(clips["base"][0], allow_window=False)
    c = r["curve"]
    assert len(c["t_ms"]) == r["points"] and c["fit_kmh"][0] == pytest.approx(r["speed_kmh"], abs=1)
    assert c["fit_kmh"][-1] < c["fit_kmh"][0]  # slows down
