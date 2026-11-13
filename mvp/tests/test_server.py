"""MVP server: auth, upload -> result, tap fallback, re-timed clips, deletion. Real HTTP, synthetic clips."""
import json
import threading
import time
import urllib.error
import urllib.request

import pytest

import server
import synth


@pytest.fixture(scope="module")
def clips(tmp_path_factory):
    d = tmp_path_factory.mktemp("clips")
    out = {}
    for name, kw in {"hall": dict(kmh=240, seed=21, hall=True),
                     "retimed": dict(kmh=250, seed=22, hall=True, retime_fps=30),
                     "no_mat": dict(kmh=260, seed=23, hall=True, with_mat=False)}.items():
        out[name] = (d / f"{name}.mov", synth.render(d / f"{name}.mov", **kw))
    return out


@pytest.fixture
def srv(tmp_path, monkeypatch):
    monkeypatch.setattr(server.speed, "DATA", tmp_path / "data")
    monkeypatch.setattr(server.speed, "LAST_POSE", tmp_path / "data" / "last_pose.json")
    httpd, app = server.serve(0, local=True, work_dir=tmp_path / "work", demo=False)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", app
    httpd.shutdown()


def call(base, app, path, method="GET", body=None, ctype="application/json", key=True):
    headers = {"Content-Type": ctype}
    if key:
        headers["X-Token"] = app.token
    req = urllib.request.Request(base + path, body, headers, method=method)
    try:
        with urllib.request.urlopen(req) as r:
            data = r.read()
            return r.status, (json.loads(data) if r.headers["Content-Type"] == "application/json" else data)
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def wait(base, app, jid, timeout=60):
    t0 = time.time()
    while time.time() - t0 < timeout:
        code, j = call(base, app, f"/api/jobs/{jid}")
        if j["status"] not in ("queued", "processing"):
            return j
        time.sleep(0.2)
    raise AssertionError("job did not finish")


def upload(base, app, clip, query):
    return call(base, app, "/api/jobs?" + query, "POST", clip.read_bytes(), "video/quicktime")


def test_page_public_api_needs_key(srv):
    base, app = srv
    code, body = call(base, app, "/", key=False)
    assert code == 200 and b"Smash" in body
    assert call(base, app, "/api/ping", key=False)[0] == 401
    assert call(base, app, "/api/ping")[1]["ok"] is True
    assert call(base, app, "/api/jobs?mode=mat", "POST", b"x", "video/quicktime", key=False)[0] == 401


def test_upload_to_verified_result_and_delete(srv, clips):
    base, app = srv
    path, truth = clips["hall"]
    code, j = upload(base, app, path, "mode=mat&shuttle=feather")
    assert code == 202
    r = wait(base, app, j["id"])
    assert r["status"] == "done", r
    res = r["result"]
    assert res["verified"] is True and res["beta"] is True
    assert abs(res["speed_kmh"] - truth["kmh"]) <= res["margin_kmh"]
    assert res["curve"]["fit_kmh"]
    assert "clip" not in res and "timing" not in res  # nothing about the file leaks back
    assert (app.work / f"{j['id']}.mov").exists()
    assert call(base, app, f"/api/jobs/{j['id']}", "DELETE")[0] == 200
    assert not (app.work / f"{j['id']}.mov").exists()
    assert call(base, app, f"/api/jobs/{j['id']}")[0] == 404


def test_no_mat_estimate(srv, clips):
    base, app = srv
    path, truth = clips["no_mat"]
    assert upload(base, app, path, "mode=estimate&shuttle=feather")[0] == 400  # distance required
    j = upload(base, app, path, "mode=estimate&shuttle=feather&distance=4.2")[1]
    r = wait(base, app, j["id"])
    assert r["status"] == "done" and r["result"]["verified"] is False
    assert abs(r["result"]["speed_kmh"] - truth["kmh"]) <= r["result"]["margin_kmh"]


def test_mat_mode_without_mat_gives_friendly_error(srv, clips):
    base, app = srv
    j = upload(base, app, clips["no_mat"][0], "mode=mat")[1]
    r = wait(base, app, j["id"])
    assert r["status"] == "error" and "mat" in r["error"] and "No-mat" in r["error"]


def test_retimed_clip_with_slomo_switch(srv, clips):
    base, app = srv
    path, truth = clips["retimed"]
    r = wait(base, app, upload(base, app, path, "mode=mat&slomo=0")[1]["id"])
    assert r["status"] == "error" and "240" in r["error"]
    r = wait(base, app, upload(base, app, path, "mode=mat&slomo=1")[1]["id"])
    assert r["status"] == "done" and r["result"]["assumed_240"] is True
    assert abs(r["result"]["speed_kmh"] - truth["kmh"]) <= r["result"]["margin_kmh"]


def test_taps_give_a_result(srv, clips):
    base, app = srv
    path, truth = clips["hall"]
    jid = upload(base, app, path, "mode=mat")[1]["id"]
    wait(base, app, jid)
    code, f = call(base, app, f"/api/jobs/{jid}/frames")
    assert code == 200 and f["count"] == truth["frames"] and f["width"] == 960
    code, img = call(base, app, f"/api/jobs/{jid}/frame/{f['start']}")
    assert code == 200 and img[:2] == b"\xff\xd8"  # JPEG
    sc = synth.Scene()
    s = f["width"] / 1920
    contact = int(truth["t_impact"] * truth["fps"])
    taps = []
    for fr in range(contact, contact + 6):
        x, z = truth["shuttle_xz"][fr]
        u, v = sc.project([[x, 0, z]])[0]
        taps.append({"frame": fr, "x": float(u * s), "y": float(v * s)})  # display-size pixels, like the phone
    assert call(base, app, f"/api/jobs/{jid}/taps", "POST", json.dumps({"taps": taps[:3]}).encode())[0] == 400
    code, _ = call(base, app, f"/api/jobs/{jid}/taps", "POST", json.dumps({"taps": taps}).encode())
    assert code == 200
    r = wait(base, app, jid)
    assert r["status"] == "done" and r["result"]["method"] == "manual"
    assert abs(r["result"]["speed_kmh"] - truth["kmh"]) <= r["result"]["margin_kmh"]


def test_bad_inputs(srv):
    base, app = srv
    assert call(base, app, "/api/jobs?mode=sideways", "POST", b"x", "video/quicktime")[0] == 400
    code, j = call(base, app, "/api/jobs?mode=mat", "POST", b"not a video at all", "video/quicktime")
    r = wait(base, app, j["id"])
    assert r["status"] == "error" and "video" in r["error"]
    assert call(base, app, "/api/jobs/../../etc/passwd")[0] == 404
    assert call(base, app, "/api/jobs/zzzz/frames")[0] == 404


def raw(base, method, path, headers, body=b""):
    """Send exact headers (urllib won't send a bad Content-Length)."""
    import http.client
    from urllib.parse import urlparse
    u = urlparse(base)
    c = http.client.HTTPConnection(u.hostname, u.port, timeout=10)
    c.putrequest(method, path)
    for k, v in headers.items():
        c.putheader(k, v)
    c.endheaders(body)
    r = c.getresponse()
    return r.status, r.read()


def test_hardening_limits(srv, clips, monkeypatch):
    base, app = srv
    path, _ = clips["hall"]
    jid = upload(base, app, path, "mode=mat")[1]["id"]
    wait(base, app, jid)
    # negative / junk lengths are refused instead of reading until the client hangs up
    for n in ("-1", "abc"):
        assert raw(base, "POST", f"/api/jobs/{jid}/taps", {"X-Token": app.token, "Content-Length": n})[0] == 400
        assert raw(base, "POST", "/api/jobs?mode=mat", {"X-Token": app.token, "Content-Length": n})[0] == 413
    # clips over the frame cap are rejected, deleted, and never decoded for tapping
    monkeypatch.setattr(server, "MAX_FRAMES", 10)
    long_id = upload(base, app, path, "mode=mat")[1]["id"]
    r = wait(base, app, long_id)
    assert r["status"] == "error" and "too long" in r["error"]
    assert not (app.work / f"{long_id}.mov").exists()
    assert call(base, app, f"/api/jobs/{long_id}/frames")[0] == 409
    taps = json.dumps({"taps": [{"frame": i, "x": 1, "y": 1} for i in range(5)]}).encode()
    assert call(base, app, f"/api/jobs/{long_id}/taps", "POST", taps)[0] == 409
    # disk cap: no more than MAX_JOBS clips at once
    monkeypatch.setattr(server, "MAX_JOBS", 1)
    assert upload(base, app, path, "mode=mat")[0] == 503


def test_new_key_revokes_old(tmp_path):
    f = tmp_path / "token.txt"
    a = server._load_token(f)
    assert server._load_token(f) == a
    b = server._load_token(f, rotate=True)
    assert b != a and len(b) >= 16 and server._load_token(f) == b
