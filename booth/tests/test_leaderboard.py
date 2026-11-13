"""Leaderboard server: consent, nickname rules, ranking, operator-only endpoints."""
import json
import threading
import urllib.error
import urllib.request

import pytest

import leaderboard


def put_pending(data_dir, rid, kmh):
    d = data_dir / "pending"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{rid}.json").write_text(json.dumps({"id": rid, "created": "2026-10-31T12:00:00+00:00", "speed_kmh": kmh,
                                              "margin_kmh": 9, "method": "auto", "shuttle": "feather",
                                              "provisional_drag": False, "clip": "x.mov"}))


@pytest.fixture
def server(tmp_path):
    srv = leaderboard.serve("Test Night", 0, False, tmp_path)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", tmp_path
    srv.shutdown()


def get(url):
    with urllib.request.urlopen(url) as r:
        return r.status, r.read(), r.headers


def post(url, body):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_pages_served_with_security_headers(server):
    base, _ = server
    code, body, h = get(base + "/")
    assert code == 200 and b"/api/board" in body
    assert "default-src 'self'" in h["Content-Security-Policy"] and h["X-Content-Type-Options"] == "nosniff"
    assert get(base + "/operator")[0] == 200


def test_consent_controls_public_name_and_ranking(server):
    base, data = server
    put_pending(data, "20261031-120000-aaaaaa", 251)
    put_pending(data, "20261031-120100-bbbbbb", 288)
    put_pending(data, "20261031-120200-cccccc", 230)
    assert len(json.loads(get(base + "/api/pending")[1])["pending"]) == 3
    assert post(base + "/api/publish", {"id": "20261031-120000-aaaaaa", "nickname": "Shuttle Shark", "consent": True})[0] == 200
    code, r = post(base + "/api/publish", {"id": "20261031-120100-bbbbbb", "nickname": "Secret Name", "consent": False})
    assert code == 200 and r["label"] == "Player 2"
    # same nickname again with a slower smash: board keeps the best one
    post(base + "/api/publish", {"id": "20261031-120200-cccccc", "nickname": "shuttle shark", "consent": True})
    b = json.loads(get(base + "/api/board")[1])
    assert [e["name"] for e in b["ranking"]] == ["Player 2", "Shuttle Shark"]
    assert b["beta"] is True and b["event"] == "Test Night"
    stored = (data / "boards" / "test-night.json").read_text()
    assert "Secret Name" not in stored  # no consent -> nickname never stored
    assert not list((data / "pending").glob("*.json"))


@pytest.mark.parametrize("nick", ["", "   ", "<script>alert(1)</script>", "x" * 21, "www.spam.com", "a@b", "___"])
def test_bad_nicknames_rejected(server, nick):
    base, data = server
    put_pending(data, "20261031-130000-dddddd", 200)
    code, r = post(base + "/api/publish", {"id": "20261031-130000-dddddd", "nickname": nick, "consent": True})
    assert code == 400 and "nickname" in r["error"]


def test_path_traversal_and_remove(server):
    base, data = server
    assert post(base + "/api/publish", {"id": "../../etc/passwd", "nickname": "A", "consent": True})[0] == 400
    put_pending(data, "20261031-140000-eeeeee", 199)
    post(base + "/api/publish", {"id": "20261031-140000-eeeeee", "nickname": "Ace", "consent": True})
    assert post(base + "/api/remove", {"id": "20261031-140000-eeeeee"})[0] == 200
    assert json.loads(get(base + "/api/board")[1])["ranking"] == []


def test_operator_endpoints_local_only(tmp_path):
    store = leaderboard.Store(tmp_path, "x")
    handler = leaderboard.make_handler(store)
    assert handler.__name__ == "Handler"
    h = handler.__new__(handler)
    h.headers = {"Host": "localhost:8000"}
    h.client_address = ("192.168.1.20", 5000)
    assert not h._local()
    h.client_address = ("127.0.0.1", 5000)
    assert h._local()
    h.headers = {"Host": "evil.example:8000"}
    assert not h._local()


def test_operator_refuses_foreign_host_header(server):
    """DNS rebinding: a page from another domain pointed at 127.0.0.1 sends its own Host."""
    import http.client
    from urllib.parse import urlparse
    base, _ = server
    u = urlparse(base)
    for host, want in (("evil.example:%d" % u.port, 403), ("localhost:%d" % u.port, 200)):
        c = http.client.HTTPConnection(u.hostname, u.port, timeout=10)
        c.request("GET", "/api/entries", headers={"Host": host})
        assert c.getresponse().status == want, host
    c = http.client.HTTPConnection(u.hostname, u.port, timeout=10)
    c.request("POST", "/api/discard", body=b"{}", headers={"Host": "evil.example", "Content-Type": "application/json"})
    assert c.getresponse().status == 403
    assert leaderboard.host_name("[::1]:8000") == "[::1]" and leaderboard.host_name("LocalHost") == "localhost"
