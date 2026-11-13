"""Local smash-speed leaderboard (no internet needed, Python standard library only).

  python leaderboard.py [--event "Club Night"] [--port 8000] [--lan]

  http://localhost:8000/          public board for the TV (auto-refreshes)
  http://localhost:8000/operator  operator page: add nickname + consent to new results (this Mac only)

Privacy: nicknames only, stored only with the player's consent (a parent/guardian's if under 16);
otherwise the board shows "Player N". No photos, faces or video are stored or served.
"""
from __future__ import annotations

import argparse
import json
import re
import threading
import unicodedata
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).parent
WEB = HERE / "web"
DATA = HERE / "data"
LOCAL = {"127.0.0.1", "::1", "::ffff:127.0.0.1"}
LOCAL_HOSTS = {"localhost", "127.0.0.1", "[::1]"}  # Host header names a page on this Mac may use
NICK_RE = re.compile(r"^[\w .'\-]{1,20}$", re.UNICODE)
MAX_BODY = 4096
BOARD_SIZE = 10


def clean_nickname(raw: str) -> str | None:
    nick = unicodedata.normalize("NFKC", str(raw or "")).strip()
    nick = re.sub(r"\s+", " ", nick)
    if not NICK_RE.match(nick) or not any(ch.isalnum() for ch in nick):
        return None
    if re.search(r"(https?:|www\.|\.com\b|@)", nick, re.I):
        return None
    return nick


def host_name(host: str) -> str:
    """'localhost:8000' -> 'localhost', '[::1]:8000' -> '[::1]' (lower case)."""
    host = host.strip().lower()
    if host.startswith("["):
        return host.split("]", 1)[0] + "]"
    return host.rsplit(":", 1)[0]


class Store:
    def __init__(self, data_dir: Path, event: str):
        slug = re.sub(r"[^a-z0-9]+", "-", event.lower()).strip("-") or "event"
        self.event = event
        self.pending_dir = data_dir / "pending"
        self.file = data_dir / "boards" / f"{slug}.json"
        self.lock = threading.Lock()
        self.pending_dir.mkdir(parents=True, exist_ok=True)
        self.file.parent.mkdir(parents=True, exist_ok=True)
        self.entries = json.loads(self.file.read_text())["entries"] if self.file.exists() else []

    def _save(self):
        tmp = self.file.with_suffix(".tmp")
        tmp.write_text(json.dumps({"event": self.event, "entries": self.entries}, indent=1))
        tmp.replace(self.file)

    def pending(self) -> list[dict]:
        out = []
        for p in sorted(self.pending_dir.glob("*.json")):
            try:
                out.append(json.loads(p.read_text()))
            except (OSError, json.JSONDecodeError):
                continue
        return out

    def _pending_path(self, rid: str) -> Path | None:
        if not re.fullmatch(r"[0-9a-f\-]{10,40}", str(rid)):
            return None
        p = self.pending_dir / f"{rid}.json"
        return p if p.exists() else None

    def publish(self, rid: str, nickname: str, consent: bool) -> dict:
        with self.lock:
            p = self._pending_path(rid)
            if not p:
                raise ValueError("result not found (already published?)")
            nick = None
            if consent:
                nick = clean_nickname(nickname)
                if not nick:
                    raise ValueError("nickname: 1-20 letters, digits, spaces or . ' - (no links)")
            r = json.loads(p.read_text())
            number = 1 + max((e.get("number", 0) for e in self.entries), default=0)
            entry = {"id": r["id"], "number": number, "nickname": nick, "consent": bool(consent and nick),
                     "speed_kmh": r["speed_kmh"], "margin_kmh": r["margin_kmh"], "method": r["method"],
                     "shuttle": r.get("shuttle", "feather"), "provisional_drag": r.get("provisional_drag", False),
                     "created": r.get("created") or datetime.now(timezone.utc).isoformat(timespec="seconds")}
            self.entries.append(entry)
            self._save()
            p.unlink()
            return entry

    def discard(self, rid: str) -> None:
        with self.lock:
            p = self._pending_path(rid)
            if p:
                p.unlink()

    def remove(self, rid: str) -> None:
        with self.lock:
            self.entries = [e for e in self.entries if e["id"] != rid]
            self._save()

    @staticmethod
    def label(e: dict) -> str:
        return e["nickname"] if e.get("consent") and e.get("nickname") else f"Player {e['number']}"

    def board(self) -> dict:
        best: dict[str, dict] = {}
        for e in self.entries:
            key = e["nickname"].casefold() if e.get("consent") and e.get("nickname") else f"#{e['number']}"
            if key not in best or e["speed_kmh"] > best[key]["speed_kmh"]:
                best[key] = e
        ranked = sorted(best.values(), key=lambda e: -e["speed_kmh"])[:BOARD_SIZE]
        latest = max(self.entries, key=lambda e: e["number"]) if self.entries else None
        pub = lambda e: {"name": self.label(e), "speed_kmh": e["speed_kmh"], "margin_kmh": e["margin_kmh"],  # noqa: E731
                         "method": e["method"], "number": e["number"]}
        return {"event": self.event, "beta": True, "ranking": [pub(e) for e in ranked],
                "latest": pub(latest) if latest else None, "count": len(self.entries)}


def make_handler(store: Store):
    class Handler(BaseHTTPRequestHandler):
        server_version = "BoothLeaderboard"
        timeout = 30  # a stalled client can't hold a thread forever

        def log_message(self, fmt, *args):  # quiet: no names in logs
            pass

        def _local(self) -> bool:
            # The Host check stops DNS rebinding: a website open on this Mac whose domain is pointed at
            # 127.0.0.1 would pass the address check but sends its own domain as Host.
            return self.client_address[0] in LOCAL and host_name(self.headers.get("Host", "")) in LOCAL_HOSTS

        def _send(self, code: int, body: bytes, ctype: str):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Content-Security-Policy",
                             "default-src 'self'; style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'; "
                             "connect-src 'self'; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code=200):
            self._send(code, json.dumps(obj).encode(), "application/json")

        def _page(self, name: str):
            self._send(200, (WEB / name).read_bytes(), "text/html; charset=utf-8")

        def do_GET(self):
            path = self.path.split("?")[0]
            if path in ("/", "/index.html"):
                return self._page("board.html")
            if path == "/api/board":
                return self._json(store.board())
            if not self._local():
                return self._json({"error": "operator pages are only available on this Mac"}, 403)
            if path == "/operator":
                return self._page("operator.html")
            if path == "/api/pending":
                return self._json({"pending": store.pending()})
            if path == "/api/entries":
                es = sorted(store.entries, key=lambda e: -e["number"])
                return self._json({"entries": [{**e, "label": store.label(e)} for e in es]})
            self._json({"error": "not found"}, 404)

        def do_POST(self):
            if not self._local():
                return self._json({"error": "forbidden"}, 403)
            try:
                n = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                n = -1
            if not 0 <= n <= MAX_BODY or self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                return self._json({"error": "bad request"}, 400)
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
                path = self.path.split("?")[0]
                if path == "/api/publish":
                    e = store.publish(body.get("id", ""), body.get("nickname", ""), body.get("consent") is True)
                    return self._json({"ok": True, "label": store.label(e)})
                if path == "/api/discard":
                    store.discard(body.get("id", ""))
                    return self._json({"ok": True})
                if path == "/api/remove":
                    store.remove(str(body.get("id", "")))
                    return self._json({"ok": True})
                return self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            except (ValueError, json.JSONDecodeError) as ex:
                return self._json({"error": str(ex)}, 400)

    return Handler


def serve(event: str, port: int, lan: bool, data_dir: Path = DATA) -> ThreadingHTTPServer:
    store = Store(data_dir, event)
    return ThreadingHTTPServer(("0.0.0.0" if lan else "127.0.0.1", port), make_handler(store))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--event", default="Smash Speed Booth", help="title; each event gets its own board file")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--lan", action="store_true", help="also show the board to other devices on this Wi-Fi (operator page stays local)")
    a = ap.parse_args()
    srv = serve(a.event, a.port, a.lan)
    print(f"Board:    http://localhost:{a.port}/")
    print(f"Operator: http://localhost:{a.port}/operator")
    if a.lan:
        print("LAN mode: other devices can open the board at http://<this-mac-ip>:%d/" % a.port)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
