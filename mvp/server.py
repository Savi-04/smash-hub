"""Smash speed MVP: phone web app + this Mac as the server (beta).

  booth/.venv/bin/python mvp/server.py [--port 8080] [--local]

Prints a link + QR code: open it on the iPhone (same Wi-Fi), "Add to Home Screen".
The phone uploads a slo-mo clip; this Mac finds the shuttle and fits its speed with the
booth engine (booth/). Videos are deleted as soon as the user is done, or after 15 minutes.
Results live on the phone only (no accounts, nothing stored here).
"""
from __future__ import annotations

import argparse
import json
import queue
import re
import secrets
import shutil
import socket
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "booth"))

import cv2  # noqa: E402

import physics  # noqa: E402
import speed  # noqa: E402
import video  # noqa: E402

STATIC = HERE / "static"
MAX_UPLOAD = 400 * 1024 * 1024
MAX_CAPTURE_SECONDS = 12.0  # real (capture) time; longer clips: trim to the smash
MAX_FRAMES = int(MAX_CAPTURE_SECONDS * 240)  # also caps re-timed clips and the frames kept in memory for tapping
MAX_JOBS = 12  # clips on disk at once (each up to MAX_UPLOAD)
REQUEST_TIMEOUT = 30  # seconds a client may stall on one read/write
JOB_TTL = 15 * 60
FRAME_LONG_SIDE = 960
CHUNK = 1 << 20


class Job:
    def __init__(self, jid: str, path: Path, params: dict):
        self.id, self.path, self.params = jid, path, params
        self.status, self.stage, self.progress = "queued", "Waiting", 0.0
        self.result: dict | None = None
        self.error: str | None = None
        self.created = time.time()
        self.frames: list[bytes] | None = None  # JPEGs for tapping, built on request
        self.frame_scale = 1.0
        self.lock = threading.Lock()

    def public(self) -> dict:
        return {"id": self.id, "status": self.status, "stage": self.stage, "progress": round(self.progress, 2),
                "result": self.result, "error": self.error}


class App:
    def __init__(self, work_dir: Path, demo_clip: Path | None = None, new_key: bool = False):
        self.work = work_dir
        self.work.mkdir(parents=True, exist_ok=True)
        self.token = _load_token(HERE / "data" / "token.txt", rotate=new_key)
        self.jobs: dict[str, Job] = {}
        self.q: queue.Queue[Job] = queue.Queue()
        self.demo_clip = demo_clip
        threading.Thread(target=self._worker, daemon=True).start()
        threading.Thread(target=self._janitor, daemon=True).start()

    # ---- jobs -------------------------------------------------------------
    def new_job(self, params: dict) -> Job | None:
        if sum(j.status != "error" for j in list(self.jobs.values())) >= MAX_JOBS:
            return None
        jid = secrets.token_hex(8)
        job = Job(jid, self.work / f"{jid}.mov", params)
        self.jobs[jid] = job
        return job

    def _run(self, job: Job, clicks=None):
        p = job.params

        def progress(stage, frac):
            job.stage, job.progress = stage, frac

        kw = dict(allow_window=False, mode=p["mode"], distance_m=p.get("distance"), shuttle=p["shuttle"],
                  progress=progress, clicks=clicks)
        try:
            info = video.probe(job.path)
        except Exception:
            raise speed.BoothError("That file isn't a video we can read.")
        if len(info.timestamps) > MAX_FRAMES:
            raise speed.BoothError(f"Clip is too long. Trim it to the smash (under {MAX_CAPTURE_SECONDS:.0f} s) in Photos.")
        span =float(info.timestamps[-1] - info.timestamps[0]) if len(info.timestamps) > 1 else 0.0
        if info.timing_ok and span > MAX_CAPTURE_SECONDS:
            raise speed.BoothError(f"Clip is {span:.0f} s long. Trim it to the smash (under {MAX_CAPTURE_SECONDS:.0f} s) in Photos.")
        try:
            return speed.analyse(job.path, **kw)
        except speed.BoothError as e:
            # iPhone slo-mo exported through the photo picker can play 240 fps frames at 30 fps
            if not info.timing_ok and p.get("slomo"):
                r = speed.analyse(job.path, capture_fps=240, **kw)
                r["notes"].append("timestamps were re-timed by the phone; assumed 240 fps capture")
                r["assumed_240"] = True
                return r
            raise e

    def _worker(self):
        while True:
            job = self.q.get()
            clicks = job.params.pop("_clicks", None)
            job.status = "processing"
            try:
                r = self._run(job, clicks)
                job.result = _public_result(r)
                job.status = "done" if r["status"] == "ok" else "needs_taps"
                if job.status == "needs_taps":
                    job.stage = "We couldn't lock onto the shuttle"
            except speed.BoothError as e:
                job.status, job.error = "error", _friendly(str(e))
            except Exception as e:  # never crash the worker; keep details out of the reply
                print(f"job {job.id} failed: {type(e).__name__}: {e}", file=sys.stderr)
                job.status, job.error = "error", "Something went wrong analysing this clip."
            finally:
                job.progress = 1.0
                if job.status == "error":  # nothing left to tap: drop the clip now, keep only the message
                    job.path.unlink(missing_ok=True)

    def _janitor(self):
        while True:
            time.sleep(30)
            now = time.time()
            for jid, job in list(self.jobs.items()):
                if now - job.created > JOB_TTL and job.status not in ("processing", "queued"):
                    self.delete(jid)

    def delete(self, jid: str):
        job = self.jobs.pop(jid, None)
        if job:
            job.frames = None
            job.path.unlink(missing_ok=True)

    def frames(self, job: Job) -> list[bytes]:
        with job.lock:
            if job.frames is None:
                out = []
                for i, fr in video.frames(job.path, gray=False):
                    if i >= MAX_FRAMES:
                        break
                    h, w = fr.shape[:2]
                    job.frame_scale = min(1.0, FRAME_LONG_SIDE / max(h, w))
                    small = cv2.resize(fr, None, fx=job.frame_scale, fy=job.frame_scale, interpolation=cv2.INTER_AREA)
                    out.append(cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 82])[1].tobytes())
                job.frames = out
            return job.frames


def _friendly(msg: str) -> str:
    if "capture-fps" in msg or "fps median" in msg:
        return ("This clip doesn't look like 240 fps slo-mo. Record in Slo-mo (240 fps) and send the original, "
                "or turn on 'Recorded in Slo-mo' when you upload.")
    if "mat not found" in msg:
        return "We couldn't see the mat. Make sure all 4 marker pages are in the picture, or use No-mat mode."
    if "no smash track" in msg:
        return "We couldn't find the shuttle's flight."
    return msg[:1].upper() + msg[1:]


def _public_result(r: dict) -> dict:
    keep = ("status", "method", "speed_kmh", "margin_kmh", "reason", "points", "mode", "fps", "shuttle",
            "provisional_drag", "lens", "beta", "seconds", "suggested_frame", "frames", "curve", "camera_distance_m",
            "shake_px", "assumed_240")
    out = {k: r.get(k) for k in keep}
    out["verified"] = r.get("mode") == "mat" and r.get("status") == "ok"
    return out


def _load_token(path: Path, rotate: bool = False) -> str:
    """Same key across restarts, so the phone's Home Screen link keeps working. rotate=True (or deleting the
    file) makes a new key and revokes every old link."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rotate and path.exists() and len(path.read_text().strip()) >= 16:
        return path.read_text().strip()
    tok = secrets.token_urlsafe(12)
    path.write_text(tok)
    path.chmod(0o600)
    return tok


def lan_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))  # no packet is sent for UDP connect
            return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"


def make_handler(app: App):
    class Handler(BaseHTTPRequestHandler):
        server_version = "SmashMVP"
        timeout = REQUEST_TIMEOUT  # a stalled phone can't hold a thread forever

        def log_message(self, *a):
            pass

        def _length(self) -> int:
            try:
                return int(self.headers.get("Content-Length") or 0)
            except ValueError:
                return -1

        def _send(self, code, body: bytes, ctype: str, cache: str = "no-store"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", cache)
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Content-Security-Policy",
                             "default-src 'self'; img-src 'self' blob: data:; style-src 'self' 'unsafe-inline'; "
                             "script-src 'self' 'unsafe-inline'; connect-src 'self'; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code=200):
            self._send(code, json.dumps(obj).encode(), "application/json")

        def _authed(self, qs) -> bool:
            tok = self.headers.get("X-Token") or (qs.get("k") or [""])[0]
            return secrets.compare_digest(tok, app.token)

        def _job(self, jid) -> Job | None:
            return app.jobs.get(jid) if re.fullmatch(r"[0-9a-f]{16}", jid or "") else None

        def do_GET(self):
            u = urlparse(self.path)
            qs = parse_qs(u.query)
            if u.path in ("/", "/index.html"):
                return self._send(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
            if u.path == "/manifest.webmanifest":
                return self._send(200, (STATIC / "manifest.webmanifest").read_bytes(), "application/manifest+json")
            if u.path == "/icon.svg":
                return self._send(200, (STATIC / "icon.svg").read_bytes(), "image/svg+xml", "max-age=86400")
            if not u.path.startswith("/api/"):
                return self._json({"error": "not found"}, 404)
            if not self._authed(qs):
                return self._json({"error": "Open the link shown on the Mac (it has a key)."}, 401)
            if u.path == "/api/ping":
                return self._json({"ok": True, "demo": app.demo_clip is not None})
            m = re.fullmatch(r"/api/jobs/([0-9a-f]+)(/frames|/frame/(\d+))?", u.path)
            job = self._job(m.group(1)) if m else None
            if not job:
                return self._json({"error": "not found"}, 404)
            if m.group(2) is None:
                return self._json(job.public())
            if job.status in ("queued", "processing"):
                return self._json({"error": "still analysing"}, 409)
            if job.status == "error":  # e.g. too long or not a video: never decode it into memory
                return self._json({"error": "no frames for this clip"}, 409)
            frames = app.frames(job)
            if m.group(2) == "/frames":
                img = cv2.imdecode(_np_bytes(frames[0]), cv2.IMREAD_COLOR) if frames else None
                h, w = img.shape[:2] if img is not None else (0, 0)
                sug = (job.result or {}).get("suggested_frame")
                return self._json({"count": len(frames), "width": w, "height": h,
                                   "start": max(0, sug) if sug is not None else 0})
            i = int(m.group(3))
            if not 0 <= i < len(frames):
                return self._json({"error": "no such frame"}, 404)
            return self._send(200, frames[i], "image/jpeg", "private, max-age=600")

        def do_POST(self):
            u = urlparse(self.path)
            qs = parse_qs(u.query)
            if not self._authed(qs):
                return self._json({"error": "Open the link shown on the Mac (it has a key)."}, 401)
            if u.path == "/api/jobs":
                return self._upload(qs)
            m = re.fullmatch(r"/api/jobs/([0-9a-f]+)/taps", u.path)
            job = self._job(m.group(1)) if m else None
            if not job:
                return self._json({"error": "not found"}, 404)
            if job.status not in ("done", "needs_taps"):
                return self._json({"error": "no frames for this clip"}, 409)
            n = self._length()
            if not 0 < n <= 8192:
                return self._json({"error": "bad taps"}, 400 if n <= 0 else 413)
            try:
                taps = json.loads(self.rfile.read(n))["taps"]
                frames = app.frames(job)
                s = job.frame_scale
                clicks = sorted({int(t["frame"]): (int(t["frame"]), float(t["x"]) / s, float(t["y"]) / s)
                                 for t in taps if 0 <= int(t["frame"]) < len(frames)}.values())
            except (ValueError, KeyError, TypeError, json.JSONDecodeError):
                return self._json({"error": "bad taps"}, 400)
            if len(clicks) < 5:
                return self._json({"error": "Tap the shuttle at contact and in at least 4 more frames."}, 400)
            job.params["_clicks"] = clicks
            job.status, job.stage, job.progress, job.error = "queued", "Fitting your taps", 0.0, None
            app.q.put(job)
            return self._json(job.public())

        def _upload(self, qs):
            mode = (qs.get("mode") or ["mat"])[0]
            shuttle = (qs.get("shuttle") or ["feather"])[0]
            try:
                distance = float((qs.get("distance") or ["0"])[0]) or None
            except ValueError:
                distance = None
            if mode not in ("mat", "estimate") or shuttle not in physics.DRAG_LENGTH_M:
                return self._json({"error": "bad options"}, 400)
            if mode == "estimate" and not (distance and 1.0 <= distance <= 12.0):
                return self._json({"error": "Enter the phone's distance (1–12 m)."}, 400)
            params = {"mode": mode, "shuttle": shuttle, "distance": distance, "slomo": (qs.get("slomo") or ["1"])[0] == "1"}
            demo = (qs.get("demo") or ["0"])[0] == "1"
            job = app.new_job(params)
            if job is None:
                return self._json({"error": "The server is busy. Close an old result or wait a minute."}, 503)
            if demo:
                if not app.demo_clip:
                    app.delete(job.id)
                    return self._json({"error": "no demo clip on this server"}, 404)
                shutil.copyfile(app.demo_clip, job.path)
            else:
                n = self._length()
                if n <= 0 or n > MAX_UPLOAD:
                    app.delete(job.id)
                    return self._json({"error": "Clip is too big (max 400 MB) or empty."}, 413)
                with open(job.path, "wb") as f:
                    left = n
                    while left:
                        chunk = self.rfile.read(min(CHUNK, left))
                        if not chunk:
                            break
                        f.write(chunk)
                        left -= len(chunk)
                if left:
                    app.delete(job.id)
                    return self._json({"error": "upload interrupted"}, 400)
            app.q.put(job)
            return self._json(job.public(), 202)

        def do_DELETE(self):
            u = urlparse(self.path)
            if not self._authed(parse_qs(u.query)):
                return self._json({"error": "unauthorised"}, 401)
            m = re.fullmatch(r"/api/jobs/([0-9a-f]{16})", u.path)
            if m:
                app.delete(m.group(1))
            return self._json({"ok": True})

    return Handler


def _np_bytes(b: bytes):
    import numpy as np
    return np.frombuffer(b, np.uint8)


def ensure_demo_clip(path: Path) -> Path:
    if not path.exists():
        import synth
        synth.render(path, kmh=263, seed=11, hall=True, angle=-12)
        path.with_suffix(".truth.json").unlink(missing_ok=True)
    return path


def serve(port: int, local: bool, work_dir: Path | None = None, demo: bool = True, new_key: bool = False):
    work = work_dir or Path(tempfile.mkdtemp(prefix="smash-mvp-"))
    demo_clip = ensure_demo_clip(HERE / "data" / "demo.mov") if demo else None
    app = App(work, demo_clip, new_key)
    srv = ThreadingHTTPServer(("127.0.0.1" if local else "0.0.0.0", port), make_handler(app))
    return srv, app


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--local", action="store_true", help="only this Mac (no phone access)")
    ap.add_argument("--no-demo", action="store_true", help="don't offer the demo clip")
    ap.add_argument("--new-key", action="store_true", help="make a new link key (old links stop working); use per event")
    a = ap.parse_args()
    srv, app = serve(a.port, a.local, demo=not a.no_demo, new_key=a.new_key)
    host = "localhost" if a.local else lan_ip()
    url = f"http://{host}:{a.port}/?k={app.token}"
    print(f"\nOpen on the iPhone (same Wi-Fi):\n  {url}\n")
    try:
        import segno
        segno.make(url).terminal(compact=True)
    except ImportError:
        pass
    print("\nCtrl+C to stop. Uploaded clips are deleted when done or after 15 min.")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        shutil.rmtree(app.work, ignore_errors=True)


if __name__ == "__main__":
    main()
