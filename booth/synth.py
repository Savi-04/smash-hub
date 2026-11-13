"""Synthetic smash clip with a known answer, for testing the whole pipeline without a real video.

Renders the printed mat on a dark floor, a dark backdrop, a grey player + racket and a white
shuttle (with motion blur) at 240 fps, simulated with drag + gravity; adds a hit sound.
Writes <out>.mov and <out>.truth.json.

Run:  python synth.py [--out out/synthetic_smash.mov] [--kmh 250] [--retime-fps 30]
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import wave
from pathlib import Path

import cv2
import numpy as np

import mat

G = 9.81


def look_at(C, target):
    C, target = np.asarray(C, float), np.asarray(target, float)
    fwd = (target - C) / np.linalg.norm(target - C)
    right = np.cross(fwd, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    R = np.stack([right, down, fwd])
    return R, -R @ C


def simulate(v0, angle_deg, L, t_end, dt=2e-5):
    """Drag (v^2 / L) + gravity, in the X-Z plane. Returns times and positions relative to impact."""
    a = np.radians(angle_deg)
    v = v0 * np.array([np.cos(a), np.sin(a)])
    p = np.zeros(2)
    ts, ps = [0.0], [p.copy()]
    t = 0.0
    while t < t_end:
        sp = np.linalg.norm(v)
        v = v + dt * (-sp * v / L + np.array([0, -G]))
        p = p + dt * v
        t += dt
        ts.append(t)
        ps.append(p.copy())
    return np.array(ts), np.array(ps)


class Scene:
    def __init__(self, w=1920, h=1080, hfov=66.0, cam=(1.2, -4.2, 1.3), target=(1.2, 0.0, 1.15)):
        self.w, self.h = w, h
        f = 0.5 * w / np.tan(np.radians(hfov) / 2)
        self.K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])
        self.R, self.t = look_at(cam, target)
        self.rvec = cv2.Rodrigues(self.R)[0]
        self.cam = np.asarray(cam, float)

    def project(self, xyz):
        pts, _ = cv2.projectPoints(np.asarray(xyz, float).reshape(-1, 3), self.rvec, self.t, self.K, None)
        return pts.reshape(-1, 2)

    def static_background(self, hall: bool = False, with_mat: bool = True) -> np.ndarray:
        """hall=False: dark backdrop booth. hall=True: ordinary sports hall, no backdrop."""
        if hall:
            yy = np.linspace(0, 1, self.h)[:, None]
            img = (135 + 30 * yy + 6 * np.sin(np.arange(self.w)[None, :] / 37.0)).astype(np.uint8).copy()
            for x0 in (-3.0, 0.5, 4.0):  # ceiling lights on the far wall
                q = self.project([[x0, 8, 4.2], [x0 + 1.2, 8, 4.2], [x0 + 1.2, 8, 3.9], [x0, 8, 3.9]])
                cv2.fillPoly(img, [q.astype(np.int32)], 252)
            floor = self.project([[-10, -3, 0], [12, -3, 0], [12, 8, 0], [-10, 8, 0]])
            cv2.fillPoly(img, [floor.astype(np.int32)], 105)
            for y in (1.0, 2.5, 6.0):  # court lines
                a, b = self.project([[-10, y, 0], [12, y, 0]])
                cv2.line(img, tuple(a.astype(int)), tuple(b.astype(int)), 215, 3)
        else:
            img = np.full((self.h, self.w), 22, np.uint8)  # dark backdrop
            floor = self.project([[-10, -3, 0], [12, -3, 0], [12, 3, 0], [-10, 3, 0]])
            cv2.fillPoly(img, [floor.astype(np.int32)], 45)
        if not with_mat:
            return img
        hp = mat.PAGE_H_MM / 2000
        pages = [(-0.42, mat.stand_page())] + [(0.42 * i, mat.marker_page(i)) for i in range(4)]
        for x0, page in pages:
            src = cv2.resize(np.array(page), (840, 594), interpolation=cv2.INTER_AREA)
            src = (src.astype(np.float32) * 0.85 + 15).astype(np.uint8)  # paper isn't pure white
            dst = self.project([[x0, hp, 0], [x0 + 0.42, hp, 0], [x0 + 0.42, -hp, 0], [x0, -hp, 0]])
            H = cv2.getPerspectiveTransform(np.float32([[0, 0], [840, 0], [840, 594], [0, 594]]), dst.astype(np.float32))
            warped = cv2.warpPerspective(src, H, (self.w, self.h), flags=cv2.INTER_AREA)
            m = cv2.warpPerspective(np.full_like(src, 255), H, (self.w, self.h)) > 128
            img[m] = warped[m]
        return img


def render(out: Path, kmh=250.0, angle=-11.0, fps=240.0, duration=0.4, t_imp=0.1013, exposure=1 / 1000,
           L=4.6, seed=0, retime_fps: float | None = None, size=(1920, 1080), hide=(), hall=False,
           with_mat=True, cam=(1.2, -4.2, 1.3), shake_px: float = 0.0) -> dict:
    """hide: frame numbers where the shuttle is not drawn (occlusion / missed detections).
    hall: ordinary hall (light walls, lights, court lines, someone walking past) instead of a dark backdrop.
    """
    rng = np.random.default_rng(seed)
    sc = Scene(*size, cam=cam, target=(cam[0], 0.0, cam[2] - 0.15))
    bg = sc.static_background(hall, with_mat)
    p_imp = np.array([-0.15, 2.45])  # contact point (X, Z), above the STAND page
    v0 = kmh / 3.6
    ts_fl, ps_fl = simulate(v0, angle, L, duration)

    def shuttle(t):
        if t <= t_imp:  # falling slowly towards the racket
            return p_imp + np.array([0.15 * (t_imp - t), 3.0 * (t_imp - t)])
        x = np.interp(t - t_imp, ts_fl, ps_fl[:, 0])
        z = np.interp(t - t_imp, ts_fl, ps_fl[:, 1])
        return p_imp + np.array([x, z])

    n = int(round(duration * fps))
    frames = []
    # hand-held shake: random walk of the whole image, shake_px per frame (std)
    walk = np.cumsum(rng.normal(0, shake_px, (n, 2)), axis=0) if shake_px else np.zeros((n, 2))
    shoulder = np.array([-0.45, 0, 1.55])
    for i in range(n):
        t = i / fps
        img = bg.astype(np.float32)
        # player: grey body + swinging racket arm (distractors, not bright enough to be "shuttle")
        body = sc.project([[-0.75, 0, 0], [-0.35, 0, 0], [-0.3, 0, 1.6], [-0.8, 0, 1.6]])
        cv2.fillPoly(img, [body.astype(np.int32)], 70)
        ang = np.radians(np.clip(150 - 900 * (t - t_imp + 0.06), -60, 150))
        head = shoulder + 1.15 * np.array([np.sin(ang) * 0.6, 0, np.cos(ang)]) + [0, 0, 0]
        a, b = sc.project([shoulder, head])
        cv2.line(img, tuple(a.astype(int)), tuple(b.astype(int)), 55 if hall else 120, 6)
        if hall:  # someone walking past behind the court
            wx = -3 + 1.5 * t
            q = sc.project([[wx, 7, 0], [wx + 0.4, 7, 0], [wx + 0.4, 7, 1.7], [wx, 7, 1.7]])
            cv2.fillPoly(img, [q.astype(np.int32)], 75)
        # shuttle: white streak over the exposure, ~6 cm wide
        lay = np.zeros_like(img)
        sub = [shuttle(t + k * exposure / 4) for k in range(5)]
        uv = sc.project([[x, 0, z] for x, z in sub])
        r = max(2, int(round(0.03 * sc.K[0, 0] / 4.2)))
        for k in range(4):
            cv2.line(lay, tuple(np.round(uv[k]).astype(int)), tuple(np.round(uv[k + 1]).astype(int)), 235, 2 * r)
        if i not in hide:
            img = np.maximum(img, cv2.GaussianBlur(lay, (5, 5), 1.2))
        if shake_px:
            M = np.float32([[1, 0, walk[i, 0]], [0, 1, walk[i, 1]]])
            img = cv2.warpAffine(img, M, (sc.w, sc.h), borderMode=cv2.BORDER_REPLICATE)
        img += rng.normal(0, 3.0, img.shape)
        frames.append(np.clip(img, 0, 255).astype(np.uint8))

    out.parent.mkdir(parents=True, exist_ok=True)
    wav = out.with_suffix(".wav")
    dist = np.linalg.norm(sc.cam - np.array([p_imp[0], 0, p_imp[1]]))
    _write_audio(wav, n / fps, t_imp + dist / 343.0, rng)
    rate = retime_fps or fps
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found: brew install ffmpeg")
    cmd = [ffmpeg, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{sc.w}x{sc.h}", "-r", str(rate),
           "-i", "-"]
    if not retime_fps:
        cmd += ["-i", str(wav), "-c:a", "pcm_s16le"]
    cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", str(out)]
    subprocess.run(cmd, input=b"".join(f.tobytes() for f in frames), check=True)
    wav.unlink()
    truth = {"shake_max_px": float(np.abs(walk).max()), "hall": hall, "with_mat": with_mat, "kmh": kmh, "angle_deg": angle, "t_impact": t_imp, "fps": fps, "L": L, "frames": n,
             "impact_xz": p_imp.tolist(), "camera": sc.cam.tolist(), "focal_px": float(sc.K[0, 0]),
             "retime_fps": retime_fps,
             "shuttle_xz": {i: shuttle(i / fps).tolist() for i in range(n)}}
    out.with_suffix(".truth.json").write_text(json.dumps(truth))
    return truth


def _write_audio(path: Path, duration, t_hit, rng, rate=48000):
    a = rng.normal(0, 0.003, int(duration * rate))
    i = int(t_hit * rate)
    k = np.arange(int(0.004 * rate))
    a[i:i + len(k)] += 0.6 * np.exp(-k / (0.0008 * rate)) * rng.normal(0, 1, len(k))
    pcm = (np.clip(a, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "out" / "synthetic_smash.mov")
    ap.add_argument("--kmh", type=float, default=250.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hall", action="store_true", help="ordinary hall, no dark backdrop")
    ap.add_argument("--no-mat", action="store_true", help="leave the mat out (for the no-mat estimate)")
    ap.add_argument("--retime-fps", type=float, default=None,
                    help="write the 240 fps frames with this playback rate (mimics a slow-motion export)")
    a = ap.parse_args()
    t = render(a.out, kmh=a.kmh, seed=a.seed, retime_fps=a.retime_fps, hall=a.hall, with_mat=not a.no_mat)
    print(f"Wrote {a.out} ({t['frames']} frames, true speed {t['kmh']:.0f} km/h)")


if __name__ == "__main__":
    main()
