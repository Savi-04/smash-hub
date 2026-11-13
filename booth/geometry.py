"""Camera pose from the calibration mat, and pixel -> metres in the hitting plane (Y = 0)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

import mat

NOMINAL_HFOV_DEG = 66.0  # starting guess only; the focal length is solved from the mat


def detect_markers(gray: np.ndarray) -> dict[int, np.ndarray]:
    """Mat markers found in a grayscale image: {id: 4x2 corners (TL, TR, BR, BL)}."""
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(mat.ARUCO_DICT), params)
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        return {}
    return {int(i): c.reshape(4, 2) for i, c in zip(ids.ravel(), corners) if int(i) in mat.MARKER_IDS}


@dataclass
class CameraPose:
    K: np.ndarray  # 3x3 intrinsics
    rvec: np.ndarray
    tvec: np.ndarray
    image_size: tuple[int, int]  # (width, height)
    reproj_px: float
    n_markers: int

    @property
    def R(self) -> np.ndarray:
        return cv2.Rodrigues(self.rvec)[0]

    @property
    def centre(self) -> np.ndarray:
        """Camera position in mat coordinates (metres)."""
        return (-self.R.T @ self.tvec.reshape(3)).astype(float)

    @property
    def plane_distance(self) -> float:
        """Distance from the camera to the hitting plane Y = 0."""
        return abs(float(self.centre[1]))

    def to_plane(self, uv) -> np.ndarray:
        """Pixel(s) (N x 2) -> (X, Z) in metres on the hitting plane Y = 0."""
        uv = np.atleast_2d(np.asarray(uv, dtype=float))
        rays = np.linalg.inv(self.K) @ np.vstack([uv.T, np.ones(len(uv))])
        d = self.R.T @ rays  # ray directions in mat coordinates
        c = self.centre
        lam = -c[1] / d[1]
        p = c[:, None] + lam * d
        return np.stack([p[0], p[2]], axis=1)

    def project(self, xyz) -> np.ndarray:
        pts, _ = cv2.projectPoints(np.asarray(xyz, float).reshape(-1, 3), self.rvec, self.tvec, self.K, None)
        return pts.reshape(-1, 2)

    def px_per_m(self) -> float:
        """Approximate image scale on the hitting plane near the mat centre."""
        x = mat.marker_centre_m(1)[0]
        a, b = self.project([[x, 0, 1.0], [x + 1.0, 0, 1.0]])
        return float(np.linalg.norm(b - a))

    def to_json(self) -> dict:
        return {"K": self.K.tolist(), "rvec": self.rvec.ravel().tolist(), "tvec": self.tvec.ravel().tolist(),
                "image_size": list(self.image_size), "reproj_px": self.reproj_px, "n_markers": self.n_markers}

    @classmethod
    def from_json(cls, d: dict) -> "CameraPose":
        return cls(np.array(d["K"]), np.array(d["rvec"]).reshape(3, 1), np.array(d["tvec"]).reshape(3, 1),
                   tuple(d["image_size"]), d["reproj_px"], d["n_markers"])

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json(), indent=1))


def nominal_K(width: int, height: int, hfov_deg: float = NOMINAL_HFOV_DEG) -> np.ndarray:
    f = 0.5 * width / np.tan(np.radians(hfov_deg) / 2)
    return np.array([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1.0]])


def estimate_pose(markers: dict[int, np.ndarray], image_size: tuple[int, int]) -> CameraPose | None:
    """Solve focal length + pose from the mat (principal point centred, no distortion)."""
    if len(markers) < 2:
        return None
    ids = sorted(markers)
    obj = np.concatenate([mat.marker_corners_m(i) for i in ids]).astype(np.float32)
    img = np.concatenate([markers[i] for i in ids]).astype(np.float32)
    w, h = image_size
    K0 = nominal_K(w, h)
    flags = (cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_FIX_ASPECT_RATIO
             | cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3)
    try:
        rms, K, _, rvecs, tvecs = cv2.calibrateCamera([obj], [img], (w, h), K0.copy(), np.zeros(5), flags=flags)
    except cv2.error:
        return None
    f = K[0, 0]
    if not (0.3 * K0[0, 0] < f < 3 * K0[0, 0]):  # degenerate view: fall back to the nominal lens
        ok, rvec, tvec = cv2.solvePnP(obj, img, K0, None)
        if not ok:
            return None
        K, rvecs, tvecs = K0, [rvec], [tvec]
        proj, _ = cv2.projectPoints(obj, rvec, tvec, K0, None)
        rms = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - img) ** 2, axis=1))))
    pose = CameraPose(K, rvecs[0], tvecs[0], (w, h), float(rms), len(ids))
    if pose.centre[1] >= 0 or pose.centre[2] <= 0:  # camera must be on the camera side, above the floor
        return None
    return pose


def pose_from_distance(image_size: tuple[int, int], distance_m: float, height_m: float = 1.2,
                       K: np.ndarray | None = None) -> CameraPose:
    """No mat: assume the phone faces the hitting line square-on and level, `distance_m` away.

    Lens: `K` from an earlier mat calibration of the same phone/resolution if given, else the
    nominal field of view. Both assumptions go into the wider "Estimate" margin.
    """
    w, h = image_size
    K = nominal_K(w, h) if K is None else np.asarray(K, float)
    C = np.array([0.0, -float(distance_m), float(height_m)])
    R = np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])  # x right, y down, z forward (+Y)
    rvec = cv2.Rodrigues(R)[0]
    return CameraPose(K, rvec, (-R @ C).reshape(3, 1), (w, h), float("nan"), 0)
