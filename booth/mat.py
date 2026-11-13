"""Calibration mat for the smash-speed booth: geometry + printable PDF.

The mat is 4 A3 landscape pages, each with one ArUco marker, laid edge to edge
in a row on the floor, plus a 5th "STAND HERE" page at the left end.

Mat coordinates (metres):
  X  along the row of pages, left -> right as seen from the camera (hit direction)
  Y  away from the camera (towards the top edge of each page)
  Z  up
  Origin: left paper edge of marker page 1, on the hit line (middle of the page height).
The hitting plane is the vertical plane Y = 0 (through all marker centres).

Run:  python mat.py [--out out/mat.pdf]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ARUCO_DICT = cv2.aruco.DICT_4X4_50
MARKER_IDS = (0, 1, 2, 3)

# Paper and print geometry in millimetres. 254 dpi = exactly 10 px per mm.
PAGE_W_MM, PAGE_H_MM = 420, 297  # A3 landscape
MARKER_MM = 210  # black square side (incl. the 1-cell black border); 6 cells of 35 mm
DPI = 254
PX_PER_MM = 10


def marker_centre_m(i: int) -> tuple[float, float]:
    """Centre of marker page i (0-based) in mat coordinates (metres)."""
    return ((i * PAGE_W_MM + PAGE_W_MM / 2) / 1000.0, 0.0)


def marker_corners_m(marker_id: int) -> np.ndarray:
    """3D corners (4x3, metres) in OpenCV's ArUco order: TL, TR, BR, BL of the marker image.

    The marker image is printed upright on the page, page top = +Y (away from camera).
    """
    i = MARKER_IDS.index(marker_id)
    cx, cy = marker_centre_m(i)
    h = MARKER_MM / 2000.0
    return np.array(
        [[cx - h, cy + h, 0], [cx + h, cy + h, 0], [cx + h, cy - h, 0], [cx - h, cy - h, 0]],
        dtype=np.float64,
    )


def _font(size_px: int):
    for path in ("/System/Library/Fonts/Helvetica.ttc", "/System/Library/Fonts/Supplemental/Arial.ttf"):
        try:
            return ImageFont.truetype(path, size_px)
        except OSError:
            pass
    return ImageFont.load_default(size=size_px)


def _mm(v: float) -> int:
    return int(round(v * PX_PER_MM))


def _arrow(d: ImageDraw.ImageDraw, x0, y, x1, width_mm=6):
    d.line([(_mm(x0), _mm(y)), (_mm(x1 - 12), _mm(y))], fill=0, width=_mm(width_mm))
    d.polygon([(_mm(x1), _mm(y)), (_mm(x1 - 18), _mm(y - 9)), (_mm(x1 - 18), _mm(y + 9))], fill=0)


def _ruler(d: ImageDraw.ImageDraw, x0, y):
    """100 mm scale bar with 10 mm ticks, for checking the print scale."""
    d.line([(_mm(x0), _mm(y)), (_mm(x0 + 100), _mm(y))], fill=0, width=_mm(0.6))
    for k in range(11):
        tick = 4 if k % 5 else 7
        d.line([(_mm(x0 + 10 * k), _mm(y)), (_mm(x0 + 10 * k), _mm(y - tick))], fill=0, width=_mm(0.5))
    d.text((_mm(x0 + 103), _mm(y - 6)), "100 mm", fill=0, font=_font(_mm(5)))


def marker_page(i: int) -> Image.Image:
    img = Image.new("L", (_mm(PAGE_W_MM), _mm(PAGE_H_MM)), 255)
    d = ImageDraw.Draw(img)
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    marker = cv2.aruco.generateImageMarker(dictionary, MARKER_IDS[i], _mm(MARKER_MM), borderBits=1)
    left = _mm((PAGE_W_MM - MARKER_MM) / 2)
    top = _mm((PAGE_H_MM - MARKER_MM) / 2)
    img.paste(Image.fromarray(marker), (left, top))
    small, big = _font(_mm(6)), _font(_mm(9))
    d.text((_mm(12), _mm(8)), f"MAT PAGE {i + 1}/4   (marker id {MARKER_IDS[i]})", fill=0, font=big)
    d.text((_mm(12), _mm(PAGE_H_MM - 16)), "CAMERA SIDE  (this edge faces the phone)", fill=0, font=small)
    # hit direction arrows left and right of the marker
    _arrow(d, 15, PAGE_H_MM / 2, (PAGE_W_MM - MARKER_MM) / 2 - 10)
    _arrow(d, (PAGE_W_MM + MARKER_MM) / 2 + 10, PAGE_H_MM / 2, PAGE_W_MM - 15)
    d.text((_mm(15), _mm(PAGE_H_MM / 2 + 10)), "HIT", fill=0, font=small)
    _ruler(d, PAGE_W_MM - 145, 18)
    d.text((_mm(PAGE_W_MM - 160), _mm(PAGE_H_MM - 16)), f"black square = {MARKER_MM} mm", fill=0, font=small)
    return img


def stand_page() -> Image.Image:
    img = Image.new("L", (_mm(PAGE_W_MM), _mm(PAGE_H_MM)), 255)
    d = ImageDraw.Draw(img)
    big, small = _font(_mm(22)), _font(_mm(7))
    d.text((_mm(20), _mm(25)), "STAND HERE", fill=0, font=big)
    # two footprints (simple ovals), body side-on to the camera
    for fx in (120, 230):
        d.ellipse([_mm(fx), _mm(110), _mm(fx + 90), _mm(150)], outline=0, width=_mm(3))
    d.text((_mm(20), _mm(175)), "Hit the shuttle towards the marker pages  ->", fill=0, font=small)
    d.text((_mm(20), _mm(190)), "Contact point: above this page, in line with the marker centres.", fill=0, font=small)
    _arrow(d, 20, 230, PAGE_W_MM - 20, width_mm=8)
    d.text((_mm(12), _mm(PAGE_H_MM - 16)), "CAMERA SIDE  (this edge faces the phone)", fill=0, font=small)
    _ruler(d, PAGE_W_MM - 145, 18)
    return img


def write_pdf(out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    pages = [stand_page()] + [marker_page(i) for i in range(len(MARKER_IDS))]
    pages[0].save(out, "PDF", resolution=float(DPI), save_all=True, append_images=pages[1:])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "out" / "mat.pdf")
    args = ap.parse_args()
    path = write_pdf(args.out)
    print(f"Wrote {path}  (5 pages, A3 landscape {PAGE_W_MM}x{PAGE_H_MM} mm)")
    print("Print at 100% / 'Actual size' (no fit-to-page). Check with a ruler:")
    print(f"  - black marker square: {MARKER_MM} mm per side")
    print("  - scale bar on every page: 100 mm")
    print(f"Lay out: STAND page, then pages 1-4 edge to edge (marker centres every {PAGE_W_MM} mm),")
    print("all 'CAMERA SIDE' edges along one straight line facing the phone.")


if __name__ == "__main__":
    main()
