# Smash-speed booth (v1, **beta**)

Free smash-speed test for tournaments and clubs. An iPhone films a smash side-on at 240 fps over a
printed calibration mat; the clip goes to the Mac; `speed.py` measures the shuttle's starting speed in a
few seconds; a local leaderboard page (laptop or TV) shows nickname + speed.

**Every result is "beta" until the setup has been checked against a radar gun** (see "Next" in `PROGRESS.md`).

No ML. It works because the setup is controlled: white shuttle, dark backdrop, known mat, known camera
position, real frame timestamps and the shuttle slowdown formula from `initial_plan.md` §11.8.

---

## 1. One-time install (Mac)

```bash
brew install ffmpeg
```

```bash
cd booth && python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

Check it works (about 30 s; renders synthetic smashes with known speeds and measures them):

```bash
cd booth && .venv/bin/python -m pytest -q
```

Print the mat (5 × A3 pages):

```bash
cd booth && .venv/bin/python mat.py
```

→ `booth/out/mat.pdf`. Print at **100 % / "Actual size"** (never "fit to page"), on matt paper if you can
(glossy reflects the hall lights). **Check with a ruler** on every page:

| What | Must measure |
|---|---|
| Black marker square (pages 1–4) | **210 mm** per side |
| Scale bar (every page) | **100 mm** |
| Page | A3, 420 × 297 mm |

If a marker is off by more than 1 mm, reprint with the scaling fixed: every mm of error is ~0.5 % of speed.

---

## 2. Venue setup checklist

```
        backdrop (dark cloth, behind the hitting area)
  ───────────────────────────────────────────────────────────
     [STAND] [ 1 ] [ 2 ] [ 3 ] [ 4 ]   ← mat pages, edge to edge, on the floor
     player   hit this way  ──────►     "CAMERA SIDE" edges all on one straight line
                     ▲
                     │ 3–4 m, square to the mat
                  [iPhone on tripod, ~1–1.5 m high, landscape]
```

- [ ] **Mat:** STAND page, then pages 1, 2, 3, 4 **edge to edge** in one straight row (use a long straight
      edge or a chalk line). All "CAMERA SIDE" edges face the phone. Tape the corners flat: no curling.
- [ ] **Backdrop:** dark, matt cloth behind the whole flight path (from the STAND page to ~4 m past page 4,
      up to ~3 m high). The shuttle must be the brightest moving thing in the picture.
- [ ] **Phone:** tripod, **3–4 m from the mat line**, looking square at it, **~1–1.5 m high**, landscape.
      Camera app → **Slo-mo, 1080p at 240 fps** (Settings → Camera → Record Slo-mo).
- [ ] **Framing:** the whole mat is visible near the bottom of the frame; someone standing on the STAND
      page with the racket at full reach (~2.7 m) is fully in frame at the left; there is ~3 m of flight
      space to the right. If the markers are not found, raise the phone towards 1.5 m or move it closer.
- [ ] **Lighting check:** no bright lights, windows or white walls behind the flight path; no flicker
      (if the slo-mo shows pulsing brightness, move away from that light).
- [ ] **Setup check** (empty mat, 2-second test clip):

      ```bash
      cd booth && .venv/bin/python speed.py ~/Downloads/TEST.MOV --check
      ```

      Must say `SETUP OK: 4/4 markers ...`. This also saves the camera position, used later if a player
      stands on a marker. **Run it again whenever the tripod is moved.**
- [ ] **Test shots:** 3 smashes by the operator; results should be consistent (± a few %).
- [ ] **Backup:** second phone, power bank for the phone, Mac charger, spare printed mat, tape.
- [ ] **Leaderboard** running on the Mac (section 3) and shown on the laptop or the TV.

Players: stand on the STAND page, side-on, hit **along the row of pages** (towards page 4), contact above the
STAND page. Self-feed or a feeder throws high from the far side.

---

## 3. Running the booth

Start the leaderboard once (keep this terminal open):

```bash
cd booth && .venv/bin/python leaderboard.py --event "Club Night"
```

- **Board (TV / laptop):** http://localhost:8000/ — refreshes itself every 3 s.
- **Operator page (this Mac only):** http://localhost:8000/operator
- To show the board on another device on the same Wi-Fi, add `--lan`; the operator page still only works on the Mac.
- Each `--event` name gets its own board file in `booth/data/boards/`.

For each player:

1. Record the smash in Slo-mo (start 1 s before the hit, stop 1 s after; short clips are faster).
2. Send it to the Mac: **AirDrop → tap "Options" → turn on "All Photos Data"** (sends the original 240 fps
   file), or a cable with the Image Capture app. The clip lands in `~/Downloads`.
3. Measure:

   ```bash
   cd booth && .venv/bin/python speed.py ~/Downloads/IMG_1234.MOV
   ```

   Prints e.g. `SPEED (beta): 251 ± 9 km/h [auto, 23 points, 3.1 s]` and puts the result on the operator page.
4. On the operator page: type the player's **nickname** (no full names), tick the **consent** box only if the
   player (or a parent/guardian if under 16) agrees, press **Add to board**. Without consent the board shows
   "Player N" and the nickname is not stored.
5. **Delete the clip** from the Mac afterwards (it shows the player's face). The booth never copies videos.

### If there is no automatic result

`speed.py` opens a **click window** by itself when automatic tracking fails or the fit is refused:

1. Step to the frame where the racket meets the shuttle (`d` / `.` next, `a` / `,` back, `f` / `s` ±10 frames).
2. Click the shuttle **at contact**, then click it in the next **4–6 frames** (each click moves one frame on).
3. `z` undoes, **Enter** finishes, Esc cancels.

The same slowdown fit runs on the clicks; the board marks it "manual". Clicks are saved to
`booth/data/clicks/` and can be re-run with `--clicks FILE`. Force the window with `--manual`.

### Other options

| Option | Use |
|---|---|
| `--shuttle nylon` | nylon shuttles (uses the feather drag length for now and flags it: not measured yet) |
| `--capture-fps 240` | only if the clip's timestamps were re-timed (see Known limits) |
| `--audio` | use the hit sound to time the impact when the racket hides the shuttle at contact |
| `--av-offset-ms N` | phone audio/video offset from a clap test (with `--audio`) |
| `--depth-tol 0.15` | how far (m) the shuttle may fly off the mat line; part of the ± margin |
| `--debug-image` | saves a frame with the track drawn to `booth/out/debug/` (shows the player: delete after) |
| `--json` | full result as JSON |
| `--no-mat 3.5` | no mat in view: **Estimate** from the phone-to-hitting-line distance in metres (about ±20 %) |

Camera shake (hand-held phone, wobbly stand) is measured and corrected automatically (`shake_px` in `--json`).
The detector no longer needs the dark backdrop: synthetic tests include a normal hall (light walls, ceiling
lights, someone walking past). A backdrop still helps on real clips.

---

## 4. How it works

1. **Timestamps:** real per-frame times from `ffprobe` (not the nominal frame rate). Clips below 100 fps
   or with mixed frame intervals are refused unless `--capture-fps` is given (a slow-motion export plays
   240 fps frames at ~30 fps, which would make every smash 8× too slow).
2. **Mat → camera:** ArUco markers (OpenCV) give the lens focal length and the camera's exact position;
   every pixel is then converted to metres on the vertical **hitting plane** above the mat line.
3. **Shuttle:** bright pixels that differ from the clip's median background, small blobs only, above the
   floor. The smash is the longest fast (40–500 km/h), straight, not-rising track; the shuttle seen just
   before it is the **contact point**.
4. **Fit:** distance from contact along the track vs. time, fitted with d(t) = L·ln(1 + v₀·(t − t₀)/L),
   L = 4.6 m (feather), solving for v₀ and the sub-frame impact time t₀. Outliers are dropped.
5. **Margin (±):** 2 × fit uncertainty, combined with the scale uncertainty from the shuttle flying up to
   `--depth-tol` off the mat line (≈ 3.5 % at 4 m). **No number is shown** if: fewer than 6 points
   (4 manual), tracked distance < 0.6 m, residuals > 3 cm, margin > 10 %, impact time outside the
   contact frames, or speed outside 50–480 km/h.

On synthetic clips (1080p, 240 fps, drag + gravity, motion blur, noise, H.264) the error is about
**+0.5 %** (max 1.2 % over 16 random smashes) against a ±3–4 % reported margin. Real clips will be worse:
that is what the radar-gun test is for.

## 5. Known limits (v1)

- **Not validated on a real clip yet.** Beta label stays until the club test with a radar gun.
- **iPhone slo-mo re-timing:** we have not yet checked which iOS transfer methods keep the 240 fps
  timestamps. AirDrop with "All Photos Data" or Image Capture are expected to; verify with `--check`
  (should say ~240 fps) at the first club test.
- **Depth:** one camera can't see how far the shuttle is off the mat line; ±15 cm is assumed (in the margin).
  A player who hits diagonally gets a wrong number.
- **Markers on the floor are seen at a steep angle;** small or far phones may miss them. The last good
  setup is reused, but run `--check` after any tripod change.
- **Detection is brightness-based:** white socks, shoes, shirts or lights behind the flight path can
  confuse it (outliers are dropped; otherwise use the click window).
- **No rolling-shutter correction** yet (§11.5); small at 240 fps, to be measured.
- **Nylon shuttles:** drag length not measured; results flagged.
- **Audio timing** needs a clap test for the phone's audio/video offset; without it only use `--audio`
  as a cross-check.
- The click window has been tested through saved clicks only, not yet by hand on the venue Mac.

## 6. Files

| File | What |
|---|---|
| `mat.py` | mat geometry + printable PDF |
| `speed.py` | the measurement (CLI) |
| `video.py` | timestamps, re-timing check, frames, audio |
| `geometry.py` | marker detection, camera pose, pixel → metres |
| `track.py` | shuttle detection and the smash track |
| `physics.py` | slowdown model and fit |
| `manual.py` | click window |
| `leaderboard.py`, `web/` | local leaderboard |
| `synth.py` | synthetic clips with known answers (for tests) |
| `tests/` | known-answer tests |
| `data/`, `out/` | runtime files (git-ignored: nicknames, poses, PDFs, clips) |
