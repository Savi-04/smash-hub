# Smash Speed MVP (beta): phone app + Mac server

One iPhone films a smash; the app finds the shuttle, tracks its flight and shows the speed.
The phone runs a web app (add it to the Home Screen and it opens like an app); this Mac does the
analysis with the booth engine in `../booth/` (classical computer vision + the slowdown physics, no ML).

**Every speed is labelled Beta** until it has been checked against a radar gun on real clips.

## Run it

One-time: set up `booth/` first (see `booth/README.md`: `brew install ffmpeg`, venv, install).

```bash
booth/.venv/bin/python mvp/server.py
```

The terminal shows a link and a QR code. Point the iPhone camera at the QR code (same Wi-Fi as the
Mac), open the link, then Share → **Add to Home Screen**. The link contains a private key, stored in
`mvp/data/token.txt` so it survives restarts; delete that file to revoke old links.

**At events:** start with `--new-key` (old links stop working) and run on the Mac's or your iPhone's own
hotspot, not venue Wi-Fi. The app talks plain HTTP, so anyone on a shared network could read the link key
and the clips.

The first start takes ~5 s longer: it renders a synthetic **demo clip** (true speed 263 km/h) so the
app can be shown without filming anything. `--no-demo` turns that off; `--local` serves this Mac only.

## Using the app

1. **+** → choose **With mat (Verified)** or **No mat (Estimate)**; for no mat, enter the distance from
   the phone to the hitting line (pace it out).
2. Film the smash in the Camera app's **Slo-mo (240 fps)**, side-on, phone still, 3–4 m away.
   Trim the clip to the smash in Photos.
3. **Choose slo-mo video** → upload → result in a few seconds: speed, ± margin, Verified/Estimate,
   a speed-over-time chart, Share image.
4. If tracking fails (or looks wrong), **Tap the shuttle yourself**: tap it at contact and in the next
   4–6 frames (press and slide for a magnifier). The same physics fit runs on the taps.

## What it measures, honestly

| Setup | Margin on synthetic tests | Why |
|---|---|---|
| Mat in view (Verified) | about ±4–5 % | mat gives exact scale and camera position |
| No mat (Estimate) | about ±20 % | scale comes from your distance guess and an assumed lens |

- Real clips will be worse than synthetic ones: that is what the radar-gun test is for.
- The ± margin is computed for every smash; if the tracked path doesn't fit a smash, no number is shown.

## Privacy

- Results are stored **only on the phone** (browser storage); no accounts.
- The clip is uploaded to this Mac, kept only while you look at the result (for the tap fallback) and
  deleted when you close it, or after 15 minutes, or when the server stops.
- Frames for tapping are kept in memory only. No names, faces or photos are stored.
- Limits: clips over 12 s of capture (2,880 frames) are refused and deleted; at most 12 clips on the Mac
  at once; a clip that fails is deleted straight away; stalled connections time out after 30 s;
  ffprobe/ffmpeg are stopped after 120 s. Keep FFmpeg updated (`brew upgrade ffmpeg`): it reads
  files people upload.

## Known limits (to check at the club test)

- **iOS photo picker and Slo-mo:** we don't know yet whether Safari's video picker uploads the original
  240 fps timestamps or a re-timed 30 fps export. The server detects re-timing; with "Recorded in Slo-mo"
  on, it then assumes 240 fps capture (shown as "assumed" in the result). Verify with a known clip.
- Not tested on a real iPhone or a real badminton clip yet (only synthetic clips and a desktop browser
  at phone size). The two example videos in the repo are screen recordings of reels, not smashes.
- The Mac must be on the same Wi-Fi as the phone; venue Wi-Fi can block device-to-device traffic
  (fallback: the Mac's Personal Hotspot / Internet Sharing, or the iPhone's hotspot with the Mac joined).
- White walls or lights directly behind the flight hide a white shuttle.
- Plain HTTP on the local network (key-protected). A cloud server with HTTPS comes later.

## Tests

```bash
cd mvp && ../booth/.venv/bin/python -m pytest -q
```

## Files

| File | What |
|---|---|
| `server.py` | HTTP server: upload, job queue, tap fallback, auto-delete, QR code |
| `static/index.html` | the whole phone app (onboarding, home, new test, analysing, result, tap, settings) |
| `static/manifest.webmanifest`, `static/icon.svg` | Home Screen app |
| `tests/test_server.py` | API tests with synthetic clips |
| `data/` | demo clip + access key (git-ignored) |
