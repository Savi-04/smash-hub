# Smash Hub (beta)

Measure a badminton smash with one phone. Film the smash in Slo-mo (240 fps). The shuttle's flight
is tracked frame by frame, and its starting speed comes from a shuttle slowdown model fitted to the
real frame timestamps, together with a ± margin.

**Beta:** speeds have only been checked on synthetic clips so far, not yet against a radar gun.

| Folder | What it is |
|---|---|
| [`booth/`](booth/README.md) | The measuring engine and a local event leaderboard. Python, OpenCV, FFmpeg, no ML. |
| [`mvp/`](mvp/README.md) | A phone web app plus a small server that runs on a Mac on the same Wi-Fi. |

## Quick start (macOS)

```bash
brew install ffmpeg
cd booth && python3 -m venv .venv && .venv/bin/pip install -r requirements.txt && cd ..
booth/.venv/bin/python mvp/server.py --new-key   # open the printed link on the iPhone
```

Tests: `cd booth && .venv/bin/python -m pytest -q`, then `cd mvp && ../booth/.venv/bin/python -m pytest -q`.

## Privacy

- Clips are deleted after analysis, or after 15 minutes at most.
- Results stay on the phone.
- The leaderboard shows nicknames only with the player's consent.

Copyright © 2026 Savi-04. All rights reserved.
