"""Pre-computes the example results shown on the demo's first tab (same code path as an upload).

    .venv/bin/python demo/make_examples.py        # writes demo/examples/<name>.mp4 and .json
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import app   # noqa: E402

ROOT, OUT = app.ROOT, app.EXAMPLES
TITLES = {"1_real_fall_in_snow": "Real fall in the snow → alarm", "2_lies_down_on_purpose": "Lies down on purpose (drone, 10 m) → no alarm", "3_walk_and_lie_down": "Walks, hides, lies down (drone, 10 m) → no alarm"}
EX = [
    ("1_real_fall_in_snow", ROOT / "test_videos/drone/mixkit_man_falls_snow_23814.mp4", "oblique",
     "A man stumbles at the start and gets up by himself (no alarm), then at 17 s collapses into the snow and lies still: "
     "alarm 4 s later.", "Mixkit (free licence)"),
    ("2_lies_down_on_purpose", ROOT / "nomad/clips_4k/Actor042_a10_0-36s.mp4", "oblique",
     "Drone at 10 m. The person walks, squats and lies down by himself, slowly: shown as LYING (blue), no fall alarm.",
     "NOMAD dataset, actor 042 (A. M. Russell Bernal et al., WACV 2024, CC BY 4.0), first 36 s"),
    ("3_walk_and_lie_down", ROOT / "nomad/clips_4k/Actor099_a10_0-42s.mp4", "oblique",
     "Drone at 10 m. The person walks, hides in the grass and lies down on the sand by herself: no fall alarm.",
     "NOMAD dataset, actor 099 (A. M. Russell Bernal et al., WACV 2024, CC BY 4.0), first 42 s"),
]
OUT.mkdir(exist_ok=True)
for name, video, view, desc, credit in EX:
    web, alarms, checks, dur, took, _ = app.run_system(video, view, max_seconds=None)
    (OUT / f"{name}.mp4").write_bytes(Path(web).read_bytes())
    s = app.summary(alarms, checks, view, dur)
    (OUT / f"{name}.json").write_text(json.dumps({"title": TITLES[name], "summary": s, "description": desc, "credit": credit,
                                                  "alarms": alarms, "checks": checks}, ensure_ascii=False, indent=1))
    print(name, f"{dur:.0f} s video, {took:.0f} s, alarms {[a['alarm_s'] for a in alarms]}, checks {[c['check_s'] for c in checks]}")
