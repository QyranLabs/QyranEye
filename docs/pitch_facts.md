# QyranEye — verified facts for the presentation (26 Nov 2026)

Use these numbers and wordings as they are. Everything here is measured (`eval/report.py`, commit 7ea5f0e/1e50f34)
or stated as a plan. If a slide needs a number that is not here, ask the team first — do not estimate.

## One-line pitch
QyranEye watches drone video and raises an alarm when a person falls and stays down, so rescuers check that
spot first instead of watching hours of footage. A person always makes the final decision.

## The problem (safe wording)
- After a search flight, rescuers have to watch long drone videos by eye; a small person on the ground is easy to miss.
- A fall followed by lying still is the moment that matters: the person may be injured or unconscious.
- (If you add statistics about accidents or search and rescue in Kazakhstan, cite the source on the slide.)

## The solution — how it works (4 steps)
1. Finds people in every frame (YOLO26 neural network) and tracks each one, correcting for the drone's movement.
2. Reads posture: standing / in between / lying — from the person's shape and, for larger people, body keypoints.
3. Fall or lying down? Speed of the shoulders relative to the feet: fast = fall, slow = lying down on purpose.
4. Confirms: seen lying and not moving for 4 s → **ALARM**. Lying still 10 s without a fall → **CHECK** for the operator.

## Working prototype
- Website: https://conrad-qyran.pages.dev (soon https://qyraneye.space)
- Upload a drone video (up to 30 s) → annotated video + list of alarms in 1–5 minutes, on a free cloud server.
- Three ready examples on the page.

## Measured results (say "development results")
| What | Result |
|---|---|
| People found by the detector, 11 test videos (Okutama-Action drone dataset) | 86 % |
| Correct status for people found (standing vs lying) | 87 % |
| False alarms on public drone videos checked (11 + 7 Okutama, 3 NOMAD, 55 MOD20 clips) | **0** |
| Real fall in the snow video | alarm 4 s after the fall (21.0 s) |
| NOMAD drone videos, people lying down on purpose | no false alarm |
| Speed of a fall vs lying down (shoulders, body heights per second) | falls 1.5 and 2.3; lying down 0.83–0.84; squatting 0.92; walking ≤ 0.85 |
| Automated scenario tests | 61 of 62 scenarios pass (the 1 failing is a known limit) |

## Honest limits (put at least the first two on a slide)
- **Only one real fall video so far** — public drone datasets contain no falls. No accuracy claim for real falls yet.
- These are development numbers: the same videos were used to build the system.
- Hard cases: very small people (< 80 px, high altitude), straight-down camera, people hidden in grass.
- Lying down quickly on purpose looks like a fall — the operator decides.

## Next steps (plan, not done)
- Own filming with a FIMI X8 SE drone, 5–20 m: 36 staged falls + 72 normal actions, 3 people;
  scored once on 2 people the system was never tuned on; plus 30–60 min of normal activity for false alarms per hour.
- Detector fine-tuned for the straight-down view (tools written, training not done yet).
- Pilot with a search-and-rescue team (to be arranged).

## Technology and licences
- Python, YOLO26 (Ultralytics, AGPL-3.0), BoT-SORT tracking, our own fall logic.
- Datasets used for testing: Okutama-Action (non-commercial licence), NOMAD (CC BY 4.0), MOD20.
- Commercial use later would need: an Ultralytics licence (or another detector) and our own training data.

## Team
[Name — role], [Name — role], [Name — role]
