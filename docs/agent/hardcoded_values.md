# `aim_agent.py` hardcoded values and assumptions

This document tracks the fixed defaults and code-level assumptions in the current `aim_agent.py` implementation.

## 1) Module-level constants

- `DEFAULT_HFOV_DEG = 60.8`
  - Fallback horizontal field-of-view when `/camera_info` is unavailable.

- `MIN_ROTATE_DEG = 0.5`
  - Local minimum accepted rotate command. Smaller values are refused and the model is nudged toward pan.

- `MOTION_ROUTES = ("POST /rotate", "POST /camera/pan", "POST /move")`
  - Route set used to classify actions as movement for reward/control accounting.

- `MOVE_ACTIONS`
  - LLM action-name to server direction map:
  - `move_forward -> forward`
  - `move_backward -> backward`
  - `move_left -> left`
  - `move_right -> right`

- `ACTIONS`
  - Allowed action vocabulary exposed to the model:
  - `rotate_robot`, `pan_camera`, movement actions, `hold`, `stop`, `finish`

## 2) Runtime fallback/default behavior in code

- Robot client request timeout: `30.0s` (`Robot(..., timeout=30.0)`).
- Frame fetch staleness policy: `grab_frame(max_age=0.6, tries=5)` with retry sleep `0.2s`.
- Grid overlay spacing (`--grid`): `every=80` pixels.
- Grid JPEG quality: `85`.
- Safety fallback IMU age threshold when server omits it: `0.5s`.
- Gyro-wait timeout: `wait_gyro(timeout=20.0)` with `1.0s` polling.
- Initial local pan position before sync: `State.pan_pos = 500`.

Fallbacks applied when `/status` limits are missing:

- Pan range: `0..1000`
- Max rotate: `90.0 deg`
- Max move: `0.30 m`
- Max speed: `0.08 m/s`
- Pan duration clamp: `0.02..2.0 s`

## 3) Search and alignment heuristics

- `scan_plan` minimum step clamp: `10.0 deg`
- `scan_plan` maximum step: `min(0.85 * hfov, server_max_rot)`
- Number of scan turns: `ceil(laps * (360 - hfov) / step)`
- Pan recenter duration before search: `min(1.0, pan_dur * 3 + 0.7)`
- Startup settle after initial center pan: `1.3 s`
- Move request minimum accepted in executor: `0.01 m`

## 4) Learning / calibration fit heuristics

- Outlier-removal pass runs only when `len(samples) >= 6`.
- MAD scale factor: `1.4826`.
- Outlier keep condition: `abs(residual - median) <= 3 * MAD`.
- Minimum retained points after outlier pass: `max(4, len(pts)//2)`.
- Direction agreement threshold: `>= 0.75`.
- Relative-error confidence divisor: `0.4`.
- Sample-count confidence target: `2 * min_samples`.
- “Big move” subset for behavior analysis: starting offset `> 2 * tol`.
- Overshoot criterion: sign crossed and landed beyond `max(2*tol, 0.25*|start|)`.
- Overshoot damping pivot: `overshoot / 0.3`.

Policy hints in generated `learned.md` also use fixed values:

- Initial pan probe when uncalibrated: `40 units`
- Unverified pan prior text assumption: `1000 units ~= 240 deg` (if prior enabled)
- Correction share guidance:
  - high overshoot (`>= 0.3`) -> use `60%`
  - otherwise -> `70% + 30% * confidence`
- Behavior verdict thresholds:
  - overshoot warning at `overshoot >= 0.3`
  - under-correcting warning when `closed < 0.6` and `overshoot < 0.15`
  - behavior diagnosis needs at least `3` big-move samples

## 5) Reward shaping constants and mode-specific terms

Reward terms are computed in `finalize(...)`.

- Movement cost: `-control_penalty` (default configurable; see CLI defaults).
- Safety penalties:
  - server-side failure: `-10.0`
  - local refusal: `-1.0`
- Oscillation penalty: `-oscillation_penalty` when estimated offset flips sign while still outside tolerance.
- Success bonus: `+10.0`.

Mode-specific term:

- `shoot`: progress term `((abs_err_before - abs_err_after) / image_width) * 100`
- `track`: tracking term `-(abs_err_after / image_width) * 100`

## 6) Prompt/schema assumptions fixed in code

- Tool name exposed to Responses API is fixed: `act`.
- Offset convention is fixed in schema/instructions: `offset = target_x - marker_x`.
- Marker reference is fixed to the pointer tip (lowest point).
- Prompt block always includes explicit run parameters and mode note (`shoot` vs `track`).

## 7) CLI defaults (hardcoded in `build_parser()`)

General:

- `--run-mode shoot`
- `--skills skills.md`
- `--learned learned.md`
- `--no-learn` disabled by default
- `--learn-window 20`
- `--min-samples 5`
- `--min-rot-deg 2.0`
- `--min-pan-units 15.0`
- `--pan-prior-deg-per-unit 0.24`
- `--max-steps 25`
- `--tol 10.0`
- `--k-stable 1`
- `--max-rotate 10.0`
- `--max-rotate-ceiling 25.0`
- `--max-pan 60.0`
- `--max-pan-ceiling 250.0`
- `--max-move 0.10`
- `--pan-duration 0.3`
- `--settle 0.3`
- `--n-lost-max 3`
- `--scan-step 50.0`
- `--scan-laps 1.0`
- `--scan-dir auto`
- `--scan-settle 0.5`
- `--acquire-frames 2`
- `--max-search-steps 30`
- `--max-local-faults 4`
- `--control-penalty 0.05`
- `--oscillation-penalty 0.2`
- `--width None`
- `--history 6`
- `--effort low`
- `--detail high`
- `--grid` off
- `--no-prev` off
- `--no-prior` off
- `--out runs`

Shoot-only options:

- `--attempts-per-round 2`
- `--free-search` off
- `--max-rounds 5`
- `--object-move manual`
- `--move-wait 10.0`

Track-only options:

- `--track-seconds 180.0`
- `--min-interval 0.0`
- `--end-on-aligned` off
- `--deadband-guard` off

## 8) Run artifacts/schema assumptions

- Logging schema currently uses `schema_version = 2`.
- Every step record includes `run_mode` to separate shoot and track analysis.
- Every run writes `run_config.json` with the effective parsed CLI values.

