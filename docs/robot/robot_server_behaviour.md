# `robot_server.py` behavior notes

This document contains information on `robot_server.py`. 

## Current role in the system

- Runs **on the robot** and is the only component that touches ROS topics for agent control.
- Exposes the HTTP contract used by `aim_agent.py`:
  - `POST /move`, `POST /rotate`, `POST /stop`, `POST /camera/pan`
  - `GET /status`, `GET /camera_info`, `GET /image`
- Requires `X-Token` on every request (`401` if missing/wrong).
- Returns JSON for all route outcomes with a status in `{done, refused, aborted, busy}`.

## Safety model (current behavior)

Safety is intentionally centralized in `robot_server.py`; callers are never trusted to enforce bounds.

- Per-call motion limits are enforced via `LIMITS`.
- Exactly one motion can run at a time (`Motion` context + `motion_lock`).
- Every motion path ends with zero-velocity publish and settle delay.
- A watchdog thread sets stop and pushes zero twist if a deadline overruns.
- Motion start can be refused on:
  - low battery (`min_battery_mv`)
  - stale gyro (`max_imu_age_s`)
- Running motion can be aborted on:
  - stop request
  - battery drop below `abort_battery_mv`
  - stale gyro during motion
- Startup can block if foreign publishers are detected on `/controller/cmd_vel` (unless `--skip-foreign-check` is passed).

## Motion semantics and why they are reported this way

- `/move` is open-loop (`speed x time`); distance is commanded, not measured.
- `/rotate` is gyro-closed-loop:
  - requires gyro bias estimate from stillness window (`BIAS_WINDOW_S`)
  - applies coast-stop offset (`ROTATE_COAST_DEG`) to reduce overshoot
  - reports `measured_deg` and `error_deg` from integrated heading
- `/camera/pan` is open-loop servo command:
  - stores commanded pan position in server state
  - does not claim measured servo feedback
- `/stop` is asynchronous and designed to be safe even during active motion.

## Sensor and camera assumptions

- Color frames are read from `COLOR_TOPIC` and exposed via `/image` as JPEG base64.
- `/image?width=...` only downsizes frames (never upsizes) to control bandwidth/latency.
- `/camera_info` serves static intrinsics (`FX`, `FY`, `CX`, `CY`, `IMG_W`, `IMG_H`) and derived horizontal FOV.
- `/status` reports health, heading, sensor ages, gyro readiness, and current active limits so the client can gate behavior.

## Operator startup reminders

- Run this script in a ROS2-sourced environment.
- Ensure no competing app publishes movement commands.
- Export `ROBOT_TOKEN` before launching.
- Keep an emergency stop path available for physical tests.


