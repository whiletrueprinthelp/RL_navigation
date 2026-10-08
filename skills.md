---
name: robot-vision-alignment-rl
description: Reinforcement-learning playbook for visual alignment using only the live HTTP routes exposed by robot_server.py.
triggers:
  - "align target"
  - "center marker"
  - "run alignment episode"
  - "vision-guided control"
tools:
  - POST /move
  - POST /rotate
  - POST /stop
  - POST /camera/pan
  - GET /status
  - GET /image
  - GET /camera_info
version: 0.3.0
---

# Robot Vision Alignment RL Playbook (Server-Aligned)

## Objective
Use perception feedback to align a target with a reference marker in camera view while respecting all server-side safety constraints.

## Authentication Contract
- Every request must include header: `X-Token: <ROBOT_TOKEN>`.
- Command-level outcomes are read from JSON field `status`: `done | refused | aborted | busy`.
- HTTP `401` means auth failure (`bad or missing X-Token`).

## Canonical Command Set (Exact Server Routes)
Only the following routes are allowed in this skill.

### Motion Control
- `POST /move`
  - Body: `{ "direction": "forward|backward|left|right", "distance_m": <float>, "speed_mps": <float, optional> }`
  - Bounded translational move only (left/right are lateral translation, not rotation).
  - Distance is commanded by time (`distance ≈ speed × duration`), not measured odometry.
  - Typical response fields: `direction`, `commanded_distance_m`, `speed_mps`, `duration_s`, `heading_change_deg`.

- `POST /rotate`
  - Body: `{ "angle_deg": <float> }`
  - Gyro-closed-loop rotation. Positive angle is left, negative is right.
  - Refused until gyro bias is ready (robot kept still for about 3 seconds).
  - Typical response fields: `target_deg`, `measured_deg`, `error_deg`, `duration_s`.

- `POST /stop`
  - Body: `{}`
  - Immediate halt; use on safety trigger, timeout, manual override, and episode end.
  - Safe to call repeatedly.

- `POST /camera/pan`
  - Body: `{ "position": <int 0..1000>, "duration_s": <float, optional> }`
  - Controls pan servo only (servo id 1).
  - Use for scene scan/reacquisition and horizontal framing.

### Perception and State
- `GET /status`
  - Use before and between actions to check readiness and safety.
  - Returns: `state`, `moving_what`, `battery_mv`, `gyro_ready`, `gyro_bias_deg_s`, `heading_deg`, `age_s`, `limits`.

- `GET /image?width=<int optional>`
  - Returns latest RGB frame as base64 JPEG for local vision processing.
  - If `width` is provided and valid, server resizes while preserving aspect ratio.

- `GET /camera_info`
  - Returns intrinsics and image dimensions: `fx`, `fy`, `cx`, `cy`, `width`, `height`, `horizontal_fov_deg`.

## RL Observation Space (State)
Build `s_t` from server-backed signals:
- Marker pixel position and confidence from local vision on `GET /image`.
- Target pixel position and confidence from local vision on `GET /image`.
- Pixel alignment error `(ex, ey)` and normalized error `(ex/W, ey/H)`.
- Heading context from `GET /status.heading_deg`.
- Robot/safety context from `GET /status` (`battery_mv`, `age_s`, moving state, limits).
- Previous action metadata.

## RL Action Space (Mapped to Server Routes)
Define policy actions as wrappers over exact routes:
- `A_MOVE_FWD_SMALL` -> `POST /move {direction:"forward", distance_m:d_small}`
- `A_MOVE_BACK_SMALL` -> `POST /move {direction:"backward", distance_m:d_small}`
- `A_MOVE_LEFT_SMALL` -> `POST /move {direction:"left", distance_m:d_small}`
- `A_MOVE_RIGHT_SMALL` -> `POST /move {direction:"right", distance_m:d_small}`
- `A_ROTATE_LEFT_SMALL` -> `POST /rotate {angle_deg:+a_small}`
- `A_ROTATE_RIGHT_SMALL` -> `POST /rotate {angle_deg:-a_small}`
- `A_PAN_LEFT_SMALL` -> `POST /camera/pan {position:p_left}`
- `A_PAN_RIGHT_SMALL` -> `POST /camera/pan {position:p_right}`
- `A_HOLD` -> no motion command for one control tick
- `A_STOP` -> `POST /stop {}`

All action magnitudes must be clipped to `limits` returned by `GET /status`.

## Alignment Loop (Episode Workflow)
1. `GET /status` and verify robot is safe/ready.
2. `GET /image` and run marker/target detection locally.
3. Build state `s_t` from image detections + `/status`.
4. Select action `a_t` from policy.
5. Convert action to exact route call (`/move`, `/rotate`, `/camera/pan`, `/stop`, or hold).
6. Execute command and check `status` in response.
7. Re-observe (`/image`, `/status`) to build `s_(t+1)`.
8. Compute reward and `done` condition.
9. Store transition `(s_t, a_t, r_t, s_(t+1), done)`; update policy in training mode.
10. On terminal condition, call `POST /stop` and log episode summary.

## Reward Design (Template)
- Progress reward: decrease in alignment error between steps.
- Control penalty: discourage unnecessary motion and oscillation.
- Safety penalty: large negative reward for `refused|aborted|busy` or safety faults.
- Success bonus: when aligned within tolerance for `K_STABLE` consecutive steps.

## Hard Safety Rules
- If any motion command returns `refused|aborted|busy`, stop episode and call `POST /stop`.
- If `GET /status` indicates stale IMU (`age_s.imu`) or unsafe battery, do not move.
- If perception confidence is below threshold for `N_LOST_MAX` consecutive steps, run recovery; else fail episode safely.
- Always finish episode with `POST /stop`.

## Recovery Policy
When marker or target is lost:
- First use bounded `POST /camera/pan` adjustments.
- Then use small `POST /rotate` sweeps.
- If still unresolved after `RECOVERY_STEPS_MAX`, call `POST /stop` and terminate as failure.

## Logging Schema
Per-step log fields:
- `episode_id, step_idx, timestamp`
- `state_summary` (error/confidence/heading/safety)
- `action` and exact `route_called`
- `server_status` and `server_reason`
- `reward, done, done_reason`

Per-episode log fields:
- `episode_id, mode, status`
- `total_reward, steps, duration_s`
- `final_alignment_error_px`
- `termination_reason`

## Notes
- Marker/target detection runs locally on frames from `GET /image`; it is not a server route.
- There is no active depth/range API in the current server contract.
- There are no standalone heading routes (`GET /heading`, `POST /heading/reset`); heading is available in `GET /status`.
- This document defines behavior and interfaces; RL weights/replay buffers are separate runtime artifacts.
