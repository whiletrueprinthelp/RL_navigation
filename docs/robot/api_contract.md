# HTTP API Contract

This document defines the HTTP contract used by `aim_agent.py` to control the robot through `robot_server.py`.

## Scope and conventions

- Base URL is typically `http://<robot-ip>:8081`.
  - port 8080 is used by the camera on the robot to send the live feed back
- Every request must include header `X-Token: <ROBOT_TOKEN>`.
- Units:
  - distances in metres (`distance_m`)
  - angular values in degrees (`angle_deg`, heading fields)
  - positive rotation means left turn
- Endpoint outcomes use API-level statuses in JSON:
  - `done`: command accepted and completed
  - `refused`: command rejected before or at start (validation/safety/precondition)
  - `aborted`: command started, then stopped for safety/runtime reason
  - `busy`: another motion is already active
- `401` is reserved for missing/invalid token.

## Common response envelope

All endpoints return this envelope plus route-specific fields:

```json
{
  "ok": true,
  "status": "done",
  "reason": "",
  "time": 1760000000.123
}
```

Field semantics:

- `ok`: `true` only when `status == "done"`.
- `status`: one of `done|refused|aborted|busy`.
- `reason`: human-readable explanation (especially for non-`done`).
- `time`: server-side UNIX timestamp (seconds).

## Authentication

Every route below requires `X-Token`.

### `bash` / `zsh` (Linux/macOS/Git Bash)

```bash
curl -H "X-Token: <token>" http://<robot-ip>:8081/status
```

### PowerShell (Windows)

```powershell
curl.exe -H "X-Token: <token>" "http://<robot-ip>:8081/status"
```

### Command Prompt (`cmd.exe`, Windows)

```cmd
curl -H "X-Token: <token>" http://<robot-ip>:8081/status
```

On token mismatch/missing, server returns HTTP `401`.

## Endpoints

### `GET /status`

Returns health, motion state, sensor freshness, heading, and active limits.

Response-specific fields (with `status=done`):

- `state`: `idle|moving`
- `moving_what`: active motion label or empty string
- `battery_mv`: battery reading in mV, or `null`
- `gyro_ready`: boolean (gyro bias estimated)
- `gyro_bias_deg_s`: bias estimate in deg/s, or `null`
- `pan_position_commanded`: last commanded pan position
- `heading_deg`: integrated heading estimate
- `age_s`: object with `imu` and `color` ages
- `limits`: full active safety/actuation limits object

### `GET /camera_info`

Returns static camera intrinsics and derived horizontal FOV.

Response-specific fields (with `status=done`):

- `fx`, `fy`, `cx`, `cy`
- `width`, `height`
- `horizontal_fov_deg`

### `GET /image`

Returns latest RGB frame as base64 JPEG.

Query params:

- `width` (optional): downscale target width; applied only when `32 <= width < source_width`.

Response-specific fields (with `status=done`):

- `image_b64`: base64-encoded JPEG payload
- `width`, `height`: returned image size
- `age_s`: frame age in seconds
- `format`: currently `jpeg`

Possible refusal examples:

- `no colour frame yet`
- `JPEG encoding failed`

### `POST /move`

Open-loop chassis translation (`speed x time`), not distance-measured feedback.

Request body:

```json
{
  "direction": "forward",
  "distance_m": 0.1,
  "speed_mps": 0.08
}
```

Validation and safety:

- `direction` must be one of `forward|backward|left|right`.
- `0 < distance_m <= limits.max_move_m`.
- `0 < speed_mps <= limits.max_speed_mps` (defaults to `limits.default_speed_mps` when omitted).
- May be `refused` when battery/IMU preconditions fail.
- May return `busy` if another motion is active.
- May return `aborted` if stop/safety fault occurs during motion.

Response-specific fields:

- `direction`
- `commanded_distance_m`
- `speed_mps`
- `duration_s`
- `heading_change_deg`
- `note` (`distance is commanded (speed x time), not measured`)

### `POST /rotate`

Gyro-closed-loop in-place rotation.

Request body:

```json
{
  "angle_deg": 12.0
}
```

Validation and safety:

- `0.5 <= abs(angle_deg) <= limits.max_rotate_deg`.
- Requires gyro bias readiness (`gyro_ready=true`), otherwise `refused`.
- May return `busy` if another motion is active.
- May return `aborted` on stop request, safety fault, or rotate timeout.

Response-specific fields:

- `target_deg`
- `measured_deg`
- `error_deg` (`measured_deg - target_deg`)
- `duration_s`

### `POST /camera/pan`

Open-loop camera pan servo command (commanded position is tracked; measured servo feedback is not provided).

Request body:

```json
{
  "position": 500,
  "duration_s": 0.1
}
```

Validation:

- `limits.pan_min <= position <= limits.pan_max`
- `limits.pan_min_duration_s <= duration_s <= limits.pan_max_duration_s`
- `duration_s` defaults to `limits.pan_default_duration_s` when omitted

Response-specific fields:

- `servo_id`
- `position`
- `duration_s`
- `note` (`position is commanded, not measured; the servo needs duration_s to arrive`)

### `POST /stop`

Asynchronous stop request.

Behavior:

- sets server stop flag
- publishes repeated zero-velocity commands

Response-specific fields:

- `was_moving`: whether a motion was active when stop was requested

## Non-functional safety semantics

- Server enforces one active motion at a time.
- Motion commands are bounded by active limits from `/status`.
- Safety checks include battery threshold and IMU freshness.
- A watchdog can trigger stop on overrun.
- Startup may fail if foreign publishers are detected on `/controller/cmd_vel`.

## Minimal verification calls

### `bash` / `zsh` (Linux/macOS/Git Bash)

```bash
curl -H "X-Token: <token>" http://<robot-ip>:8081/status
curl -H "X-Token: <token>" http://<robot-ip>:8081/camera_info
curl -H "X-Token: <token>" "http://<robot-ip>:8081/image?width=640"
curl -X POST -H "Content-Type: application/json" -H "X-Token: <token>" \
  -d '{"angle_deg":5}' http://<robot-ip>:8081/rotate
curl -X POST -H "Content-Type: application/json" -H "X-Token: <token>" \
  -d '{}' http://<robot-ip>:8081/stop
```

### PowerShell (Windows)

```powershell
curl.exe -H "X-Token: <token>" "http://<robot-ip>:8081/status"
curl.exe -H "X-Token: <token>" "http://<robot-ip>:8081/camera_info"
curl.exe -H "X-Token: <token>" "http://<robot-ip>:8081/image?width=640"
curl.exe -X POST -H "Content-Type: application/json" -H "X-Token: <token>" -d '{"angle_deg":5}' "http://<robot-ip>:8081/rotate"
curl.exe -X POST -H "Content-Type: application/json" -H "X-Token: <token>" -d "{}" "http://<robot-ip>:8081/stop"
```

### Command Prompt (`cmd.exe`, Windows)

```cmd
curl -H "X-Token: <token>" http://<robot-ip>:8081/status
curl -H "X-Token: <token>" http://<robot-ip>:8081/camera_info
curl -H "X-Token: <token>" "http://<robot-ip>:8081/image?width=640"
curl -X POST -H "Content-Type: application/json" -H "X-Token: <token>" -d "{\"angle_deg\":5}" http://<robot-ip>:8081/rotate
curl -X POST -H "Content-Type: application/json" -H "X-Token: <token>" -d "{}" http://<robot-ip>:8081/stop
```

### Expected output (all environments)

The exact numbers change per robot state.

For a healthy idle system, these checks typically return JSON with `"ok": true` and `"status": "done"`.

Some checks can validly return non-`done` statuses depending on runtime conditions (for example no camera frame yet, gyro not ready, or motion slot busy). In those cases, use `status` + `reason` to interpret the result.

- `GET /status`
  - Expected shape (example):

```json
{
  "ok": true,
  "status": "done",
  "reason": "",
  "state": "idle",
  "battery_mv": 12000,
  "gyro_ready": true,
  "heading_deg": 0.0,
  "age_s": {"imu": 0.02, "color": 0.04},
  "limits": {"max_move_m": 0.3, "max_rotate_deg": 90.0}
}
```

- `GET /camera_info`
  - Expected shape (example):

```json
{
  "ok": true,
  "status": "done",
  "fx": 546.1217,
  "fy": 546.1217,
  "cx": 317.078,
  "cy": 240.1273,
  "width": 640,
  "height": 480,
  "horizontal_fov_deg": 60.8
}
```

- `GET /image?width=640`
  - Expected shape (example):

```json
{
  "ok": true,
  "status": "done",
  "image_b64": "<very long base64 jpeg>",
  "width": 640,
  "height": 480,
  "age_s": 0.05,
  "format": "jpeg"
}
```

- `POST /rotate {"angle_deg":5}`
  - Expected shape (example):

```json
{
  "ok": true,
  "status": "done",
  "target_deg": 5,
  "measured_deg": 4.9,
  "error_deg": -0.1,
  "duration_s": 0.5
}
```

- `POST /stop {}`
  - Expected shape (example):

```json
{
  "ok": true,
  "status": "done",
  "was_moving": false
}
```

Typical failure outputs:

- Token mismatch: HTTP `401` with `{"detail":"bad or missing X-Token"}`.
- Safety refusal: `{"ok":false,"status":"refused","reason":"..."}`.
- Busy motion slot: `{"ok":false,"status":"busy","reason":"another motion is running (...)"}`.

