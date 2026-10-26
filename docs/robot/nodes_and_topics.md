# ROS Nodes and Topics Reference (`robot_server.py`)

This document lists the ROS node and topic dependencies used by `robot_server.py`, how each is used, and how to verify presence/health.

## Scope

- Focuses on ROS graph usage by `robot_server.py`.
- `aim_agent.py` is HTTP-only and does not create ROS nodes/topics directly.

## Node inventory

### Node created by `robot_server.py`

- `/robot_server`
  - Created by `Node('robot_server')`.
  - Owns all ROS pub/sub used by the HTTP bridge.

## Nodes needed for real operation

For `robot_server.py` to do more than just start, these upstream node groups must also be running:

- `depth_cam` camera pipeline node(s)
  - Must publish `sensor_msgs/msg/Image` on `/depth_cam/rgb/image_raw`.
  - Without this, `GET /image` returns `refused` (`no colour frame yet`).

- `ros_robot_controller` sensor/control node(s)
  - Must publish `sensor_msgs/msg/Imu` on `/imu_corrected`.
  - Must publish `std_msgs/msg/UInt16` on `/ros_robot_controller/battery`.
  - Must consume `geometry_msgs/msg/Twist` from `/controller/cmd_vel`.
  - Must consume `ros_robot_controller_msgs/msg/ServosPosition` from `/ros_robot_controller/bus_servo/set_position`.
  - Without IMU/battery streams, motion routes are typically refused for safety.

Note: exact upstream node names can differ by launch/config; the required topics and message types above are the operational contract.

## Topic inventory (comprehensive)

### 1) `/controller/cmd_vel`

- Type: `geometry_msgs/msg/Twist`
- `robot_server` role:
  - publisher (motion output for `/move`, `/rotate`, `/stop`)
  - subscriber (detects foreign publishers that may conflict)
- Used by HTTP routes:
  - `POST /move`
  - `POST /rotate`
  - `POST /stop`

### 2) `/ros_robot_controller/bus_servo/set_position`

- Type: `ros_robot_controller_msgs/msg/ServosPosition`
- `robot_server` role:
  - publisher (camera pan servo command)
- Used by HTTP route:
  - `POST /camera/pan`

### 3) `/depth_cam/rgb/image_raw`

- Type: `sensor_msgs/msg/Image`
- `robot_server` role:
  - subscriber (latest color frame cache)
- Used by HTTP route:
  - `GET /image`
- Also affects:
  - `GET /status` (`age_s.color`)

### 4) `/imu_corrected`

- Type: `sensor_msgs/msg/Imu`
- `robot_server` role:
  - subscriber (gyro bias estimation + heading integration + freshness checks)
- Used by HTTP routes:
  - `POST /rotate` (closed-loop heading control)
  - `POST /move` (safety precheck and abort checks)
- Also affects:
  - `GET /status` (`gyro_ready`, `gyro_bias_deg_s`, `heading_deg`, `age_s.imu`)

### 5) `/ros_robot_controller/battery`

- Type: `std_msgs/msg/UInt16`
- `robot_server` role:
  - subscriber (battery safety precheck/abort)
- Used by HTTP routes:
  - `POST /move` (refuse/abort logic)
  - `POST /rotate` (refuse/abort logic)
- Also affects:
  - `GET /status` (`battery_mv`)

## Route-to-topic dependency matrix

- `GET /status`
  - reads cached values from `/imu_corrected`, `/depth_cam/rgb/image_raw`, `/ros_robot_controller/battery`
- `GET /camera_info`
  - no live ROS topic dependency (static intrinsics/constants)
- `GET /image`
  - requires `/depth_cam/rgb/image_raw`
- `POST /move`
  - publishes to `/controller/cmd_vel`
  - safety checks depend on `/imu_corrected` and `/ros_robot_controller/battery`
- `POST /rotate`
  - publishes to `/controller/cmd_vel`
  - closed-loop control depends on `/imu_corrected`
  - safety checks depend on `/ros_robot_controller/battery`
- `POST /camera/pan`
  - publishes to `/ros_robot_controller/bus_servo/set_position`
- `POST /stop`
  - publishes zero twists to `/controller/cmd_vel`

## Verification checklist

For a more comprehensive check, run these on the robot in another ROS2-sourced terminal, separate from the one used to run `robot_server.py`.

### A) Confirm node presence

```bash
ros2 node list
ros2 node info /robot_server
```

Expected:

- `/robot_server` appears in node list.
- Node info shows pubs/subs that include the topics listed above.

### B) Confirm topics and types

```bash
ros2 topic list -t
ros2 topic info -v /controller/cmd_vel
ros2 topic info -v /ros_robot_controller/bus_servo/set_position
ros2 topic info -v /depth_cam/rgb/image_raw
ros2 topic info -v /imu_corrected
ros2 topic info -v /ros_robot_controller/battery
```

Expected:

- Each topic exists.
- Types match this document.
- `/robot_server` appears as expected pub/sub endpoint per topic.

### C) Confirm data freshness on subscribed inputs

```bash
ros2 topic hz /depth_cam/rgb/image_raw
ros2 topic hz /imu_corrected
ros2 topic echo --once /ros_robot_controller/battery
```

Expected:

- camera and IMU streams show non-zero update rates.
- battery echo returns one numeric sample.

### D) Confirm end-to-end HTTP bridge

```bash
curl -H "X-Token: <token>" http://localhost:8081/status
curl -H "X-Token: <token>" http://localhost:8081/image
curl -X POST -H "Content-Type: application/json" -H "X-Token: <token>" -d '{"angle_deg":5}' http://localhost:8081/rotate
curl -X POST -H "Content-Type: application/json" -H "X-Token: <token>" -d '{}' http://localhost:8081/stop
```

Expected:

- `status`/`image` return `{"ok":true,"status":"done",...}`.
- `rotate` returns `done` when gyro is ready (or clear `refused` reason if not ready).
- `stop` returns `done`.

## Common verification failure patterns


- Topic exists but no data:
  - upstream driver node is down or camera/IMU hardware path is unavailable.
- `/move`/`/rotate` returns `refused`:
  - low battery, stale IMU, or gyro bias not ready.
- Startup exits on foreign publisher warning:
  - another app is publishing on `/controller/cmd_vel`.

