#!/usr/bin/env python3
"""robot_server.py

Safety-first HTTP bridge from JetAuto ROS topics to a small control API.

Runs on the robot and is the single authority for motion limits/safety checks.
The agent calls these endpoints over LAN and must provide `X-Token`.

Core routes
- POST /move, /rotate, /stop, /camera/pan
- GET  /status, /camera_info, /image

For full operator setup and run workflow, see README.md.
"""

import argparse
import atexit
import base64
import hmac
import json
import math
import os
import sys
import threading
import time
from collections import deque
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

# ---- ROS topics and camera intrinsics ---------------------------------------
CMD_TOPIC = '/controller/cmd_vel'
COLOR_TOPIC = '/depth_cam/rgb/image_raw'
IMU_TOPIC = '/imu_corrected'
BATT_TOPIC = '/ros_robot_controller/battery'
SERVO_TOPIC = '/ros_robot_controller/bus_servo/set_position'
FX = FY = 546.1217
CX, CY = 317.078, 240.1273
IMG_W, IMG_H = 640, 480

# ---- safety/actuation limits ------------------------------------------------
LIMITS = {
    'max_move_m': 0.30, 'max_speed_mps': 0.10, 'default_speed_mps': 0.08,
    'max_rotate_deg': 90.0, 'max_rotate_rate_rps': 0.30, 'min_rotate_rate_rps': 0.12,
    'min_battery_mv': 10000, 'abort_battery_mv': 10000, 'max_imu_age_s': 0.5,
    'pan_servo_id': 1, 'pan_min': 0, 'pan_max': 1000,
    'pan_default_duration_s': 0.08, 'pan_min_duration_s': 0.02, 'pan_max_duration_s': 2.0,
}
RATE_HZ = 20
SETTLE_S = 0.4              # wait after every motion so heading_change includes the coast
ROTATE_COAST_DEG = 1.0      # stop this many degrees early (tune from your tests)
BIAS_WINDOW_S = 3.0         # robot must be still this long for a gyro bias estimate
MAX_BIAS = 0.02             # rad/s, a larger resting mean means the robot is not still
QUIET_SPREAD = 0.10         # rad/s, max-min of gyro readings in the window
DIRS = {'forward': (1.0, 0.0), 'backward': (-1.0, 0.0), 'left': (0.0, 1.0), 'right': (0.0, -1.0)}

TOKEN = os.environ.get('ROBOT_TOKEN')
LOG_PATH = 'robot_server_log.jsonl'

# ---- gyro bias + heading ----------------------------------------------------
class GyroTracker:
    """Estimates the gyro bias while the robot is still and integrates heading."""

    def __init__(self):
        self.lock = threading.Lock()
        self.heading = 0.0      # radians, positive = left
        self.bias = None        # rad/s, None until the robot has been still for BIAS_WINDOW_S
        self._last_t = None
        self._win = deque()

    def clear_window(self):
        with self.lock:
            self._win.clear()

    def update(self, t, wz, moving):
        with self.lock:
            if not moving:
                self._win.append((t, wz))
                while self._win and t - self._win[0][0] > BIAS_WINDOW_S:
                    self._win.popleft()
                if len(self._win) >= 30 and t - self._win[0][0] >= 0.9 * BIAS_WINDOW_S:
                    vals = [w for _, w in self._win]
                    mean = sum(vals) / len(vals)
                    if abs(mean) < MAX_BIAS and max(vals) - min(vals) < QUIET_SPREAD:
                        self.bias = mean
            if self.bias is not None and self._last_t is not None and t > self._last_t:
                self.heading += (wz - self.bias) * (t - self._last_t)
            self._last_t = t

# ---- the ROS side ------------------------------------------------------------
class RosBridge:
    """Subscribes to the robot's topics and publishes velocity. Spins in a background thread.

    The server only needs: tracker, battery_mv, foreign_count, imu_age(), color(),
    publish_twist(), set_moving(), shutdown(). A fake with the same members can replace this
    class for testing (see test_robot_server.py)."""

    def __init__(self):
        import rclpy
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.node import Node
        from geometry_msgs.msg import Twist
        from sensor_msgs.msg import Image, Imu
        from std_msgs.msg import UInt16
        from ros_robot_controller_msgs.msg import ServosPosition, ServoPosition
        self._Twist = Twist
        self._ServosPosition = ServosPosition
        self._ServoPosition = ServoPosition
        self._rclpy = rclpy
        rclpy.init()    # must run before the web server starts, so the web server's Ctrl-C handler wins
        self.node = Node('robot_server')
        self.pub = self.node.create_publisher(Twist, CMD_TOPIC, 10)
        self.servo_pub = self.node.create_publisher(ServosPosition, SERVO_TOPIC, 10)
        self.tracker = GyroTracker()
        self.battery_mv = None
        self.foreign_count = 0
        self._moving = False
        self._imu_wall = None
        self._color = None      # (message, arrival time); converted only when requested
        self.node.create_subscription(Image, COLOR_TOPIC, self._on_color, 1)
        self.node.create_subscription(Imu, IMU_TOPIC, self._on_imu, 50)
        self.node.create_subscription(UInt16, BATT_TOPIC, self._on_batt, 10)
        self.node.create_subscription(Twist, CMD_TOPIC, self._on_foreign, 10)
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)

    def start(self):
        threading.Thread(target=self.executor.spin, daemon=True).start()

    def _on_color(self, m):
        self._color = (m, time.monotonic())

    def _on_batt(self, m):
        self.battery_mv = int(m.data)

    def _on_foreign(self, m):
        self.foreign_count += 1

    def _on_imu(self, m):
        self._imu_wall = time.monotonic()
        t = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
        self.tracker.update(t, m.angular_velocity.z, self._moving)

    def imu_age(self):
        return float('inf') if self._imu_wall is None else time.monotonic() - self._imu_wall

    def set_moving(self, moving):
        self._moving = moving
        if moving:
            self.tracker.clear_window()

    def publish_twist(self, vx, vy, wz):
        m = self._Twist()
        m.linear.x, m.linear.y, m.angular.z = float(vx), float(vy), float(wz)
        self.pub.publish(m)

    def publish_servo(self, servo_id, position, duration_s):
        m = self._ServosPosition()
        m.duration = float(duration_s)
        p = self._ServoPosition()
        p.id = int(servo_id)
        p.position = int(position)
        m.position = [p]
        self.servo_pub.publish(m)

    def color(self):
        """Returns (RGB array HxWx3, age in seconds) or None."""
        if self._color is None:
            return None
        m, wall = self._color
        a = np.frombuffer(m.data, np.uint8).reshape(m.height, m.step)[:, :m.width * 3]
        a = a.reshape(m.height, m.width, 3)
        if m.encoding == 'bgr8':
            a = a[:, :, ::-1]
        return a, time.monotonic() - wall

    def shutdown(self):
        try:
            self.executor.shutdown()
            self.node.destroy_node()
            self._rclpy.shutdown()
        except Exception:
            pass

# ---- shared state and helpers -----------------------------------------------
ros = None
motion_lock = threading.Lock()
stop_event = threading.Event()
state = {'moving': False, 'what': '', 'deadline': 0.0, 'pan': None}   # pan = last commanded position

def reply(status='done', reason='', **extra):
    """Build a normalized API response object.

    Args:
        status: Route outcome (`done|refused|aborted|busy`).
        reason: Optional human-readable reason.
        **extra: Additional route-specific response fields.

    Returns:
        Response dictionary serialized by FastAPI.
    """
    out = {'ok': status == 'done', 'status': status, 'reason': reason, 'time': time.time()}
    out.update(extra)
    return out

def logline(event, **data):
    """Append one JSONL event entry to the server log file.

    Args:
        event: Event label.
        **data: Event payload.
    """
    try:
        with open(LOG_PATH, 'a') as f:
            f.write(json.dumps(dict(time=time.time(), event=event, **data), default=str) + '\n')
    except OSError:
        pass

def check_token(x_token):
    """Reject requests without matching `X-Token`.

    Args:
        x_token: Token supplied by the HTTP header.

    Raises:
        HTTPException: 401 when token is missing or mismatched.
    """
    if not TOKEN or not x_token or not hmac.compare_digest(str(x_token), TOKEN):
        raise HTTPException(status_code=401, detail='bad or missing X-Token')

def send_zero(n=5, gap=0.03):
    """Publish repeated zero-velocity commands.

    Repetition reduces the chance of a single dropped publish leaving stale motion.

    Args:
        n: Number of zero commands to publish.
        gap: Sleep duration (seconds) between publishes.
    """
    for _ in range(n):
        try:
            ros.publish_twist(0.0, 0.0, 0.0)
        except Exception:
            pass
        time.sleep(gap)

def refuse_reason():
    """Why a new motion must not start now, or None."""
    b = ros.battery_mv
    if b is None:
        return 'no battery reading yet'
    if b < LIMITS['min_battery_mv']:
        return 'battery %d is below the %d minimum' % (b, LIMITS['min_battery_mv'])
    if ros.imu_age() > LIMITS['max_imu_age_s']:
        return 'no fresh gyro data'
    return None

def abort_reason():
    """Why a running motion must stop now, or None."""
    if stop_event.is_set():
        return 'stop requested'
    b = ros.battery_mv
    if b is not None and b < LIMITS['abort_battery_mv']:
        return 'battery %d fell below %d' % (b, LIMITS['abort_battery_mv'])
    if ros.imu_age() > LIMITS['max_imu_age_s']:
        return 'gyro data went stale'
    return None

class Busy(Exception):
    """Raised when a new motion is requested while another is active."""

    pass

class Motion:
    """Context manager: one motion at a time, always ends with zero velocity."""

    def __init__(self, what, max_s):
        self.what, self.max_s = what, max_s

    def __enter__(self):
        if not motion_lock.acquire(blocking=False):
            raise Busy()
        stop_event.clear()
        state.update(moving=True, what=self.what, deadline=time.monotonic() + self.max_s + 1.0)
        ros.set_moving(True)
        return self

    def __exit__(self, *exc):
        try:
            send_zero()
            time.sleep(SETTLE_S)
        finally:
            ros.set_moving(False)
            state['moving'] = False
            motion_lock.release()
        return False

def heading_deg_since(h0):
    """Return heading delta in degrees since `h0` (None until gyro bias is ready)."""
    return None if ros.tracker.bias is None else round(math.degrees(ros.tracker.heading - h0), 2)

# ---- the web app -------------------------------------------------------------
app = FastAPI(title='robot_server')

class MoveReq(BaseModel):
    """Payload for `POST /move`.

    Attributes:
        direction: One of `forward|backward|left|right`.
        distance_m: Positive commanded travel distance in meters.
        speed_mps: Optional commanded speed in meters/second.
    """

    direction: str
    distance_m: float
    speed_mps: Optional[float] = None

class RotateReq(BaseModel):
    """Payload for `POST /rotate`.

    Attributes:
        angle_deg: Signed commanded rotation angle in degrees.
    """

    angle_deg: float

class CamerapanReq(BaseModel):
    """Payload for `POST /camera/pan`.

    Attributes:
        position: Absolute servo position command in range `pan_min..pan_max`.
        duration_s: Optional command duration in seconds.
    """

    position: int
    duration_s: Optional[float] = None

def _watchdog():
    """If a motion overruns its deadline (for example a hung request), stop the robot."""
    while True:
        time.sleep(0.1)
        if state['moving'] and time.monotonic() > state['deadline']:
            stop_event.set()
            try:
                ros.publish_twist(0.0, 0.0, 0.0)
            except Exception:
                pass

def _final_zero():
    """Send a last zero-velocity burst during shutdown/exit."""
    if ros is not None:
        send_zero(3, 0.02)

@app.on_event('startup')
def _startup():
    """Start watchdog thread once web app is up."""
    threading.Thread(target=_watchdog, daemon=True).start()

@app.on_event('shutdown')
def _shutdown():
    """Stop active motion and send final zero command on shutdown."""
    stop_event.set()
    _final_zero()

@app.post('/move')
def move(req: MoveReq, x_token: Optional[str] = Header(None)):
    """Execute an open-loop translate command.

    Args:
        req: Direction, distance, and optional speed.
        x_token: Authentication header.

    Returns:
        JSON response with status, commanded motion details, and heading delta.
    """
    check_token(x_token)
    if req.direction not in DIRS:
        return reply('refused', 'direction must be forward, backward, left or right')
    speed = req.speed_mps if req.speed_mps is not None else LIMITS['default_speed_mps']
    if not 0 < req.distance_m <= LIMITS['max_move_m']:
        return reply('refused', 'distance_m must be above 0 and at most %.2f' % LIMITS['max_move_m'])
    if not 0 < speed <= LIMITS['max_speed_mps']:
        return reply('refused', 'speed_mps must be above 0 and at most %.2f' % LIMITS['max_speed_mps'])
    why = refuse_reason()
    if why:
        return reply('refused', why)
    # Open-loop move: execute commanded velocity for computed duration.
    duration, (sx, sy), abort = req.distance_m / speed, DIRS[req.direction], None
    try:
        with Motion('move', duration):
            h0, t0 = ros.tracker.heading, time.monotonic()
            while time.monotonic() - t0 < duration:
                abort = abort_reason()
                if abort:
                    break
                ros.publish_twist(sx * speed, sy * speed, 0.0)
                time.sleep(1.0 / RATE_HZ)
            elapsed = time.monotonic() - t0
    except Busy:
        return reply('busy', 'another motion is running (%s)' % state['what'])
    out = dict(direction=req.direction, commanded_distance_m=req.distance_m, speed_mps=speed,
               duration_s=round(elapsed, 2), heading_change_deg=heading_deg_since(h0),
               note='distance is commanded (speed x time), not measured')
    logline('move', abort=abort, **out)
    return reply('aborted', abort, **out) if abort else reply('done', **out)

@app.post('/rotate')
def rotate(req: RotateReq, x_token: Optional[str] = Header(None)):
    """Execute a closed-loop gyro-guided rotate command.

    Args:
        req: Signed angle command in degrees.
        x_token: Authentication header.

    Returns:
        JSON response with status and measured heading change.
    """
    check_token(x_token)
    a = req.angle_deg
    if not 0.5 <= abs(a) <= LIMITS['max_rotate_deg']:
        return reply('refused', 'abs(angle_deg) must be between 0.5 and %.0f' % LIMITS['max_rotate_deg'])
    why = refuse_reason()
    if why:
        return reply('refused', why)
    if ros.tracker.bias is None:
        return reply('refused', 'gyro bias not ready: keep the robot still for about %.0f s' % BIAS_WINDOW_S)
    sign, target = (1.0 if a > 0 else -1.0), math.radians(abs(a))
    # Coast offset reduces overshoot from drivetrain inertia.
    coast = math.radians(ROTATE_COAST_DEG)
    timeout = target / LIMITS['min_rotate_rate_rps'] + 3.0
    abort, reached = None, False
    try:
        with Motion('rotate', timeout):
            h0, t0 = ros.tracker.heading, time.monotonic()
            while time.monotonic() - t0 < timeout:
                abort = abort_reason()
                if abort:
                    break
                remaining = target - sign * (ros.tracker.heading - h0)
                if remaining <= coast:
                    reached = True
                    break
                rate = min(max(1.5 * remaining, LIMITS['min_rotate_rate_rps']), LIMITS['max_rotate_rate_rps'])
                ros.publish_twist(0.0, 0.0, sign * rate)
                time.sleep(1.0 / RATE_HZ)
            elapsed = time.monotonic() - t0
    except Busy:
        return reply('busy', 'another motion is running (%s)' % state['what'])
    measured = heading_deg_since(h0)
    out = dict(target_deg=a, measured_deg=measured, error_deg=None if measured is None else round(measured - a, 2),
               duration_s=round(elapsed, 2))
    if not abort and not reached:
        abort = 'timed out before reaching the angle (is the robot on the floor?)'
    logline('rotate', abort=abort, **out)
    return reply('aborted', abort, **out) if abort else reply('done', **out)

@app.post('/stop')
def stop(x_token: Optional[str] = Header(None)):
    """Request immediate stop.

    Args:
        x_token: Authentication header.

    Returns:
        JSON response confirming stop signal emission.
    """
    check_token(x_token)
    stop_event.set()
    was_moving = state['moving']
    for _ in range(3):
        ros.publish_twist(0.0, 0.0, 0.0)
    logline('stop', was_moving=was_moving)
    return reply('done', was_moving=was_moving)

@app.post('/camera/pan')
def camera_pan(req: CamerapanReq, x_token: Optional[str] = Header(None)):
    """Command absolute camera pan servo position.

    Args:
        req: Servo position and optional duration.
        x_token: Authentication header.

    Returns:
        JSON response with commanded pan details.
    """
    check_token(x_token)
    pos = req.position
    if not LIMITS['pan_min'] <= pos <= LIMITS['pan_max']:
        return reply('refused', 'position must be %d..%d' % (LIMITS['pan_min'], LIMITS['pan_max']))
    dur = float(req.duration_s) if req.duration_s is not None else LIMITS['pan_default_duration_s']
    if not LIMITS['pan_min_duration_s'] <= dur <= LIMITS['pan_max_duration_s']:
        return reply('refused', 'duration_s must be %.2f..%.2f' % (LIMITS['pan_min_duration_s'], LIMITS['pan_max_duration_s']))
    try:
        ros.publish_servo(LIMITS['pan_servo_id'], pos, dur)
    except Exception as e:
        return reply('refused', 'servo publish failed: %s' % e)
    state['pan'] = pos
    out = dict(servo_id=LIMITS['pan_servo_id'], position=pos, duration_s=round(dur, 3),
               note='position is commanded, not measured; the servo needs duration_s to arrive')
    logline('camera_pan', **out)
    return reply('done', **out)

@app.get('/status')
def status(x_token: Optional[str] = Header(None)):
    """Return server health snapshot and active limits.

    Args:
        x_token: Authentication header.

    Returns:
        JSON response with battery, gyro readiness, sensor ages, and limits.
    """
    check_token(x_token)
    c, b = ros.color(), ros.tracker.bias
    return reply('done', state='moving' if state['moving'] else 'idle',
                 moving_what=state['what'] if state['moving'] else '',
                 battery_mv=ros.battery_mv, gyro_ready=b is not None,
                 gyro_bias_deg_s=None if b is None else round(math.degrees(b), 3),
                 pan_position_commanded=state['pan'],
                 heading_deg=round(math.degrees(ros.tracker.heading), 2),
                 age_s={'imu': round(min(ros.imu_age(), 999.0), 3),
                        'color': None if c is None else round(c[1], 3)},
                 limits=LIMITS)

@app.get('/camera_info')
def camera_info(x_token: Optional[str] = Header(None)):
    """Return static camera intrinsics and derived horizontal FOV.

    Args:
        x_token: Authentication header.

    Returns:
        JSON response with camera matrix values and image size.
    """
    check_token(x_token)
    return reply('done', fx=FX, fy=FY, cx=CX, cy=CY, width=IMG_W, height=IMG_H,
                 horizontal_fov_deg=round(2 * math.degrees(math.atan(IMG_W / 2 / FX)), 1),)

@app.get('/image')
def image(width: Optional[int] = None, x_token: Optional[str] = Header(None)):
    """Return latest color frame as base64 JPEG.

    Args:
        width: Optional output width for server-side downscale.
        x_token: Authentication header.

    Returns:
        JSON response with JPEG payload and frame age.
    """
    check_token(x_token)
    got = ros.color()
    if got is None:
        return reply('refused', 'no colour frame yet')
    rgb, age = got
    bgr = np.ascontiguousarray(rgb[:, :, ::-1])
    if width and 32 <= width < bgr.shape[1]:
        bgr = cv2.resize(bgr, (width, int(bgr.shape[0] * width / bgr.shape[1])))
    ok, buf = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        return reply('refused', 'JPEG encoding failed')
    return reply('done', image_b64=base64.b64encode(buf.tobytes()).decode('ascii'),
                 width=int(bgr.shape[1]), height=int(bgr.shape[0]), age_s=round(age, 3), format='jpeg')

def main():
    """Boot ROS bridge, validate startup safety, then run FastAPI/uvicorn server."""
    global ros
    ap = argparse.ArgumentParser(description='HTTP control server for the JetAuto (run on the robot).')
    ap.add_argument('--host', default='0.0.0.0')
    ap.add_argument('--port', type=int, default=8081)
    ap.add_argument('--skip-foreign-check', action='store_true',
                    help='start even if another program seems to publish velocity')
    a = ap.parse_args()
    if not TOKEN:
        sys.exit('Set a secret first:  export ROBOT_TOKEN=choose-something')
    ros = RosBridge()
    ros.start()
    time.sleep(1.5)
    if ros.foreign_count and not a.skip_foreign_check:
        ros.shutdown()
        sys.exit('Another program is publishing to %s. Close any app game, or use --skip-foreign-check.' % CMD_TOPIC)
    if ros.battery_mv is None:
        print('warning: no battery reading yet')
    print('Gyro bias needs about %.0f s of stillness before /rotate is allowed.' % BIAS_WINDOW_S)
    atexit.register(_final_zero)
    import uvicorn
    uvicorn.run(app, host=a.host, port=a.port)

if __name__ == '__main__':
    main()
