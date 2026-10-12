#!/usr/bin/env python3
"""aim_agent.py - LLM-in-the-loop aiming, driven by skills.md. Perception is the LLM alone.

Runs on your laptop/server, NOT on the robot. It talks to robot_server.py over HTTP and to
Azure OpenAI over the Responses API.

Setup:
    pip install requests openai python-dotenv   # add opencv-python numpy only if you use --grid

    Create a file named .env in the directory you run this from (and keep it out of git):
        ROBOT_URL=http://<robot-ip>:8081
        ROBOT_TOKEN=<same secret as on the robot>
        AZURE_OPENAI_ENDPOINT=https://<resource>.services.ai.azure.com/openai/v1
        AZURE_OPENAI_API_KEY=<your Azure key/token>
        AZURE_OPENAI_DEPLOYMENT=<your deployment name, e.g. gpt-5.3-codex>

    The script loads .env automatically at startup (load_dotenv). Variables already
    exported in the shell take priority over the .env values.

    Put skills.md next to this script (or pass --skills PATH).

    python3 aim_agent.py              # learns from earlier runs in runs/ (default)
    python3 aim_agent.py --no-learn   # baseline: no learned notes, base step caps

Learning from past runs (all code-derived, the LLM never writes the notes):
    At startup the script reads runs/*/log.jsonl, pairs every executed rotate/pan with the
    offset estimates before and after it, and fits how many pixels of offset one degree of
    rotation / one pan unit moves (robust least squares, outliers dropped). It also measures
    how much of the error each correction closed and how often it overshot. The result is
    written to learned.md (overwritten after every run) and added to the system prompt.
    Confidence in the calibration (0..1) raises the per-step caps from the base values
    (--max-rotate, --max-pan) towards the ceilings (--max-rotate-ceiling, --max-pan-ceiling),
    and overshooting lowers it again. The server's own limits always apply on top.

How a run works (follows the skills.md episode workflow):
    1. GET /status safety check, GET /camera_info, centre the pan servo.
    2. Every step: GET /status (stale IMU / low battery -> stop), GET /image, send the
       frame(s) to the LLM, the LLM calls act() with an action, the action is clipped to
       the limits from /status and run on the exact server route.
    3. Hard rules: the first server refused|aborted|busy ends the episode; the episode
       always ends with POST /stop.
    4. Success = the LLM confirms alignment on K_STABLE consecutive steps.

Perception is the LLM only: it finds the marker and target in the image and estimates the
offset. The reward is therefore computed from the LLM's own estimates (a noisy proxy).

Output in runs/<episode_id>/:  step_NNN.jpg frames, log.jsonl (one record per step, with
reward, done, done_reason and next_state), episode_summary.json.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import statistics
import sys
import time
import traceback
from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()   # loads ROBOT_URL, ROBOT_TOKEN and the AZURE_OPENAI_* settings from .env (current directory)

DEFAULT_HFOV_DEG = 60.8                  # used only if /camera_info is unavailable
MIN_ROTATE_DEG = 0.5                     # server minimum for /rotate
MOTION_ROUTES = ("POST /rotate", "POST /camera/pan", "POST /move")
MOVE_ACTIONS = {"move_forward": "forward", "move_backward": "backward",
                "move_left": "left", "move_right": "right"}
ACTIONS = ["rotate_robot", "pan_camera", *MOVE_ACTIONS, "hold", "stop", "finish"]


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


# ---------------------------------------------------------------- robot client
class Robot:
    def __init__(self, base_url: str, token: str, timeout: float = 30.0):
        self.base = base_url.rstrip("/")
        self.timeout = timeout
        self.s = requests.Session()
        self.s.headers["X-Token"] = token

    def _call(self, method, path, **kw):
        r = self.s.request(method, self.base + path, timeout=self.timeout, **kw)
        if r.status_code == 401:
            sys.exit("Robot rejected the token (check ROBOT_TOKEN).")
        r.raise_for_status()
        return r.json()

    def status(self):
        return self._call("GET", "/status")

    def camera_info(self):
        return self._call("GET", "/camera_info")

    def image(self, width: Optional[int] = None):
        return self._call("GET", "/image", params={"width": width} if width else None)

    def rotate(self, deg: float):
        return self._call("POST", "/rotate", json={"angle_deg": deg})

    def move(self, direction: str, distance_m: float, speed_mps: float):
        return self._call("POST", "/move", json={"direction": direction, "distance_m": distance_m,
                                                 "speed_mps": speed_mps})

    def pan(self, position: int, duration_s: float):
        return self._call("POST", "/camera/pan",
                          json={"position": int(position), "duration_s": duration_s})

    def stop(self):
        return self._call("POST", "/stop", json={})


def safe_stop(robot: Robot):
    try:
        robot.stop()
    except Exception as e:  # never let a failed stop hide the original error
        print("warning: /stop failed:", e)


def grab_frame(robot: Robot, width: Optional[int], max_age: float = 0.6, tries: int = 5):
    """Return (jpeg_bytes, meta). Retries a few times to avoid a stale frame."""
    meta = None
    for _ in range(tries):
        meta = robot.image(width)
        if meta.get("status") == "done" and meta.get("age_s", 99) <= max_age:
            return base64.b64decode(meta["image_b64"]), meta
        time.sleep(0.2)
    if meta and meta.get("status") == "done":
        print("warning: using a stale frame, age_s =", meta.get("age_s"))
        return base64.b64decode(meta["image_b64"]), meta
    raise RuntimeError("could not get a frame: %s" % (meta,))


def add_grid(jpeg: bytes, every: int = 80) -> bytes:
    """Optional overlay: vertical lines labelled with x pixel coordinates."""
    import cv2
    import numpy as np
    img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    for x in range(0, w, every):
        cv2.line(img, (x, 0), (x, h), (255, 255, 255), 1)
        cv2.putText(img, str(x), (x + 2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes() if ok else jpeg


def make_caps(lim: dict, args, L: Optional["Learned"] = None) -> dict:
    """Effective action limits: the tighter of the server's limits (from /status) and my caps.
    Rotate and pan caps start at the base values and grow towards the ceilings with the
    learned calibration confidence (0 without history = base caps, same as before)."""
    pmin, pmax = int(lim.get("pan_min", 0)), int(lim.get("pan_max", 1000))
    rot_hi, pan_hi = max(args.max_rotate, args.max_rotate_ceiling), max(args.max_pan, args.max_pan_ceiling)
    c_rot, c_pan = (L.rot.conf, L.pan.conf) if L else (0.0, 0.0)
    rot = args.max_rotate + c_rot * (rot_hi - args.max_rotate)
    pan = args.max_pan + c_pan * (pan_hi - args.max_pan)
    return dict(
        rot=round(min(rot, float(lim.get("max_rotate_deg", 90.0))), 1),
        move=min(args.max_move, float(lim.get("max_move_m", 0.30))),
        speed=float(lim.get("default_speed_mps", 0.08)),
        pan_min=pmin, pan_max=pmax, pan_center=(pmin + pmax) // 2, pan_step=int(round(pan)),
        rot_base=args.max_rotate, rot_ceil=rot_hi, pan_base=args.max_pan, pan_ceil=pan_hi,
        pan_dur=clamp(args.pan_duration, float(lim.get("pan_min_duration_s", 0.02)),
                      float(lim.get("pan_max_duration_s", 2.0))),
    )


def safety_reason(st: dict) -> Optional[str]:
    """skills.md: stale IMU or unsafe battery -> do not move."""
    lim = st.get("limits") or {}
    b = st.get("battery_mv")
    if b is None:
        return "no battery reading"
    if b < lim.get("min_battery_mv", 0):
        return "battery %s mV is below the %s mV minimum" % (b, lim.get("min_battery_mv"))
    imu = (st.get("age_s") or {}).get("imu")
    if imu is None or imu > lim.get("max_imu_age_s", 0.5):
        return "stale IMU data (age %s s)" % imu
    return None


def wait_gyro(robot: Robot, timeout: float = 20.0) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if robot.status().get("gyro_ready"):
            return True
        time.sleep(1.0)
    return False


# -------------------------------------------------------------------- learning
@dataclass
class Calib:
    n: int = 0
    k: Optional[float] = None          # change of the offset (px) per degree / per pan unit
    rel_err: Optional[float] = None    # standard error of k, relative to |k|
    agree: Optional[float] = None      # share of samples whose direction matches k
    conf: float = 0.0                  # 0..1: how far to trust k


@dataclass
class Learned:
    rot: Calib = field(default_factory=Calib)
    pan: Calib = field(default_factory=Calib)
    episodes: int = 0
    successes: int = 0
    med_steps_ok: Optional[float] = None
    fails: Counter = field(default_factory=Counter)
    big_moves: int = 0                 # corrections made while the offset exceeded 2 x tolerance
    overshoot: Optional[float] = None  # share of those that flipped past the marker
    closed: Optional[float] = None     # median share of the error closed by those corrections
    text: str = ""


def read_runs(out_dir, window: int, exclude: Optional[str] = None):
    """[(episode_id, log rows, summary or None)] for the newest `window` episodes."""
    dirs = sorted(d for d in Path(out_dir).glob("*") if (d / "log.jsonl").exists() and d.name != exclude)
    runs = []
    for d in dirs[-window:]:
        rows = []
        try:
            for line in (d / "log.jsonl").read_text(encoding="utf-8").splitlines():
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    pass                                    # half-written or foreign line
        except OSError:
            continue
        try:
            summ = json.loads((d / "episode_summary.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            summ = None
        runs.append((d.name, rows, summ))
    return runs


def transitions(rows):
    """One (kind, amount, d_offset, offset_before, offset_after) per executed rotate/pan whose
    before and after offsets were both estimated with the target visible."""
    out = []
    for r in rows:
        try:
            if r.get("server_status") != "done":
                continue
            det = r.get("details") or {}
            if r.get("route_called") == "POST /rotate":
                kind, amt = "rot", det.get("measured_deg")      # what the gyro says was turned
                if amt is None:
                    amt = det.get("sent_deg")
            elif r.get("route_called") == "POST /camera/pan":
                kind, amt = "pan", det.get("sent_delta")
            else:
                continue
            cur, nxt = r.get("state_summary") or {}, r.get("next_state") or {}
            o1, o2 = cur.get("est_offset_px"), nxt.get("est_offset_px")
            if amt is None or o1 is None or o2 is None:
                continue
            if not (cur.get("target_visible") and nxt.get("target_visible")):
                continue
            out.append((kind, float(amt), float(o2) - float(o1), float(o1), float(o2)))
        except (TypeError, ValueError, AttributeError):
            continue
    return out


def fit_slope(samples, min_amt: float, min_n: int) -> Calib:
    """Least squares through the origin of d_offset on amount, with one round of outlier removal.
    Large moves weigh more, so noisy estimates after tiny moves do little harm."""
    pts = [(a, d) for a, d in samples if abs(a) >= min_amt]
    if len(pts) < 2:
        return Calib(n=len(pts))

    def slope(p):
        return sum(a * d for a, d in p) / sum(a * a for a, _ in p)

    k = slope(pts)
    if len(pts) >= 6:
        res = [d - k * a for a, d in pts]
        med = statistics.median(res)
        mad = 1.4826 * statistics.median(abs(x - med) for x in res)
        if mad > 0:
            keep = [p for p, x in zip(pts, res) if abs(x - med) <= 3 * mad]
            if len(keep) >= max(4, len(pts) // 2):
                pts, k = keep, slope(keep)
    n, saa = len(pts), sum(a * a for a, _ in pts)
    se = math.sqrt(sum((d - k * a) ** 2 for a, d in pts) / max(n - 1, 1) / saa)
    c = Calib(n=n, k=k, rel_err=(se / abs(k)) if k else None,
              agree=sum(1 for a, d in pts if d * k * a > 0) / n)
    if k and n >= min_n and c.agree >= 0.75:
        c.conf = clamp(1 - c.rel_err / 0.4, 0.0, 1.0) * min(1.0, n / (2.0 * min_n))
    return c


def learn(out_dir, args, exclude: Optional[str] = None) -> Learned:
    L, rot_s, pan_s, trans, ok_steps = Learned(), [], [], [], []
    for _, rows, summ in read_runs(out_dir, args.learn_window, exclude):
        tr = transitions(rows)
        trans += tr
        for kind, a, d, _, _ in tr:
            (rot_s if kind == "rot" else pan_s).append((a, d))
        if summ:
            L.episodes += 1
            if summ.get("status") == "success":
                L.successes += 1
                ok_steps.append(summ.get("steps") or 0)
            else:
                L.fails[str(summ.get("termination_reason") or "unknown").split(":")[0]] += 1
    L.med_steps_ok = statistics.median(ok_steps) if ok_steps else None
    L.rot = fit_slope(rot_s, args.min_rot_deg, args.min_samples)
    L.pan = fit_slope(pan_s, args.min_pan_units, args.min_samples)
    big = [(o1, o2) for _, _, _, o1, o2 in trans if abs(o1) > 2 * args.tol]
    L.big_moves = len(big)
    if big:
        # overshoot = landed on the other side of the marker by more than a quarter of the starting
        # error (and more than 2 x tolerance), so ordinary noise in the estimates does not count
        L.overshoot = sum(1 for o1, o2 in big
                          if o1 * o2 < 0 and abs(o2) > max(2 * args.tol, 0.25 * abs(o1))) / len(big)
        L.closed = statistics.median((abs(o1) - abs(o2)) / abs(o1) for o1, o2 in big)
        damp = 1 - min(1.0, L.overshoot / 0.3)              # overshooting pulls confidence back down
        L.rot.conf *= damp
        L.pan.conf *= damp
    return L


def render_notes(L: Learned, caps: dict, args, w: int, hfov: float) -> str:
    """The learned.md text: facts measured from past runs, in the units the LLM acts in."""
    ppd = w / hfov
    out = ["# Learned notes (measured by the backend from past episodes; rewritten after every run)", "",
           "## Track record"]
    if L.episodes:
        s = "- %d past episodes, %d succeeded" % (L.episodes, L.successes)
        if L.med_steps_ok is not None:
            s += " (median %g steps to succeed)" % L.med_steps_ok
        out.append(s)
        if L.fails:
            out.append("- Failures: " + ", ".join("%s x%d" % kv for kv in L.fails.most_common()))
    else:
        out.append("- No past episodes yet.")

    out += ["", "## Calibration (offset = target_x - marker_x in px; derived from your own earlier estimates)"]
    c = L.rot
    if c.k is not None:
        trust = "reliable" if c.conf >= 0.5 else ("usable" if c.conf > 0 else "not yet reliable, treat as a hint")
        out.append("- rotate_robot: +1 deg changes the offset by about %+.1f px (n=%d, error about %.0f%%, "
                   "confidence %.2f, %s). To cancel an offset of X px use amount = -X / %.1f."
                   % (c.k, c.n, 100 * (c.rel_err or 0), c.conf, trust, c.k))
    else:
        out.append("- rotate_robot: not measured yet. Field-of-view prior: about %+.1f px per degree "
                   "(expected sign: +1 deg raises the offset). Check the sign on your first rotate." % ppd)
    c = L.pan
    if c.k is not None:
        trust = "reliable" if c.conf >= 0.5 else ("usable" if c.conf > 0 else "not yet reliable, treat as a hint")
        out.append("- pan_camera: +1 unit changes the offset by about %+.2f px (n=%d, error about %.0f%%, "
                   "confidence %.2f, %s). To cancel an offset of X px use amount = -X / %.2f."
                   % (c.k, c.n, 100 * (c.rel_err or 0), c.conf, trust, c.k))
    elif args.pan_prior_deg_per_unit > 0:
        out.append("- pan_camera: not measured yet. UNVERIFIED guess: if 1000 units span about 240 deg (typical "
                   "for this kind of bus servo) one unit is about %.2f px in size; the DIRECTION is unknown. "
                   "Make your first pan a probe of about 40 units, see which way the target moved, then size "
                   "later pans from that." % (args.pan_prior_deg_per_unit * ppd))
    else:
        out.append("- pan_camera: scale and direction not measured yet. Make your first pan a probe of about "
                   "40 units and size later pans from what you see.")

    out += ["", "## Correction behaviour"]
    if L.big_moves >= 3 and L.closed is not None:
        verdict = "sizing looks about right."
        if L.overshoot >= 0.3:
            verdict = "you have been OVERSHOOTING: use a smaller share of the error until it settles."
        elif L.closed < 0.6 and L.overshoot < 0.15:
            verdict = "you have been UNDER-CORRECTING: apply the full computed correction, do not nudge."
        out.append("- %d earlier corrections started with the offset above 2 x tolerance: the median one closed "
                   "%.0f%% of the error and %.0f%% overshot past the marker by more than a quarter of the starting error; %s"
                   % (L.big_moves, 100 * L.closed, 100 * L.overshoot, verdict))
    else:
        out.append("- Not enough earlier corrections to judge sizing yet.")
    f_rot = 0.6 if (L.overshoot or 0) >= 0.3 else 0.7 + 0.3 * L.rot.conf
    f_pan = 0.6 if (L.overshoot or 0) >= 0.3 else 0.7 + 0.3 * L.pan.conf
    out.append("- Correct about %d%% of the remaining error per rotate step and about %d%% per pan step. "
               "If the amount you computed is inside the step limit, send it in one move instead of "
               "splitting it into several small ones." % (round(100 * f_rot), round(100 * f_pan)))

    out += ["", "## Step limits now allowed",
            "- rotate_robot up to %g deg per step (base %g, ceiling %g); pan_camera up to %d units per step "
            "(base %g, ceiling %g)." % (caps["rot"], caps["rot_base"], caps["rot_ceil"], caps["pan_step"],
                                        caps["pan_base"], caps["pan_ceil"]),
            "- The limits grow with calibration confidence and shrink if overshooting appears. The server's "
            "own limits always apply."]
    return "\n".join(out) + "\n"


def write_notes(path, text: str):
    try:
        Path(path).write_text(text, encoding="utf-8")
    except OSError as e:
        print("warning: could not write %s: %s" % (path, e))


# ------------------------------------------------------------------------- LLM
ACT_TOOL = {
    "type": "function",
    "name": "act",
    "description": "Report what you see and choose the next action.",
    "strict": True,
    "parameters": {
        "type": "object",
        "properties": {
            "observation": {
                "type": "string",
                "description": "One or two sentences: where the marker and the target are in the "
                               "current frame, what your last action did, and your plan."},
            "target_visible": {"type": "boolean"},
            "estimated_offset_px": {
                "type": ["number", "null"],
                "description": "Horizontal offset target_x minus marker_x in pixels "
                               "(positive = target is to the right of the marker). Null if unknown."},
            "action": {"type": "string", "enum": ACTIONS},
            "amount": {
                "type": ["number", "null"],
                "description": "rotate_robot: degrees (positive = turn left). pan_camera: change in "
                               "servo position units. move_*: distance in metres (positive). "
                               "Null for hold, stop and finish."},
        },
        "required": ["observation", "target_visible", "estimated_offset_px", "action", "amount"],
        "additionalProperties": False,
    },
}


def load_text(path, strip_frontmatter: bool = True) -> Optional[str]:
    p = Path(path)
    if not p.exists():
        return None
    t = p.read_text(encoding="utf-8")
    if strip_frontmatter and t.startswith("---"):
        parts = t.split("---", 2)
        if len(parts) == 3:
            t = parts[2]
    return t.strip()


def make_addendum(args, w: int, h: int, hfov: float, caps: dict) -> str:
    text = f"""TASK ADDENDUM (this overrides the playbook above wherever they differ)

PERCEPTION
- There is NO local marker/target detector. You are the only perception: look at the frame, find the marker and the target yourself, and estimate the offset. Ignore the playbook's detection confidences, normalized errors and transition storage: the backend does the bookkeeping, you only call act().
- A marker is mounted on the camera itself, so it stays at the SAME place in every frame and never moves in the image. The target is somewhere in front of the robot.
- GOAL: make the target's centre sit on the marker's centre, horizontally. There is no tilt control, so ignore any vertical difference.
- Images are {w}x{h} px, x increases to the right. offset = target_x - marker_x (positive: target is right of the marker).

HOW TO ACT (call act() every step)
- rotate_robot(amount): degrees, positive = turn left. Minimum {MIN_ROTATE_DEG:g}, at most {caps['rot']:g} per step, precision about 1 degree. Turning left normally shifts the scene right in the image; check against what you see.
- pan_camera(amount): change in servo position, range {caps['pan_min']}-{caps['pan_max']}, at most {caps['pan_step']:g} units per step. Its direction and pixels-per-unit scale are NOT known: work them out from the previous and current frames and your history. Good for fine corrections.
- move_forward / move_backward / move_left / move_right(amount): metres, positive, at most {caps['move']:g} per step. Open loop (time based), so it is imprecise and it changes the scene. Prefer rotate and pan; use a move only if they cannot help.
- hold: no motion for one step. stop: calls /stop and ends the episode as a failure (use only if something looks unsafe).
- finish: use only when the target centre is within about {args.tol:g} px of the marker centre. Success needs {args.k_stable} consecutive finish calls, each on a fresh frame (nothing moves in between). Any other action resets that count. Never finish when the target is not visible.

RULES
- Any server refusal, abort or busy reply ends the episode, so keep every request inside the limits.
- If target_visible is false for {args.n_lost_max} steps in a row, recovery mode starts: only pan_camera is allowed for the first {args.recovery_pan_steps} recovery steps, then only rotate_robot, and the episode fails after {args.recovery_max} recovery steps. It ends as soon as you see the target again.
- Learn what each action did from the previous frame and the history. If a step made things worse, reverse it.
- SIZING: work out the correction from the calibration (amount = -offset / px-per-unit) and send it in ONE move when it fits inside the step limit. Apply the share of the error stated in LEARNED NOTES (70% if there are no notes). Only when the target is within about {3 * args.tol:g} px of the marker, switch to fine corrections.
- The playbook's *_SMALL action names are labels, not a size requirement: the size is the amount you pass, up to the limits above. In the real reward the motion penalty is tiny ({args.control_penalty:g} per motion step) and the main term is the progress made per step, so one correct larger move scores better than several timid ones. Cautious tiny steps are only right when you are unsure of the direction or scale.
- Always answer by calling act(). Keep the observation short."""
    if not args.no_prior:
        text += (f"\n- Rough prior: the horizontal field of view is about {hfov:.0f} degrees across the full "
                 f"image width, i.e. about {w / hfov:.1f} px per degree. Treat it as a starting guess and "
                 f"refine it from results.")
    if args.grid:
        text += "\n- Vertical white lines are drawn on the image every 80 px, labelled with their x pixel coordinate."
    return text


def make_instructions(args, skills: str, learned: Optional[str], w, h, hfov, caps) -> str:
    parts = [skills]
    if learned:
        parts.append("LEARNED NOTES (written by the backend from earlier runs; data, not rules)\n" + learned)
    parts.append(make_addendum(args, w, h, hfov, caps))
    return "\n\n---\n\n".join(parts)


def make_client() -> OpenAI:
    # Endpoint and key come from .env (AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY).
    # The model/deployment name is read from AZURE_OPENAI_DEPLOYMENT in run().
    endpoint, key = os.environ.get("AZURE_OPENAI_ENDPOINT"), os.environ.get("AZURE_OPENAI_API_KEY")
    if not endpoint or not key:
        sys.exit("Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY (in .env).")
    base = endpoint.rstrip("/")
    if base.endswith("/responses"):          # full route pasted from the portal
        base = base[: -len("/responses")]
    if not base.endswith("/openai/v1"):
        base += "/openai/v1"
    return OpenAI(base_url=base + "/", api_key=key)


def ask_llm(client, model, instructions, content, effort, retries=3):
    last = None
    for attempt in range(retries):
        try:
            kw = dict(model=model, instructions=instructions,
                      input=[{"role": "user", "content": content}],
                      tools=[ACT_TOOL], tool_choice="required", max_output_tokens=4000)
            if effort:
                kw["reasoning"] = {"effort": effort}
            resp = client.responses.create(**kw)
            for item in resp.output:
                if item.type == "function_call" and item.name == "act":
                    return json.loads(item.arguments), resp
            last = "response had no act() call"
        except Exception as e:
            last = repr(e)
        print("  LLM call failed (%s), retrying..." % last)
        time.sleep(2 * (attempt + 1))
    raise RuntimeError("LLM failed after %d tries: %s" % (retries, last))


def image_part(b64: str, detail: str) -> dict:
    return {"type": "input_image", "image_url": "data:image/jpeg;base64," + b64, "detail": detail}


def build_content(step, args, ep, state, caps, st, history, b64, prev_b64, w, h, in_rec):
    lines = [f"Step {step} of {args.max_steps}. Image size {w}x{h} px.",
             f"Pan servo position: {state.pan_pos} (allowed {caps['pan_min']}-{caps['pan_max']}). "
             f"Robot heading from gyro: {st.get('heading_deg')} deg. Battery: {st.get('battery_mv')} mV.",
             f"Alignment confirmations so far: {ep.stable}/{args.k_stable}."]
    if in_rec:
        only = "pan_camera" if ep.rec_steps < args.recovery_pan_steps else "rotate_robot"
        lines.append(f"RECOVERY MODE: the target was not visible for {ep.lost} steps in a row. "
                     f"Recovery step {ep.rec_steps + 1} of {args.recovery_max}. Only {only} is allowed "
                     f"until you see the target again.")
    lines.append("Recent history:\n" + ("\n".join(history) if history else "(none yet - this is the first step)"))
    content = [{"type": "input_text", "text": "\n".join(lines)}]
    if prev_b64:
        content += [{"type": "input_text", "text": "PREVIOUS frame (before your last action):"},
                    image_part(prev_b64, args.detail)]
    content += [{"type": "input_text", "text": "CURRENT frame:"}, image_part(b64, args.detail),
                {"type": "input_text", "text": "Call act() with your next action."}]
    return content


# ------------------------------------------------------------------- execution
@dataclass
class State:
    pan_pos: int = 500


@dataclass
class Episode:
    id: str
    w: int = 640
    steps: int = 0
    total_reward: float = 0.0
    stable: int = 0            # consecutive confirmed-aligned steps
    lost: int = 0              # consecutive target_visible=false reports
    rec_steps: int = 0         # recovery steps used
    local_faults: int = 0      # consecutive invalid (locally refused) actions
    last_err: Optional[float] = None
    pending: Optional[dict] = None
    t0: float = field(default_factory=time.time)


def local_refusal(why: str) -> dict:
    return {"route": None, "status": "local_refused", "reason": why, "fault": "local"}


def execute(robot: Robot, action: str, amount, state: State, caps: dict, args) -> dict:
    """Clip the LLM's request to the limits and run it on the exact server route."""
    if action == "hold":
        return {"route": None, "status": "done", "reason": "hold", "fault": None}
    if action == "stop":
        r = robot.stop()
        return {"route": "POST /stop", "status": r.get("status"), "reason": "agent requested stop",
                "fault": None, "terminal": True}
    if amount is None:
        return local_refusal("amount was null")
    if action == "rotate_robot":
        deg = clamp(float(amount), -caps["rot"], caps["rot"])
        if abs(deg) < MIN_ROTATE_DEG:
            return local_refusal("rotation under %.1f deg is not possible; use pan_camera" % MIN_ROTATE_DEG)
        r = robot.rotate(deg)
        out = {"route": "POST /rotate", "sent_deg": round(deg, 2), "measured_deg": r.get("measured_deg")}
    elif action == "pan_camera":
        delta = clamp(float(amount), -caps["pan_step"], caps["pan_step"])
        old = state.pan_pos
        new = int(clamp(old + delta, caps["pan_min"], caps["pan_max"]))
        if new == old:
            return local_refusal("pan is at its limit or the change rounds to zero")
        r = robot.pan(new, caps["pan_dur"])
        if r.get("status") == "done":
            state.pan_pos = new
            time.sleep(caps["pan_dur"] + args.settle)   # pan is fire-and-forget: wait for it
        out = {"route": "POST /camera/pan", "pan_pos_now": state.pan_pos, "sent_delta": new - old}
    elif action in MOVE_ACTIONS:
        dist = min(float(amount), caps["move"])
        if dist < 0.01:
            return local_refusal("move distance must be at least 0.01 m")
        r = robot.move(MOVE_ACTIONS[action], dist, caps["speed"])
        out = {"route": "POST /move", "sent_m": round(dist, 3),
               "heading_change_deg": r.get("heading_change_deg")}
    else:
        return local_refusal("unknown action %r" % action)
    out.update(status=r.get("status"), reason=r.get("reason"),
               fault=None if r.get("status") == "done" else "server")
    return out


def history_line(step, d, result) -> str:
    obs = (d.get("observation") or "")[:160]
    s = (f"step {step}: saw '{obs}' (est. offset {d.get('estimated_offset_px')} px) -> "
         f"{d.get('action')} {d.get('amount')} => {result.get('status')}")
    if result.get("reason") and result.get("reason") != "hold":
        s += f" ({result['reason']})"
    if result.get("measured_deg") is not None:
        s += f", measured {result['measured_deg']} deg"
    if result.get("heading_change_deg") is not None:
        s += f", heading change {result['heading_change_deg']} deg"
    if "pan_pos_now" in result:
        s += f", pan now {result['pan_pos_now']}"
    return s


# ---------------------------------------------------------------------- reward
def finalize(ep: Episode, log, args, next_state: Optional[dict], done=False, done_reason="", success=False):
    """Write the pending step record once its reward is known (needs the next observation)."""
    p = ep.pending
    if p is None:
        return
    cur, parts = p["record"]["state_summary"], {}
    if p["moved"]:
        parts["control"] = -args.control_penalty
    if p["fault"] == "server":
        parts["safety"] = -10.0
    elif p["fault"] == "local":
        parts["safety"] = -1.0
    if next_state is not None:
        a, b = cur.get("abs_err_px"), next_state.get("abs_err_px")
        if a is not None and b is not None:
            parts["progress"] = round((a - b) / ep.w * 100.0, 3)    # % of image width gained
        o1, o2 = cur.get("est_offset_px"), next_state.get("est_offset_px")
        if o1 is not None and o2 is not None and o1 * o2 < 0 and min(abs(o1), abs(o2)) > args.tol:
            parts["oscillation"] = -args.oscillation_penalty
    if success:
        parts["success"] = 10.0
    r = round(sum(parts.values()), 3)
    ep.total_reward += r
    log.write(json.dumps(dict(p["record"], reward=r, reward_parts=parts, done=done,
                              done_reason=done_reason, next_state=next_state)) + "\n")
    ep.pending = None


# ------------------------------------------------------------------------ main
def run(args):
    skills = load_text(args.skills)
    if skills is None:
        sys.exit("skills file not found: %s (pass --skills PATH)" % args.skills)
    learned = None                       # learned notes text, built below from earlier runs

    robot = Robot(os.environ.get("ROBOT_URL", "http://localhost:8081"), os.environ.get("ROBOT_TOKEN", ""))
    client = make_client()
    model = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-5.3-codex")

    episode_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.out) / episode_id
    run_dir.mkdir(parents=True, exist_ok=True)
    log = open(run_dir / "log.jsonl", "a", buffering=1)

    # 1. readiness: /status, /camera_info, gyro, centre the pan servo
    st = robot.status()
    why = safety_reason(st)
    if why:
        sys.exit("Not safe to start: " + why)
    lim = st.get("limits") or {}
    try:
        ci = robot.camera_info()
        hfov = float(ci.get("horizontal_fov_deg", DEFAULT_HFOV_DEG))
        w0, h0 = int(ci.get("width", 640)), int(ci.get("height", 480))
    except Exception as e:
        print("warning: /camera_info failed (%s), using defaults" % e)
        hfov, w0, h0 = DEFAULT_HFOV_DEG, 640, 480
    # width of the frames the model will actually see (the server only shrinks, never enlarges)
    w_eff = args.width if args.width and 32 <= args.width < w0 else w0
    L = None if args.no_learn else learn(args.out, args, exclude=episode_id)   # learn from earlier runs
    caps = make_caps(lim, args, L)
    if L is not None:
        learned = L.text = render_notes(L, caps, args, w_eff, hfov)
        write_notes(args.learned, learned)
        print("learning: %d past episodes | rotate k=%s conf %.2f | pan k=%s conf %.2f | caps: rotate %g deg, pan %d units"
              % (L.episodes, "n/a" if L.rot.k is None else "%+.1f" % L.rot.k, L.rot.conf,
                 "n/a" if L.pan.k is None else "%+.2f" % L.pan.k, L.pan.conf, caps["rot"], caps["pan_step"]))
    else:
        print("learning off (--no-learn): base step caps, no learned notes")
    print("battery_mv:", st.get("battery_mv"), "| gyro_ready:", st.get("gyro_ready"),
          "| image %dx%d, hfov %.1f deg" % (w0, h0, hfov))
    if not st.get("gyro_ready") and not wait_gyro(robot):
        print("warning: gyro not ready, rotate will be refused (pan still works)")

    state, ep = State(pan_pos=caps["pan_center"]), Episode(id=episode_id, w=w0)
    robot.pan(caps["pan_center"], min(1.0, caps["pan_dur"] * 3 + 0.7))
    time.sleep(1.3)
    print("episode:", episode_id, "| dir:", run_dir)

    history: deque = deque(maxlen=args.history)
    prev_b64, termination = None, None
    try:
        for step in range(1, args.max_steps + 1):
            ep.steps = step

            # 2. observe: /status safety check, then the frame
            st = robot.status()
            why = safety_reason(st)
            if why:
                termination = ("failure", "safety: " + why, False)
                break
            jpeg, meta = grab_frame(robot, args.width)
            shown = add_grid(jpeg) if args.grid else jpeg
            frame_path = run_dir / f"step_{step:03d}.jpg"
            frame_path.write_bytes(shown)
            b64 = base64.b64encode(shown).decode("ascii")
            w, h = int(meta.get("width", w0)), int(meta.get("height", h0))
            ep.w = w

            # 3-4. ask the LLM
            in_rec = ep.lost >= args.n_lost_max
            content = build_content(step, args, ep, state, caps, st, history, b64,
                                    None if args.no_prev else prev_b64, w, h, in_rec)
            t0 = time.time()
            d, resp = ask_llm(client, model, make_instructions(args, skills, learned, w, h, hfov, caps),
                              content, args.effort)
            llm_s = time.time() - t0
            action, amount = d["action"], d["amount"]
            off, vis = d.get("estimated_offset_px"), bool(d.get("target_visible"))
            err = abs(off) if (vis and off is not None) else None
            ep.last_err = err if err is not None else ep.last_err
            print(f"[{step}] {llm_s:.1f}s  {action} {amount}  offset~{off}  vis={vis}  | {d['observation'][:90]}")

            now = dict(est_offset_px=off, abs_err_px=err, target_visible=vis,
                       heading_deg=st.get("heading_deg"), battery_mv=st.get("battery_mv"),
                       imu_age_s=(st.get("age_s") or {}).get("imu"), pan_pos=state.pan_pos,
                       stable=ep.stable, lost=ep.lost)
            finalize(ep, log, args, now)                       # reward for the previous action

            # 5-6. gate and execute the action
            term, result = None, None
            if in_rec and not vis and ep.rec_steps >= args.recovery_max:
                result = {"route": None, "status": "not_executed", "reason": "recovery exhausted", "fault": None}
                term = ("failure", "recovery exhausted: target not found", False)
            elif action == "finish":
                if not vis or off is None or abs(off) > args.tol:
                    ep.stable = 0
                    result = local_refusal("finish needs the target visible and within %g px" % args.tol)
                else:
                    ep.stable += 1
                    result = {"route": None, "status": "done", "fault": None,
                              "reason": "confirming alignment %d/%d" % (ep.stable, args.k_stable)}
                    if ep.stable >= args.k_stable:
                        term = ("success", "aligned for %d consecutive steps" % args.k_stable, True)
            else:
                ep.stable = 0
                gate = None
                if in_rec and not vis:
                    only = "pan_camera" if ep.rec_steps < args.recovery_pan_steps else "rotate_robot"
                    ep.rec_steps += 1
                    if action not in (only, "stop"):
                        gate = "recovery mode: only %s is allowed now" % only
                result = local_refusal(gate) if gate else execute(robot, action, amount, state, caps, args)
                if result.get("terminal"):
                    term = ("failure", "agent requested stop", False)
                elif result.get("fault") == "server":
                    term = ("failure", "server %s: %s" % (result.get("status"), result.get("reason")), False)
            print("      ->", {k: v for k, v in result.items() if k != "fault"})

            # bookkeeping: lost counter, invalid-action counter, pending log record
            ep.lost, ep.rec_steps = (0, 0) if vis else (ep.lost + 1, ep.rec_steps)
            if result.get("fault") == "local":
                ep.local_faults += 1
                if ep.local_faults >= args.max_local_faults and term is None:
                    term = ("failure", "%d invalid actions in a row" % ep.local_faults, False)
            elif result.get("status") == "done":
                ep.local_faults = 0
            usage = getattr(resp, "usage", None)
            ep.pending = dict(
                moved=result.get("route") in MOTION_ROUTES, fault=result.get("fault"),
                record=dict(
                    episode_id=episode_id, step_idx=step, timestamp=time.time(), frame=frame_path.name,
                    state_summary=now, observation=d["observation"], action=action, amount=amount,
                    route_called=result.get("route"), server_status=result.get("status"),
                    server_reason=result.get("reason"),
                    details={k: v for k, v in result.items()
                             if k not in ("route", "status", "reason", "fault", "terminal")},
                    llm_seconds=round(llm_s, 2),
                    tokens={"in": getattr(usage, "input_tokens", None), "out": getattr(usage, "output_tokens", None)}))
            history.append(history_line(step, d, result))
            prev_b64 = b64
            if term:
                termination = term
                break
        else:
            termination = ("failure", "step limit reached", False)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        termination = ("interrupted", "keyboard interrupt", False)
    except Exception as e:
        traceback.print_exc()
        termination = ("failure", "error: %r" % (e,), False)
    finally:
        safe_stop(robot)                                       # always end with POST /stop
        status, reason, success = termination or ("failure", "unknown", False)
        finalize(ep, log, args, None, done=True, done_reason=reason, success=success)
        log.close()
        summary = dict(episode_id=episode_id, mode="llm_pure_vision", status=status,
                       total_reward=round(ep.total_reward, 3), steps=ep.steps,
                       duration_s=round(time.time() - ep.t0, 1), final_alignment_error_px=ep.last_err,
                       error_source="llm_estimate", termination_reason=reason, model=model,
                       policy=dict(learning=not args.no_learn, rotate_cap_deg=caps["rot"],
                                   pan_cap_units=caps["pan_step"],
                                   conf_rot=round(L.rot.conf, 3) if L else None,
                                   conf_pan=round(L.pan.conf, 3) if L else None,
                                   notes_sha=hashlib.sha1((learned or "").encode()).hexdigest()[:10]))
        (run_dir / "episode_summary.json").write_text(json.dumps(summary, indent=2))
        print("\nEpisode %s: %s (%s) | reward %.2f | %d steps" % (episode_id, status, reason, ep.total_reward, ep.steps))
        print("Log and frames saved in", run_dir)
        if not args.no_learn:                                   # fold this episode into the notes
            try:
                L2 = learn(args.out, args)
                caps2 = make_caps(lim, args, L2)
                write_notes(args.learned, render_notes(L2, caps2, args, w_eff, hfov))
                print("learned notes updated (%s): next run's caps rotate %g deg, pan %d units"
                      % (args.learned, caps2["rot"], caps2["pan_step"]))
            except Exception as e:
                print("warning: could not update learned notes:", e)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="LLM aiming loop (pure-vision perception, skills.md playbook).")
    ap.add_argument("--skills", default=str(Path(__file__).with_name("skills.md")), help="playbook file")
    ap.add_argument("--learned", default="learned.md", help="learned notes file, rewritten after every run")
    ap.add_argument("--no-learn", action="store_true", help="baseline: ignore past runs, no learned notes")
    ap.add_argument("--learn-window", type=int, default=20, help="newest episodes in --out to learn from")
    ap.add_argument("--min-samples", type=int, default=5, help="calibration samples needed before it is trusted")
    ap.add_argument("--min-rot-deg", type=float, default=2.0, help="smallest rotation used for calibration")
    ap.add_argument("--min-pan-units", type=float, default=15.0, help="smallest pan used for calibration")
    ap.add_argument("--pan-prior-deg-per-unit", type=float, default=0.24,
                    help="unverified guess of degrees per pan unit for the first runs (0 = give no guess)")
    ap.add_argument("--max-steps", type=int, default=25)
    ap.add_argument("--tol", type=float, default=10.0, help="px tolerance for alignment")
    ap.add_argument("--k-stable", type=int, default=3, help="consecutive confirmations needed for success")
    ap.add_argument("--max-rotate", type=float, default=10.0, help="base deg per rotate step before calibration")
    ap.add_argument("--max-rotate-ceiling", type=float, default=25.0,
                    help="deg per rotate step at full calibration confidence (server allows 90)")
    ap.add_argument("--max-pan", type=float, default=60.0, help="base servo units per pan step")
    ap.add_argument("--max-pan-ceiling", type=float, default=250.0,
                    help="units per pan step at full calibration confidence")
    ap.add_argument("--max-move", type=float, default=0.10, help="metres per move step (server allows 0.30)")
    ap.add_argument("--pan-duration", type=float, default=0.3)
    ap.add_argument("--settle", type=float, default=0.3, help="extra wait after a pan, seconds")
    ap.add_argument("--n-lost-max", type=int, default=3, help="steps without the target before recovery")
    ap.add_argument("--recovery-pan-steps", type=int, default=4, help="recovery steps restricted to pan")
    ap.add_argument("--recovery-max", type=int, default=8, help="recovery steps before the episode fails")
    ap.add_argument("--max-local-faults", type=int, default=4, help="invalid actions in a row before failing")
    ap.add_argument("--control-penalty", type=float, default=0.05)
    ap.add_argument("--oscillation-penalty", type=float, default=0.2)
    ap.add_argument("--width", type=int, default=None, help="downscale frames to this width")
    ap.add_argument("--history", type=int, default=6, help="past steps kept as text")
    ap.add_argument("--effort", default="low", help="reasoning effort: low|medium|high ('' to omit)")
    ap.add_argument("--detail", default="high", help="image detail: low|high|auto")
    ap.add_argument("--grid", action="store_true", help="overlay labelled vertical gridlines")
    ap.add_argument("--no-prev", action="store_true", help="do not send the previous frame")
    ap.add_argument("--no-prior", action="store_true", help="do not tell the LLM the field of view")
    ap.add_argument("--out", default="runs")
    return ap


if __name__ == "__main__":
    run(build_parser().parse_args())