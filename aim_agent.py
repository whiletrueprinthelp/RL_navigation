#!/usr/bin/env python3
"""aim_agent.py - Step 4: LLM-in-the-loop aiming, pure vision (gpt-5.3-codex on Azure).

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

    python3 aim_agent.py

Each step: grab frame -> send frame(s) + short history to the LLM -> LLM calls act() ->
clamp and execute on the robot -> log. The marker is fixed on the camera, so only the
target moves in the image. Every step is logged to runs/<timestamp>/log.jsonl together
with the frame the LLM saw, so you can score runs afterwards.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()   # loads ROBOT_URL, ROBOT_TOKEN and the AZURE_OPENAI_* settings from .env (current directory)

PAN_MIN, PAN_MAX, PAN_CENTER = 0, 1000, 500
HFOV_DEG = 60.8          # from camera_info: 2*atan(320/546.1)


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

    def image(self, width: Optional[int] = None):
        return self._call("GET", "/image", params={"width": width} if width else None)

    def rotate(self, deg: float):
        return self._call("POST", "/rotate", json={"angle_deg": deg})

    def pan(self, position: int, duration_s: float):
        return self._call("POST", "/camera/pan",
                          json={"position": int(position), "duration_s": duration_s})

    def stop(self):
        try:
            self._call("POST", "/stop")
        except Exception as e:  # never let a failed stop hide the original error
            print("warning: /stop failed:", e)


def grab_frame(robot: Robot, width: Optional[int], max_age: float = 0.6, tries: int = 5):
    """Return (jpeg_bytes, meta). Retries a few times to avoid a stale frame."""
    meta = None
    for i in range(tries):
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


def score_frame(jpeg: bytes) -> Optional[dict]:
    """Hook for the ground-truth CV scorer (target pixel minus the fixed marker pixel).
    Not needed for the pure-vision run; return a dict later and it is logged each step."""
    return None


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
            "action": {"type": "string", "enum": ["rotate_robot", "pan_camera", "finish"]},
            "amount": {
                "type": ["number", "null"],
                "description": "rotate_robot: degrees (positive = robot turns left). "
                               "pan_camera: change in servo position units. Null for finish."},
        },
        "required": ["observation", "target_visible", "estimated_offset_px", "action", "amount"],
        "additionalProperties": False,
    },
}


def make_instructions(args, w: int, h: int) -> str:
    px_per_deg = w / HFOV_DEG
    text = f"""You control a small wheeled robot with a camera on a servo pan mount. Your job is to aim it.

SETUP
- A marker is mounted on the camera itself, so it stays at the SAME place in every frame. It never moves in the image.
- A target is somewhere in front of the robot. Your goal is to make the target's centre sit on the marker's centre, horizontally.
- target is specifically the black label center on the front bin.
- Images are {w}x{h} px. x increases to the right. Offset = target_x - marker_x (positive: target is right of the marker).
- There is no tilt control, so ignore any vertical difference. Only horizontal alignment matters.

ACTIONS (call act() every step)
- rotate_robot(amount): turn the whole robot by `amount` degrees. Positive = turn left (ROS convention), which normally shifts the scene to the right in the image; check this against what you see. Minimum 0.5 deg, at most {args.max_rotate:g} deg per step, precision about 1 deg.
- pan_camera(amount): move the camera servo by `amount` units (position range {PAN_MIN}-{PAN_MAX}, at most {args.max_pan:g} units per step). The direction and the pixels-per-unit scale are NOT known: work them out by comparing the previous and current frames and your history. Pan is good for fine corrections.
- finish: use only when the target centre is within about {args.tol:g} px of the marker centre. If unsure, take one more small step instead.

METHOD
- Use the previous frame and the history to learn what each action did. If a step made things worse, reverse it.
- Correct about 70-80% of the estimated error per step and use smaller steps as you get close, to avoid overshoot.
- If the target is not visible, set target_visible=false and search with moderate pans or rotations.
- Always answer by calling act(). Keep the observation short."""
    if not args.no_prior:
        text += (f"\n- Rough prior: the horizontal field of view is about {HFOV_DEG:.0f} degrees across the "
                 f"full image width, i.e. about {px_per_deg:.1f} px per degree. Treat this as a starting "
                 f"guess and refine it from results.")
    if args.grid:
        text += "\n- Vertical white lines are drawn on the image every 80 px, labelled with their x pixel coordinate."
    return text


def make_client() -> OpenAI:
    # Endpoint and key come from .env (AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY).
    # The model/deployment name is read from AZURE_OPENAI_DEPLOYMENT in run().
    endpoint, key = os.environ.get("AZURE_OPENAI_ENDPOINT"), os.environ.get("AZURE_OPENAI_API_KEY")
    if not endpoint or not key:
        sys.exit("Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY.")
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


def build_content(step, args, state, history, b64, prev_b64, w, h):
    head = (f"Step {step} of {args.max_steps}. Image size {w}x{h} px. "
            f"Camera pan servo position: {state.pan_pos} (allowed {PAN_MIN}-{PAN_MAX}).\n"
            "Recent history:\n" + ("\n".join(history) if history else "(none yet - this is the first step)"))
    content = [{"type": "input_text", "text": head}]
    if prev_b64:
        content += [{"type": "input_text", "text": "PREVIOUS frame (before your last action):"},
                    image_part(prev_b64, args.detail)]
    content += [{"type": "input_text", "text": "CURRENT frame:"}, image_part(b64, args.detail),
                {"type": "input_text", "text": "Call act() with your next action."}]
    return content


# ------------------------------------------------------------------- execution
@dataclass
class State:
    pan_pos: int = PAN_CENTER


def execute(robot: Robot, action: str, amount, state: State, args) -> dict:
    """Clamp the LLM's request and run it. Always returns {status, reason, ...}."""
    if amount is None:
        return {"status": "refused", "reason": "amount was null"}
    if action == "rotate_robot":
        deg = clamp(float(amount), -args.max_rotate, args.max_rotate)
        if abs(deg) < 0.5:
            return {"status": "refused", "reason": "rotation under 0.5 deg is not possible; use pan_camera"}
        r = robot.rotate(deg)
        return {"status": r.get("status"), "reason": r.get("reason"), "sent_deg": round(deg, 2),
                "measured_deg": r.get("measured_deg")}
    if action == "pan_camera":
        delta = clamp(float(amount), -args.max_pan, args.max_pan)
        new = int(clamp(state.pan_pos + delta, PAN_MIN, PAN_MAX))
        if new == state.pan_pos:
            return {"status": "refused", "reason": "pan is at its limit or the change rounds to zero"}
        old = state.pan_pos
        r = robot.pan(new, args.pan_duration)
        if r.get("status") == "done":
            state.pan_pos = new
            time.sleep(args.pan_duration + args.settle)   # pan is fire-and-forget: wait for it
        return {"status": r.get("status"), "reason": r.get("reason"), "pan_pos_now": state.pan_pos,
                "sent_delta": new - old}
    return {"status": "refused", "reason": "unknown action %r" % action}


def history_line(step, d, result) -> str:
    obs = (d.get("observation") or "")[:160]
    s = f"step {step}: saw '{obs}' (est. offset {d.get('estimated_offset_px')} px) -> {d.get('action')} {d.get('amount')} => {result.get('status')}"
    if result.get("reason"):
        s += f" ({result['reason']})"
    if result.get("measured_deg") is not None:
        s += f", measured {result['measured_deg']} deg"
    if "pan_pos_now" in result:
        s += f", pan now {result['pan_pos_now']}"
    return s


def wait_gyro(robot: Robot, timeout: float = 20.0) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if robot.status().get("gyro_ready"):
            return True
        time.sleep(1.0)
    return False


# ------------------------------------------------------------------------ main
def run(args):
    robot = Robot(os.environ.get("ROBOT_URL", "http://localhost:8081"), os.environ.get("ROBOT_TOKEN", ""))
    client = make_client()
    model = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-5.3-codex")
    run_dir = Path(args.out) / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    log = open(run_dir / "log.jsonl", "a", buffering=1)

    st = robot.status()
    print("battery_mv:", st.get("battery_mv"), "| gyro_ready:", st.get("gyro_ready"))
    if not st.get("gyro_ready") and not wait_gyro(robot):
        print("warning: gyro not ready, rotate will be refused (pan still works)")

    state = State()
    robot.pan(PAN_CENTER, 1.0)
    time.sleep(1.3)
    print("run dir:", run_dir)

    history: deque = deque(maxlen=args.history)
    prev_b64, refusals = None, 0
    try:
        for step in range(1, args.max_steps + 1):
            jpeg, meta = grab_frame(robot, args.width)
            shown = add_grid(jpeg) if args.grid else jpeg
            frame_path = run_dir / f"step_{step:03d}.jpg"
            frame_path.write_bytes(shown)
            b64 = base64.b64encode(shown).decode("ascii")
            w, h = meta.get("width", 640), meta.get("height", 480)

            content = build_content(step, args, state, history, b64,
                                    None if args.no_prev else prev_b64, w, h)
            t0 = time.time()
            d, resp = ask_llm(client, model, make_instructions(args, w, h), content, args.effort)
            llm_s = time.time() - t0
            print(f"[{step}] {llm_s:.1f}s  {d['action']} {d['amount']}  offset~{d['estimated_offset_px']}  | {d['observation'][:100]}")

            if d["action"] == "finish":
                result = {"status": "done", "reason": "agent declared aligned"}
            else:
                result = execute(robot, d["action"], d["amount"], state, args)
                print("      ->", result)

            usage = getattr(resp, "usage", None)
            log.write(json.dumps({
                "step": step, "t": time.time(), "frame": frame_path.name, "decision": d,
                "result": result, "pan_pos": state.pan_pos, "llm_seconds": round(llm_s, 2),
                "tokens": {"in": getattr(usage, "input_tokens", None), "out": getattr(usage, "output_tokens", None)},
                "cv_score": score_frame(jpeg),
            }) + "\n")

            if d["action"] == "finish":
                print("Agent declared the target aligned.")
                break
            history.append(history_line(step, d, result))
            refusals = refusals + 1 if result.get("status") != "done" else 0
            if refusals >= args.max_refusals:
                print("Stopping: %d non-successful actions in a row." % refusals)
                break
            prev_b64 = b64
        else:
            print("Reached the step limit without finishing.")
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        robot.stop()
        log.close()
    print("Log and frames saved in", run_dir)


def main():
    ap = argparse.ArgumentParser(description="LLM aiming loop (pure vision).")
    ap.add_argument("--max-steps", type=int, default=25)
    ap.add_argument("--tol", type=float, default=10.0, help="px tolerance quoted to the LLM")
    ap.add_argument("--max-rotate", type=float, default=10.0, help="deg per rotate step (server allows 90)")
    ap.add_argument("--max-pan", type=float, default=60.0, help="servo units per pan step")
    ap.add_argument("--pan-duration", type=float, default=0.3)
    ap.add_argument("--settle", type=float, default=0.3, help="extra wait after a pan, seconds")
    ap.add_argument("--width", type=int, default=None, help="downscale frames to this width")
    ap.add_argument("--history", type=int, default=6, help="past steps kept as text")
    ap.add_argument("--effort", default="low", help="reasoning effort: low|medium|high ('' to omit)")
    ap.add_argument("--detail", default="high", help="image detail: low|high|auto")
    ap.add_argument("--grid", action="store_true", help="overlay labelled vertical gridlines")
    ap.add_argument("--no-prev", action="store_true", help="do not send the previous frame")
    ap.add_argument("--no-prior", action="store_true", help="do not tell the LLM the field of view")
    ap.add_argument("--max-refusals", type=int, default=4)
    ap.add_argument("--out", default="runs")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
