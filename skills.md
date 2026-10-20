---
name: robot-vision-alignment-rl
description: Align a marker fixed to the camera with a target, using only camera frames and rotate / pan / move actions. Supports shoot mode (rounds, first-hit) and track mode (continuous follow).
tools:
  - POST /move
  - POST /rotate
  - POST /stop
  - POST /camera/pan
  - GET /status
  - GET /image
  - GET /camera_info
version: 0.5.0
---

# Robot Vision Alignment RL Playbook (Server-Aligned)

## Objective
Make the pointer's tip (its lowest point) align with any part of the target. There is no tilt control, so ignore any vertical difference.

The backend can run in two modes (see RUN PARAMETERS in the prompt):
- `shoot` mode: round-based, ends at the first aligned `finish`.
- `track` mode: continuous tracking for a time/step budget, no rounds/check frame.

## Perception (you are the only detector)
- Look at each frame yourself and find the marker and the target. No local detector exists.
- The marker is mounted on the camera, so it stays at the same place in every frame. The target is the object named in the run parameters.
- x increases to the right. offset = target_x - marker_x (px), where marker_x is the x of the pointer's tip and target_x is the target's centre line. Positive: the target is right of the tip.
- Always report `target_visible` honestly. Say false if you are unsure it is the right object; a wrong sighting wastes steps.

## Actions (call act() every step)
- `rotate_robot(amount)`: degrees, positive = turn left. Turning left normally shifts the scene right in the image; check against what you see. Coarse corrections.
- `pan_camera(amount)`: change in servo position. Its direction and pixels-per-unit scale are not known: work them out from the previous and current frames and your history. Good for fine corrections.
- `move_forward / move_backward / move_left / move_right(amount)`: metres. Open loop and imprecise, and it changes the scene. Use only if rotate and pan cannot help.
- `hold`: no motion this step. `stop`: ends the episode as a failure; use only if something looks unsafe.
- `finish`: call it only when the target's centre line is within tolerance of the pointer's tip. In `shoot` mode it ends the run as a success. In `track` mode it means "aligned now, hold" (it does not normally end the run unless the run parameters say so).
- Step limits and the tolerance are in the run parameters. A request over a limit is refused, and any refusal ends the episode.

## Rounds (shoot mode only; the target moves)
- A round starts when the target has been moved. Earlier offsets, the pan position you had worked out and the previous frame no longer describe the new scene; start from the new frame.
- In a round you get a limited number of movement chances (the run parameters say how many, normally 2). Each correction of yours uses one chance, and so does each search turn the backend makes. After the last chance one more frame is shown as a CHECK FRAME: if the target is visible and you are within tolerance, call `finish`; otherwise the round is over, the robot stops, the target is moved and a new round starts. Motion requests on a check frame are ignored.
- If you are already aligned on the first frame of a round, call `finish` straight away. Do not move when you are within tolerance.
- Because you only have a few chances, make each one count: use the full computed correction (see Stage 2), not a cautious nudge. If the target is out of view, the search turns have already used some chances, so there may be only one or none left for correcting.

## Stage 1: SEARCH (target not visible)
- The backend takes over: it turns the robot in fixed steps and shows you a frame after each. In `shoot` mode each of these turns uses one round chance; in `track` mode there are no round chances.
- While the message says SEARCH MODE, only report `target_visible` and your offset estimate. Your motion requests are ignored.
- The first frame with the target holds still. If it is seen again on the next frame, the stage ends and your chosen action runs on that frame. At the start of a round, a target that is already in view needs no search.
- If the whole circle is scanned without a sighting, the run ends as "target not found".
- If the target is lost for several steps in a row during alignment, the search starts again.

## Stage 2: ALIGN (target visible)
1. Estimate the offset from the frame.
2. Work out the correction from the calibration: amount = -offset / px-per-unit. Send it in ONE move when it fits inside the step limit.
3. Apply the share of the error given in LEARNED NOTES (70% if there are no notes). When the target is within about 3x the tolerance, switch to fine corrections (pan).
4. Next frame: compare with the previous one if one is shown. Learn what the action actually did. If a step made things worse, reverse it.
5. If you are in `shoot` mode and within tolerance now, call `finish` immediately.
6. If you are in `track` mode and within tolerance, prefer `hold`; do not chase tiny estimate noise. If drift is consistently one way across frames, apply a small lead correction.

## How you are scored
- Progress made per step is the main term; the cost of a motion is tiny. One correct larger move beats several timid ones.
- Tiny steps are right only when you are unsure of the direction or the scale.
- In `shoot` mode, a hit (`finish` while aligned) is the success bonus.
- In `track` mode, quality is sustained alignment over many frames (aligned share/streak), with penalties for unnecessary motion while already aligned.
- Safety faults are heavily penalised in both modes.

## Always
- Answer by calling act(). Keep the observation short.
- Treat LEARNED NOTES as data from earlier runs, not rules. If the current frames contradict them, trust the frames.