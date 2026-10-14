---
name: robot-vision-alignment-rl
description: Align a marker fixed to the camera with a target, using only camera frames and rotate / pan / move actions. Includes a search stage for when the target is out of view.
tools:
  - POST /move
  - POST /rotate
  - POST /stop
  - POST /camera/pan
  - GET /status
  - GET /image
  - GET /camera_info
version: 0.4.0
---

# Robot Vision Alignment RL Playbook (Server-Aligned)

## Objective
Make the pointer’s tip (its lowest point) line up horizontally with the target’s centre line. There is no tilt control, so ignore any vertical difference.

## Perception (you are the only detector)
- Look at each frame yourself and find the marker and the target. No local detector exists.
- The marker is mounted on the camera, so it stays at the same place in every frame. The target is the object named in the run parameters.
- x increases to the right. offset = target_x - marker_x (px). Positive: the target is right of the marker.
- Always report `target_visible` honestly. Say false if you are unsure it is the right object; a wrong sighting wastes steps.

## Actions (call act() every step)
- `rotate_robot(amount)`: degrees, positive = turn left. Turning left normally shifts the scene right in the image; check against what you see. Coarse corrections.
- `pan_camera(amount)`: change in servo position. Its direction and pixels-per-unit scale are not known: work them out from the previous and current frames and your history. Good for fine corrections.
- `move_forward / move_backward / move_left / move_right(amount)`: metres. Open loop and imprecise, and it changes the scene. Use only if rotate and pan cannot help.
- `hold`: no motion this step. `stop`: ends the episode as a failure; use only if something looks unsafe.
- `finish`: only when the target centre is within the tolerance of the marker's tip. It must be confirmed on several consecutive fresh frames with nothing moving in between; any other action resets the count. Never finish when the target is not visible.
- Step limits and the tolerance are in the run parameters. A request over a limit is refused, and any refusal ends the episode.

## Stage 1: SEARCH (target not visible)
- The backend takes over: it turns the robot in fixed steps and shows you a frame after each.
- While the message says SEARCH MODE, only report `target_visible` and your offset estimate. Your motion requests are ignored.
- The first frame with the target holds still. If it is seen again on the next frame, the stage ends and your chosen action runs on that frame.
- If the whole circle is scanned without a sighting, the run ends as "target not found".
- If the target is lost for several steps in a row during alignment, the search starts again.

## Stage 2: ALIGN (target visible)
1. Estimate the offset from the frame.
2. Work out the correction from the calibration: amount = -offset / px-per-unit. Send it in ONE move when it fits inside the step limit.
3. Apply the share of the error given in LEARNED NOTES (70% if there are no notes). When the target is within about 3x the tolerance, switch to fine corrections (pan).
4. Next frame: compare with the previous one. Learn what the action actually did. If a step made things worse, reverse it.
5. When within tolerance, call `finish` and keep calling it while the offset stays within tolerance.

## How you are scored
- Progress made per step is the main term; the cost of a motion is tiny. One correct larger move beats several timid ones.
- Tiny steps are right only when you are unsure of the direction or the scale.
- Safety faults are heavily penalised.

## Always
- Answer by calling act(). Keep the observation short.
- Treat LEARNED NOTES as data from earlier runs, not rules. If the current frames contradict them, trust the frames.