# How the Agent Works

This project uses an **application-layer RL-style loop**: the LLM proposes actions from camera observations, the app executes safety-bounded robot commands, computes rewards, logs transitions, and updates run-time guidance from past episodes.

## End-to-end loop (simple)
1. **Observe**  
   The agent queries robot state and image data from `robot_server.py` (`GET /status`, `GET /image`).

2. **Decide**  
   The app builds a prompt from policy docs (`skills.md`) + learned notes and asks the LLM for a structured action (`act()` tool call).

3. **Act safely**  
   The app clips the action to configured + server limits and sends exact HTTP commands (`POST /rotate`, `POST /camera/pan`, `POST /move`, `POST /stop`).

4. **Evaluate**  
   Reward is computed at application level from observed transitions (progress, control cost, safety penalties, oscillation, success bonus).

5. **Learn (application layer)**  
   The app reads previous `runs/*` logs, estimates calibration trends, and rewrites `learned.md` for future runs.

## What is “RL-style” here?
- **State**: frame + status + short action/history context.
- **Action space**: high-level robot operations (`rotate`, `pan`, `move`, `hold`, `stop`, `finish`).
- **Reward**: computed from step-to-step outcome metrics, not a model loss tensor.
- **Policy improvement**: updates prompt-side guidance and dynamic caps from prior episodes.

## What this is *not*
- Not PPO/Q-learning inside model weights.
- No weight fine-tuning in this repo.
- Learning happens in orchestration logic and notes, not in neural parameter updates.

## Safety model
- `robot_server.py` is the safety authority and enforces hard limits.
- Agent-side constraints are advisory and conservative; server constraints remain final.
- Episodes terminate immediately on server-side safety faults (`refused|aborted|busy`).
- Local agent-side refusals are tolerated up to `--max-local-faults`, then the episode is ended.

## Artifacts and inspectability
Each run writes to `runs/<episode_id>/`:
- `step_###.jpg` (frames)
- `log.jsonl` (per-step decisions/results/reward)
- `episode_summary.json` (rollup metrics)

This makes trajectories and failures easy to audit.
