# TIM-BOT (AIMBOT)

Pure-vision robot alignment project using an LLM control loop.

The agent (`aim_agent.py`) reads camera frames from `robot_server.py`, asks the LLM to estimate horizontal offset and choose actions, executes bounded robot commands, logs outcomes, and updates prompt-side guidance from prior runs.

## Documentation

- **Robot server docs**
  - API contract: `docs/robot/api_contract.md`
  - ROS nodes/topics map: `docs/robot/nodes_and_topics.md`
  - Server behavior and safety rationale: `docs/robot/robot_server_behaviour.md`
- **Agent docs**
  - How the loop works: `docs/agent/how_agent_works.md`
  - Runtime defaults/hardcoded assumptions: `docs/agent/hardcoded_values.md`
  - Recommended run presets: `docs/agent/run_presets.md`
  - Not-yet-implemented features and rationale: `docs/agent/not_implemented_features.md`
- **Operations docs**
  - Day-to-day runbook and troubleshooting: `docs/regular_operations.md`

## Repository Roles

- `robot_server.py`: Runs **on the robot**. Exposes safe HTTP routes (`/status`, `/image`, `/rotate`, `/camera/pan`, `/move`, `/stop`) and enforces hard limits.
- `aim_agent.py`: Runs **on operator laptop/PC**. Calls `robot_server.py`, calls Azure OpenAI Responses API, and runs shoot/track control loops.
- `skills.md`: Policy/playbook given to the LLM for visual alignment behavior.

## System Architecture (Physical Setup)

1. JetAuto robot runs ROS2 and `robot_server.py`.
2. Operator machine runs `aim_agent.py`.
3. Both are on the same network.
4. Operator machine has Azure OpenAI credentials in `.env`.
5. A visible target object is in camera view (or searchable by scan).

## Prerequisites

### Robot side

- ROS2 environment sourced.
- No conflicting control app (for example WonderAi game) publishing motion commands.
- `ROBOT_TOKEN` set before starting server.

Start server on robot:

```bash
export ROBOT_TOKEN=choose-a-secret
python3 robot_server.py
```

### Operator side

Install dependencies:

```bash
pip install requests openai python-dotenv
```

(Optional if using `--grid` overlay):

```bash
pip install opencv-python numpy
```

Create `.env` in repo root:

```env
ROBOT_URL=http://<robot-ip>:8081
ROBOT_TOKEN=<same secret as robot>
AZURE_OPENAI_ENDPOINT=https://<resource>.services.ai.azure.com
AZURE_OPENAI_API_KEY=<azure-key>
AZURE_OPENAI_DEPLOYMENT=<deployment-name>
```

Head to `docs/regular_operations.md` for more detailed step-by-step in running the project.

## Run Modes

`aim_agent.py` supports two modes:

- `shoot`: Round-based, first-hit behavior. Ends when aligned `finish` succeeds.
- `track`: Continuous follow behavior for a time/step budget. No round check-frame gating.

Detailed mode behavior: `docs/agent/how_agent_works.md`.

## Recommended Commands

Additional presets and tuning baselines: `docs/agent/run_presets.md`.

### Shoot (default behavior)

```bash
python aim_agent.py --run-mode shoot --out runs
```

### Track (stable baseline)

```bash
python aim_agent.py --run-mode track --out runs_track --no-learn --track-seconds 180 --max-steps 300 --min-interval 0.3
```

### Track with anti-jitter deadband guard

```bash
python aim_agent.py --run-mode track --out runs_track --no-learn --track-seconds 180 --max-steps 300 --min-interval 0.3 --deadband-guard
```

## Key Runtime Knobs

Full default values and hardcoded assumptions: `docs/agent/hardcoded_values.md`.

- `--tol`: pixel alignment tolerance.
- `--max-steps`: shoot = alignment frames across rounds; track = total control steps.
- `--k-stable`: aligned frames required for confirmation.
- `--track-seconds`: max wall time for track mode.
- `--min-interval`: pacing floor between track loop steps.
- `--attempts-per-round`, `--max-rounds`, `--object-move`: shoot-only round controls.
- `--width`, `--detail`: image payload/quality controls (latency vs visual detail).

## Outputs and Artifacts

Each run writes to `runs/<episode_id>/` (or `runs_track/<episode_id>/` if using track with default output root):

- `step_###.jpg`: captured frames.
- `log.jsonl`: per-step records (actions, status, timings, reward parts, tokens).
- `episode_summary.json`: run summary and metrics.
- `run_config.json`: exact runtime configuration for reproducibility.
- `learned.md` (run-local copy): backend-written learned notes from prior runs.

## Learning and Reward Model

This project is **RL-style orchestration**, not model-weight training:

- Reward is computed from transition outcomes (progress/tracking error, control cost, safety penalties, success bonus).
- Reward is logged for evaluation/analysis.
- Improvement comes from backend calibration (`learn()`), dynamic caps, and updated prompt notes (`learned.md`), not gradient updates to LLM weights.

For project scope limits and why independent scorer/auto-sweep are not in baseline, see `docs/agent/not_implemented_features.md`.

## Safety Notes

- Detailed robot safety behavior: `docs/robot/robot_server_behaviour.md`.
- API response semantics and route contract: `docs/robot/api_contract.md`.
- ROS node/topic dependencies and checks: `docs/robot/nodes_and_topics.md`.

- `robot_server.py` is the safety authority; it enforces movement limits and abort conditions.
- Any server refusal/abort/busy is treated as terminal by the current loop logic.
- Keep emergency stop access available during physical runs.