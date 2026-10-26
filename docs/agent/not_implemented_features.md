# Not Yet Implemented Features (and Why)

This note captures important components that are reasonable to add in future versions, but are intentionally not in the current baseline.

## Current project posture

The current system is best described as **RL-style control and evaluation**, not full reinforcement learning training:

- The loop observes state, takes actions, and computes per-step rewards.
- Rewards are logged and used for analysis/tuning.
- Prompt-side behavior is adjusted via `learned.md` and action caps.
- **No model weights are updated** by gradient-based RL during runs.

This keeps the stack simple and operational for physical testing, while still allowing policy iteration at the system level.

## Feature: independent scorer (not implemented)

### What it would add

- A separate perception path that estimates marker/target offset independently of the LLM.
- Reward terms and success checks based on this independent signal.
- Better separation between “actor” (LLM decisions) and “evaluator” (objective metric).

### Why it is currently not included

- The LLM-only vision loop is already capable of stable alignment/tracking in current experiments.
- Maintaining a second vision stack (color/shape/depth rules, calibration, drift handling) adds complexity and failure modes.
- For this project phase, speed of iteration and robustness under varied scenes were prioritized over strict scorer independence.

### Known limitation from not having it

- Reward is partly self-referential because it uses LLM-estimated offsets.
- This can inflate or distort reward quality when visual estimates are noisy or biased.
- It weakens claims of objective policy improvement compared with systems using external ground-truth scoring.

## Feature: automatic marker calibration / auto-sweep (not implemented)

### What it would add

- Automatic scan/calibration routine to identify marker position and robust alignment reference at startup.
- Reduced manual assumptions for marker tip location and tolerance behavior.
- More consistent initialization across hardware resets and camera remounting.

### Why it is currently not included

- Current setup assumes a fixed camera-mounted marker and works sufficiently when hardware is unchanged.
- Auto-sweep introduces extra startup time, additional state transitions, and additional recovery logic.
- Existing scan/search behavior already covers target reacquisition for normal runs.

### Known limitation from not having it

- If marker geometry/placement changes, performance can degrade until settings/prompts are adjusted.
- Cross-run comparability depends more on keeping hardware setup consistent.

## Why this is still a valid baseline

Even without independent scorer and auto-marker sweep, the project remains useful as a baseline for:

- end-to-end closed-loop robot control with safety constraints,
- mode comparison (`shoot` vs `track`),
- latency and action-quality analysis from run logs,
- incremental architecture evolution before adding heavier perception/training modules.

## Suggested next-step additions (when needed)

1. Add optional independent scorer behind a feature flag.
2. Log both LLM-estimated and scorer-estimated offsets for side-by-side analysis.
3. Define dual metrics:
   - operational reward (current),
   - objective reward (independent scorer).
4. Add optional startup marker auto-calibration routine for unstable hardware setups.


