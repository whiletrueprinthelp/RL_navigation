# Recommended Command Presets

Use these as practical starting points, then tune from logs.

## Shoot Mode (first-hit, round-based)

```bash
python aim_agent.py --run-mode shoot --out runs
```

- Keeps the current round/chance workflow.
- Ends on the first successful aligned `finish`.

## Track Mode (continuous follow)

```bash
python aim_agent.py --run-mode track --out runs_track --no-learn --track-seconds 180 --min-interval 0.3
```

- Runs continuous tracking (no rounds/check-frame).
- Uses a conservative default with `--no-learn` to avoid drift-contaminated calibration.

## Track Mode (with deadband anti-jitter guard)

```bash
python aim_agent.py --run-mode track --skills skills.md --out runs_track --no-learn --track-seconds 180 --min-interval 0.3 --deadband-guard
```

- Same as above, but converts motion requests to `hold` when already within tolerance.
- Useful when logs show chatter while aligned.
