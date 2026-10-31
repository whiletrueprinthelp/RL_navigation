# Frontend dashboard

Simple Node.js dashboard to:

- view live robot camera feed (`/image`)
- watch agent + LLM messages in a scrollable left panel
- start the Python agent in `shoot` or `track` mode
- stop the running agent

## Run

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`.

## Notes

- The server reads configuration from the repo root `.env`.
- Required values in root `.env`: `ROBOT_URL` and `ROBOT_TOKEN`.
- LLM messages are derived from `log.jsonl` rows (`observation`, `action`, `amount`) once an episode directory is created.

