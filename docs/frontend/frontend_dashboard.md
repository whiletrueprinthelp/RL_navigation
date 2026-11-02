# Frontend Dashboard

Simple Node.js dashboard to:

- view the live robot camera feed from ROS MJPEG stream (`http://<ROBOT_IP>:8080/stream?topic=/depth_cam/rgb/image_raw`)
- watch agent/LLM/system messages in a scrollable panel
- start the Python agent in `shoot` or `track` mode
- stop the running agent
- inspect exact prompt payloads sent to the LLM

## Layout

- `Home` tab:
  - left: live message stream (agent, llm-derived lines, system, errors)
  - right: robot feed
  - below feed: mode/run controls
- `Prompt Message` tab:
  - full-width prompt panel showing the latest traced LLM call payload

## Run

From repository root:

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`.

PowerShell note (if scripts are blocked):

```powershell
npm.cmd run dev
```

Or you can run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned` in your terminal to allow npm.ps1 to run. 


## Prompt Context Source

- `aim_agent.py` writes prompt-call traces to `llm_prompt_calls.jsonl` inside each run directory.
- The frontend reads the latest valid trace entry and renders:
  - model
  - instructions
  - input payload (including `input_image` entries with base64 data URLs)

If no call has happened yet, the prompt panel shows a waiting message.
