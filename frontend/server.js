const express = require('express');
const path = require('path');
const fsp = require('fs/promises');
const { spawn } = require('child_process');
const readline = require('readline');
const http = require('http');
const { Server } = require('socket.io');
const dotenv = require('dotenv');

const FRONTEND_ROOT = __dirname;
const REPO_ROOT = path.resolve(__dirname, '..');

dotenv.config({ path: path.join(REPO_ROOT, '.env') });

function requireEnv(name) {
  const value = process.env[name];
  if (!value || !value.trim()) {
    throw new Error(`Missing required env var ${name} in ${path.join(REPO_ROOT, '.env')}`);
  }
  return value;
}

const PORT = Number(process.env.FRONTEND_PORT || 5173);
const ROBOT_URL = requireEnv('ROBOT_URL').replace(/\/$/, '');
const ROBOT_TOKEN = requireEnv('ROBOT_TOKEN');
const ROBOT_IP = (process.env.ROBOT_IP || '').trim() || new URL(ROBOT_URL).hostname;
const LIVE_STREAM_TOPIC = process.env.LIVE_STREAM_TOPIC || '/depth_cam/rgb/image_raw';
const LIVE_STREAM_URL = `http://${ROBOT_IP}:8080/stream?topic=${LIVE_STREAM_TOPIC}`;
const PYTHON_CMD = process.env.AGENT_PYTHON || 'python';
const AGENT_SCRIPT = process.env.AGENT_SCRIPT || path.join(REPO_ROOT, 'aim_agent.py');
const MAX_MESSAGES = Number(process.env.MAX_MESSAGES || 500);

const SHOOT_ARGS = splitArgs(process.env.SHOOT_MODE_ARGS || '');
const TRACK_ARGS = splitArgs(process.env.TRACK_MODE_ARGS || '--out runs_track');
const COMMON_ARGS = splitArgs(process.env.COMMON_AGENT_ARGS || '');

const app = express();
const server = http.createServer(app);
const io = new Server(server);

app.use(express.json());
app.use(express.static(path.join(FRONTEND_ROOT, 'public')));

const runtime = {
  agent: null,
  mode: null,
  startedAt: null,
  currentRunDir: null,
  awaitingObjectMove: false,
  promptTracePath: null,
  promptTraceOffset: 0,
  latestPromptRecord: null,
  tailInterval: null,
  tailOffset: 0,
  messages: [],
};

const MANUAL_MOVE_PROMPT_RE = /round\s+\d+:\s+move the object, then press Enter \(q \+ Enter ends the run\):/i;

function splitArgs(raw) {
  if (!raw || !raw.trim()) {
    return [];
  }
  const out = [];
  const pattern = /"([^"]*)"|'([^']*)'|\S+/g;
  let match;
  while ((match = pattern.exec(raw)) !== null) {
    out.push(match[1] ?? match[2] ?? match[0]);
  }
  return out;
}

function isRunning() {
  return !!runtime.agent;
}

function emitState() {
  io.emit('state', {
    running: isRunning(),
    mode: runtime.mode,
    startedAt: runtime.startedAt,
    runDir: runtime.currentRunDir,
    awaitingObjectMove: runtime.awaitingObjectMove,
  });
}

function setAwaitingObjectMove(value) {
  if (runtime.awaitingObjectMove === value) {
    return;
  }
  runtime.awaitingObjectMove = value;
  emitState();
}

function addMessage(msg) {
  const full = { ...msg, ts: new Date().toISOString() };
  runtime.messages.push(full);
  if (runtime.messages.length > MAX_MESSAGES) {
    runtime.messages = runtime.messages.slice(-MAX_MESSAGES);
  }
  io.emit('message', full);
}

function clearTail() {
  if (runtime.tailInterval) {
    clearInterval(runtime.tailInterval);
    runtime.tailInterval = null;
  }
  runtime.tailOffset = 0;
}

function clearPromptTrace() {
  runtime.promptTracePath = null;
  runtime.promptTraceOffset = 0;
  runtime.latestPromptRecord = null;
}

async function refreshPromptTrace() {
  if (!runtime.promptTracePath) {
    return;
  }
  try {
    const text = await fsp.readFile(runtime.promptTracePath, 'utf-8');
    const lines = text.split(/\r?\n/).filter(Boolean);

    for (let i = lines.length - 1; i >= 0; i -= 1) {
      try {
        runtime.latestPromptRecord = JSON.parse(lines[i]);
        runtime.promptTraceOffset = Buffer.byteLength(text, 'utf-8');
        return;
      } catch {
        // Keep scanning backwards for the latest complete valid JSON line.
      }
    }
  } catch {
    // file may not exist yet
  }
}

function formatPromptRecord(rec) {
  const request = rec?.request || {};
  const parts = [
    `Episode: ${rec.episode_id || 'unknown'}`,
    `Step: ${rec.step ?? 'n/a'} | Phase: ${rec.phase || 'n/a'} | Attempt: ${rec.attempt ?? 'n/a'}`,
    `Model: ${request.model || 'n/a'}`,
    '',
    'INSTRUCTIONS',
    String(request.instructions || ''),
    '',
    'INPUT',
    JSON.stringify(request.input || null, null, 2),
  ];
  return parts.join('\n');
}

function resolveRunDir(runDirText) {
  if (!runDirText || typeof runDirText !== 'string') {
    return null;
  }
  return path.isAbsolute(runDirText) ? runDirText : path.join(REPO_ROOT, runDirText);
}

async function tailLogFile(logPath) {
  try {
    const stat = await fsp.stat(logPath);
    if (stat.size <= runtime.tailOffset) {
      return;
    }

    const fh = await fsp.open(logPath, 'r');
    const size = stat.size - runtime.tailOffset;
    const buffer = Buffer.alloc(size);
    await fh.read(buffer, 0, size, runtime.tailOffset);
    await fh.close();
    runtime.tailOffset = stat.size;

    const chunk = buffer.toString('utf-8');
    const lines = chunk.split(/\r?\n/).filter(Boolean);
    for (const line of lines) {
      try {
        const row = JSON.parse(line);
        const obs = row.observation || '';
        const action = row.action || 'unknown';
        const amount = row.amount === null || row.amount === undefined ? '' : ` ${row.amount}`;
        addMessage({
          source: 'llm',
          text: `${obs}\nAction: ${action}${amount}`,
        });
      } catch {
        addMessage({ source: 'agent', text: line });
      }
    }
  } catch {
    // File may not exist yet; ignore quietly.
  }
}

function startTail(runDir) {
  const resolvedRunDir = resolveRunDir(runDir);
  if (!resolvedRunDir) {
    return;
  }
  clearTail();
  clearPromptTrace();
  runtime.currentRunDir = resolvedRunDir;
  runtime.promptTracePath = path.join(resolvedRunDir, 'llm_prompt_calls.jsonl');
  emitState();
  const logPath = path.join(resolvedRunDir, 'log.jsonl');
  runtime.tailInterval = setInterval(() => {
    tailLogFile(logPath).catch(() => undefined);
  }, 400);
}

function stopAgent(reason = 'stopped') {
  if (!runtime.agent) {
    return false;
  }
  clearTail();
  setAwaitingObjectMove(false);
  addMessage({ source: 'system', text: `Stopping agent (${reason})...` });
  runtime.agent.kill('SIGINT');
  return true;
}

function modeArgs(mode) {
  if (mode === 'shoot') {
    if (SHOOT_ARGS.includes('--object-move')) {
      return SHOOT_ARGS;
    }
    return [...SHOOT_ARGS, '--object-move', 'manual'];
  }
  if (mode === 'track') {
    const normalizedTrackArgs = TRACK_ARGS.map((arg) => (arg === '--nolearn' ? '--no-learn' : arg));
    if (normalizedTrackArgs.includes('--no-learn')) {
      return normalizedTrackArgs;
    }
    return [...normalizedTrackArgs, '--no-learn'];
  }
  return [];
}

function launchAgent(mode) {
  const args = [AGENT_SCRIPT, '--run-mode', mode, ...COMMON_ARGS, ...modeArgs(mode)];
  const childEnv = {
    ...process.env,
    PYTHONIOENCODING: 'utf-8',
    PYTHONUTF8: '1',
    PYTHONUNBUFFERED: '1',
  };
  const child = spawn(PYTHON_CMD, args, {
    cwd: REPO_ROOT,
    env: childEnv,
    stdio: ['pipe', 'pipe', 'pipe'],
  });

  runtime.agent = child;
  runtime.mode = mode;
  runtime.startedAt = new Date().toISOString();
  runtime.currentRunDir = null;
  runtime.awaitingObjectMove = false;
  clearTail();
  clearPromptTrace();

  addMessage({ source: 'system', text: `Started ${mode} mode: ${PYTHON_CMD} ${args.join(' ')}` });
  emitState();

  const outRl = readline.createInterface({ input: child.stdout });
  outRl.on('line', (line) => {
    let shownLine = line;
    if (runtime.awaitingObjectMove) {
      shownLine = shownLine.replace(MANUAL_MOVE_PROMPT_RE, '').trim();
    }
    if (shownLine) {
      addMessage({ source: 'agent', text: shownLine });
    }
    if (/move the object now; the agent looks again in/i.test(line)) {
      setAwaitingObjectMove(false);
    }
    const match = line.match(/episode:\s+\S+\s+\|\s+dir:\s+(.+)$/i);
    if (match?.[1]) {
      startTail(match[1].trim());
    }
  });

  child.stdout.on('data', (chunk) => {
    const text = chunk.toString('utf-8');
    const promptMatch = text.match(MANUAL_MOVE_PROMPT_RE);
    if (promptMatch && !runtime.awaitingObjectMove) {
      addMessage({ source: 'agent', text: promptMatch[0] });
      setAwaitingObjectMove(true);
    }
  });

  const errRl = readline.createInterface({ input: child.stderr });
  errRl.on('line', (line) => {
    addMessage({ source: 'error', text: line });
  });

  child.on('close', (code, signal) => {
    clearTail();
    addMessage({ source: 'system', text: `Agent exited (code=${code}, signal=${signal || 'none'}).` });
    runtime.agent = null;
    runtime.mode = null;
    runtime.startedAt = null;
    runtime.currentRunDir = null;
    runtime.awaitingObjectMove = false;
    emitState();
  });
}

app.get('/api/state', (_req, res) => {
  res.json({
    running: isRunning(),
    mode: runtime.mode,
    startedAt: runtime.startedAt,
    runDir: runtime.currentRunDir,
    awaitingObjectMove: runtime.awaitingObjectMove,
  });
});

app.get('/api/messages', (_req, res) => {
  res.json({ messages: runtime.messages.slice(-200) });
});

app.get('/api/prompt', async (_req, res) => {
  await refreshPromptTrace();

  if (runtime.latestPromptRecord) {
    const rec = runtime.latestPromptRecord;
    return res.json({
      ok: true,
      meta: `LLM call: step ${rec.step ?? 'n/a'} (${rec.phase || 'unknown'}) attempt ${rec.attempt ?? 'n/a'}`,
      text: formatPromptRecord(rec),
    });
  }

  if (!runtime.currentRunDir) {
    return res.json({
      ok: true,
      meta: 'Waiting for run...',
      text: 'No active run yet. Start shoot/track to display exact LLM prompt calls.',
    });
  }

  return res.json({
    ok: true,
    meta: `Run ${path.basename(runtime.currentRunDir)} (waiting for first LLM call)`,
    text: 'Prompt trace file not available yet. Wait for the first LLM call in this run.',
  });
});

app.get('/api/feed-source', (_req, res) => {
  res.json({
    ok: true,
    streamUrl: LIVE_STREAM_URL,
    robotIp: ROBOT_IP,
    topic: LIVE_STREAM_TOPIC,
  });
});

app.post('/api/start', (req, res) => {
  const mode = req.body?.mode;
  if (!['shoot', 'track'].includes(mode)) {
    return res.status(400).json({ ok: false, reason: 'mode must be shoot or track' });
  }
  if (isRunning()) {
    return res.status(409).json({ ok: false, reason: 'agent already running' });
  }
  launchAgent(mode);
  res.json({ ok: true });
});

app.post('/api/stop', (_req, res) => {
  if (!isRunning()) {
    return res.status(409).json({ ok: false, reason: 'agent is not running' });
  }
  stopAgent('user request');
  res.json({ ok: true });
});

app.post('/api/object-moved', (_req, res) => {
  if (!isRunning()) {
    return res.status(409).json({ ok: false, reason: 'agent is not running' });
  }
  if (runtime.mode !== 'shoot') {
    return res.status(409).json({ ok: false, reason: 'object-moved is only valid in shoot mode' });
  }
  if (!runtime.agent.stdin || runtime.agent.stdin.destroyed || !runtime.awaitingObjectMove) {
    return res.status(409).json({ ok: false, reason: 'agent is not waiting for manual move confirmation' });
  }
  runtime.agent.stdin.write('\n');
  addMessage({ source: 'system', text: 'Operator confirmed object moved. Resuming next round.' });
  setAwaitingObjectMove(false);
  return res.json({ ok: true });
});

io.on('connection', (socket) => {
  socket.emit('state', {
    running: isRunning(),
    mode: runtime.mode,
    startedAt: runtime.startedAt,
    runDir: runtime.currentRunDir,
    awaitingObjectMove: runtime.awaitingObjectMove,
  });
});

server.listen(PORT, () => {
  // eslint-disable-next-line no-console
  console.log(`Frontend server running at http://localhost:${PORT}`);
  // eslint-disable-next-line no-console
  console.log(`Live feed source: ${LIVE_STREAM_URL}`);
});

process.on('SIGINT', () => {
  stopAgent('server shutdown');
  process.exit(0);
});

