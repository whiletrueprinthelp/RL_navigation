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
const PYTHON_CMD = process.env.AGENT_PYTHON || 'python';
const AGENT_SCRIPT = process.env.AGENT_SCRIPT || path.join(REPO_ROOT, 'aim_agent.py');
const FRAME_WIDTH = Number(process.env.FRAME_WIDTH || 640);
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
  tailInterval: null,
  tailOffset: 0,
  messages: [],
};

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
  clearTail();
  runtime.currentRunDir = runDir;
  emitState();
  const logPath = path.join(runDir, 'log.jsonl');
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
    return TRACK_ARGS;
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

  addMessage({ source: 'system', text: `Started ${mode} mode: ${PYTHON_CMD} ${args.join(' ')}` });
  emitState();

  const outRl = readline.createInterface({ input: child.stdout });
  outRl.on('line', (line) => {
    addMessage({ source: 'agent', text: line });
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
    if (/move the object, then press Enter/i.test(text)) {
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

app.get('/api/frame', async (req, res) => {
  const width = Number(req.query.width || FRAME_WIDTH);
  const url = `${ROBOT_URL}/image?width=${encodeURIComponent(width)}`;
  try {
    const r = await fetch(url, {
      headers: ROBOT_TOKEN ? { 'X-Token': ROBOT_TOKEN } : {},
    });
    if (!r.ok) {
      return res.status(r.status).json({ ok: false, reason: `robot_server returned ${r.status}` });
    }
    const payload = await r.json();
    if (payload?.status !== 'done' || !payload?.image_b64) {
      return res.status(502).json({ ok: false, reason: 'invalid /image payload', payload });
    }
    res.json({
      ok: true,
      age_s: payload.age_s,
      width: payload.width,
      height: payload.height,
      frameDataUrl: `data:image/jpeg;base64,${payload.image_b64}`,
    });
  } catch (err) {
    res.status(502).json({ ok: false, reason: String(err) });
  }
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
  console.log(`Proxying feed from ${ROBOT_URL}/image`);
});

process.on('SIGINT', () => {
  stopAgent('server shutdown');
  process.exit(0);
});

