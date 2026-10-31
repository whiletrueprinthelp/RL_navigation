const socket = io();

const logPane = document.getElementById('logPane');
const statusBadge = document.getElementById('statusBadge');
const feedMeta = document.getElementById('feedMeta');
const feedImage = document.getElementById('feedImage');
const feedFallback = document.getElementById('feedFallback');

const startShootBtn = document.getElementById('startShootBtn');
const startTrackBtn = document.getElementById('startTrackBtn');
const objectMovedBtn = document.getElementById('objectMovedBtn');
const stopBtn = document.getElementById('stopBtn');

let state = { running: false, mode: null, awaitingObjectMove: false };
let frameTimer = null;

function formatTime(ts) {
  try {
    return new Date(ts).toLocaleTimeString();
  } catch {
    return '';
  }
}

function setState(next) {
  state = { ...state, ...next };

  if (state.running) {
    statusBadge.textContent = `Running (${state.mode})`;
    statusBadge.className = 'status running';
  } else {
    statusBadge.textContent = 'Idle';
    statusBadge.className = 'status idle';
  }

  startShootBtn.disabled = state.running;
  startTrackBtn.disabled = state.running;
  objectMovedBtn.disabled = !(state.running && state.mode === 'shoot' && state.awaitingObjectMove);
  stopBtn.disabled = !state.running;
}

function appendMessage(msg) {
  const nearBottom = logPane.scrollTop + logPane.clientHeight >= logPane.scrollHeight - 24;
  const isAgentExit = /^Agent exited \(code=.*\)\.$/i.test(msg.text || '');

  const box = document.createElement('article');
  box.className = `msg ${msg.source || 'agent'}${isAgentExit ? ' exit-msg' : ''}`;

  const meta = document.createElement('div');
  meta.className = 'meta';
  meta.textContent = `${(msg.source || 'agent').toUpperCase()} - ${formatTime(msg.ts)}`;

  const text = document.createElement('div');
  text.className = 'text';
  text.textContent = msg.text || '';

  box.append(meta, text);
  logPane.appendChild(box);

  while (logPane.children.length > 300) {
    logPane.removeChild(logPane.firstElementChild);
  }

  if (nearBottom) {
    logPane.scrollTop = logPane.scrollHeight;
  }
}

async function fetchState() {
  const r = await fetch('/api/state');
  const data = await r.json();
  setState(data);
}

async function fetchMessages() {
  const r = await fetch('/api/messages');
  const data = await r.json();
  logPane.innerHTML = '';
  for (const msg of data.messages || []) {
    appendMessage(msg);
  }
}

async function startMode(mode) {
  const r = await fetch('/api/start', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ mode }),
  });
  const data = await r.json();
  if (!data.ok) {
    appendMessage({ source: 'error', text: `Start failed: ${data.reason}` });
  }
}

async function stopMode() {
  const r = await fetch('/api/stop', { method: 'POST' });
  const data = await r.json();
  if (!data.ok) {
    appendMessage({ source: 'error', text: `Stop failed: ${data.reason}` });
  }
}

async function confirmObjectMoved() {
  const r = await fetch('/api/object-moved', { method: 'POST' });
  const data = await r.json();
  if (!data.ok) {
    appendMessage({ source: 'error', text: `Object move confirm failed: ${data.reason}` });
  }
}

async function refreshFrame() {
  try {
    const r = await fetch('/api/frame');
    const data = await r.json();
    if (!data.ok || !data.frameDataUrl) {
      feedImage.style.display = 'none';
      feedFallback.style.display = 'block';
      feedMeta.textContent = data.reason || 'Frame unavailable';
      return;
    }

    feedImage.src = data.frameDataUrl;
    feedImage.style.display = 'block';
    feedFallback.style.display = 'none';
    feedMeta.textContent = `${data.width}x${data.height} - age ${Number(data.age_s || 0).toFixed(2)}s`;
  } catch (err) {
    feedImage.style.display = 'none';
    feedFallback.style.display = 'block';
    feedMeta.textContent = String(err);
  }
}

function startFramePolling() {
  if (frameTimer) {
    clearInterval(frameTimer);
  }
  refreshFrame();
  frameTimer = setInterval(refreshFrame, 350);
}

socket.on('state', setState);
socket.on('message', appendMessage);

startShootBtn.addEventListener('click', () => startMode('shoot'));
startTrackBtn.addEventListener('click', () => startMode('track'));
objectMovedBtn.addEventListener('click', confirmObjectMoved);
stopBtn.addEventListener('click', stopMode);

fetchState().catch(() => undefined);
fetchMessages().catch(() => undefined);
startFramePolling();

