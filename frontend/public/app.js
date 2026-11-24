const socket = io();

// Core dashboard panels.
const logPane = document.getElementById('logPane');
const statusBadge = document.getElementById('statusBadge');
const feedMeta = document.getElementById('feedMeta');
const feedImage = document.getElementById('feedImage');
const feedFallback = document.getElementById('feedFallback');
const promptMeta = document.getElementById('promptMeta');
const promptPane = document.getElementById('promptPane');
const homeTabBtn = document.getElementById('homeTabBtn');
const promptTabBtn = document.getElementById('promptTabBtn');
const homeView = document.getElementById('homeView');
const promptView = document.getElementById('promptView');

const startShootBtn = document.getElementById('startShootBtn');
const startTrackBtn = document.getElementById('startTrackBtn');
const objectMovedBtn = document.getElementById('objectMovedBtn');
const stopBtn = document.getElementById('stopBtn');
const targetInput = document.getElementById('targetInput');

// Frontend UI state mirrors backend /api/state payload.
let state = { running: false, mode: null, awaitingObjectMove: false };
let activeErrorTextNode = null;
let promptTimer = null;
let activeTab = 'home';
let promptHistory = [];
let lastPromptKey = null;
const MAX_PROMPT_HISTORY = 120;

function isPromptRecord(meta) {
  return typeof meta === 'string' && meta.startsWith('LLM call:');
}

function renderPromptHistory() {
  if (!promptHistory.length) {
    promptPane.textContent = 'No run prompt available yet.';
    return;
  }

  const blocks = promptHistory.map((entry, idx) => {
    const header = `=== Prompt ${idx + 1} | ${entry.meta} ===`;
    return `${header}\n${entry.text}`;
  });

  promptPane.textContent = blocks.join('\n\n');
}

function setActiveTab(tab) {
  // Toggle between operational dashboard (Home) and exact prompt viewer.
  activeTab = tab;
  const onHome = tab === 'home';

  homeTabBtn.classList.toggle('active', onHome);
  promptTabBtn.classList.toggle('active', !onHome);

  homeTabBtn.setAttribute('aria-selected', String(onHome));
  promptTabBtn.setAttribute('aria-selected', String(!onHome));

  homeView.classList.toggle('active', onHome);
  promptView.classList.toggle('active', !onHome);

  promptView.setAttribute('aria-hidden', String(onHome));
}

function formatTime(ts) {
  try {
    return new Date(ts).toLocaleTimeString();
  } catch {
    return '';
  }
}

function setState(next) {
  // Drive button visibility/enablement from current run mode and wait state.
  state = { ...state, ...next };

  if (state.running) {
    statusBadge.textContent = `Running (${state.mode})`;
    statusBadge.className = 'status running';
  } else {
    statusBadge.textContent = 'Idle';
    statusBadge.className = 'status idle';
  }

  const inShootMode = state.running && state.mode === 'shoot';
  const inTrackMode = state.running && state.mode === 'track';

  startShootBtn.style.display = state.running ? 'none' : 'inline-block';
  startTrackBtn.style.display = state.running ? 'none' : 'inline-block';
  targetInput.disabled = state.running;

  if (state.running && typeof state.target === 'string' && state.target.trim()) {
    targetInput.value = state.target;
  }

  objectMovedBtn.style.display = inShootMode ? 'inline-block' : 'none';
  objectMovedBtn.disabled = !state.awaitingObjectMove;

  stopBtn.style.display = state.running ? 'inline-block' : 'none';
  stopBtn.disabled = !(inShootMode || inTrackMode);
}

function appendMessage(msg) {
  // Append one message card; merge consecutive stderr chunks into a single error card.
  const nearBottom = logPane.scrollTop + logPane.clientHeight >= logPane.scrollHeight - 24;
  const isAgentExit = /^Agent exited \(code=.*\)\.$/i.test(msg.text || '');
  const isError = (msg.source || 'agent') === 'error';

  if (!isError) {
    activeErrorTextNode = null;
  }

  if (isError && activeErrorTextNode) {
    activeErrorTextNode.textContent += `\n${msg.text || ''}`;
    if (nearBottom) {
      logPane.scrollTop = logPane.scrollHeight;
    }
    return;
  }

  const box = document.createElement('article');
  box.className = `msg ${msg.source || 'agent'}${isAgentExit ? ' exit-msg' : ''}`;

  const meta = document.createElement('div');
  meta.className = 'meta';
  meta.textContent = `${(msg.source || 'agent').toUpperCase()} - ${formatTime(msg.ts)}`;

  const text = document.createElement('div');
  text.className = 'text';
  text.textContent = msg.text || '';

  if (isError) {
    activeErrorTextNode = text;
  }

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
  activeErrorTextNode = null;
  for (const msg of data.messages || []) {
    appendMessage(msg);
  }
}

async function startMode(mode) {
  // Start shoot/track via backend launcher.
  const target = (targetInput.value || '').trim() || 'a blue bottle';
  const r = await fetch('/api/start', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ mode, target }),
  });
  const data = await r.json();
  if (!data.ok) {
    appendMessage({ source: 'error', text: `Start failed: ${data.reason}` });
    return;
  }
  if (data.target) {
    targetInput.value = data.target;
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

async function initLiveStream() {
  // Fetch direct camera stream URL and bind it to the <img> element.
  try {
    const r = await fetch('/api/feed-source');
    const data = await r.json();
    if (!data.ok || !data.streamUrl) {
      feedImage.style.display = 'none';
      feedFallback.style.display = 'block';
      feedMeta.textContent = data.reason || 'Live stream unavailable';
      return;
    }

    feedMeta.textContent = `${data.robotIp}:8080 stream - topic ${data.topic}`;
    feedImage.style.display = 'block';
    feedFallback.style.display = 'none';
    feedImage.onerror = () => {
      feedImage.style.display = 'none';
      feedFallback.style.display = 'block';
      feedMeta.textContent = 'Failed to load live stream';
    };
    feedImage.src = data.streamUrl;
  } catch (err) {
    feedImage.style.display = 'none';
    feedFallback.style.display = 'block';
    feedMeta.textContent = String(err);
  }
}

async function refreshPrompt() {
  // Poll backend for latest traced LLM call payload.
  try {
    const r = await fetch('/api/prompt');
    const data = await r.json();
    if (!data.ok) {
      promptMeta.textContent = 'Prompt unavailable';
      promptPane.textContent = data.reason || 'Failed to load prompt context.';
      return;
    }

    const meta = data.meta || 'Prompt context';
    const text = data.text || 'No prompt text available.';

    if (isPromptRecord(meta)) {
      const promptKey = `${meta}\n${text}`;
      if (promptKey !== lastPromptKey) {
        promptHistory.push({ meta, text });
        if (promptHistory.length > MAX_PROMPT_HISTORY) {
          promptHistory = promptHistory.slice(-MAX_PROMPT_HISTORY);
        }
        lastPromptKey = promptKey;
      }

      promptMeta.textContent = `Prompt history (${promptHistory.length})`;
      const nearBottom = promptPane.scrollTop + promptPane.clientHeight >= promptPane.scrollHeight - 24;
      renderPromptHistory();
      if (nearBottom) {
        promptPane.scrollTop = promptPane.scrollHeight;
      }
      return;
    }

    promptMeta.textContent = meta;
    if (promptHistory.length) {
      renderPromptHistory();
    } else {
      promptPane.textContent = text;
    }
  } catch (err) {
    promptMeta.textContent = 'Prompt unavailable';
    promptPane.textContent = String(err);
  }
}

function startPromptPolling() {
  // Keep prompt panel fresh even when hidden so tab switch is instant.
  if (promptTimer) {
    clearInterval(promptTimer);
  }
  refreshPrompt();
  promptTimer = setInterval(refreshPrompt, 1500);
}

socket.on('state', setState);
socket.on('message', appendMessage);

startShootBtn.addEventListener('click', () => startMode('shoot'));
startTrackBtn.addEventListener('click', () => startMode('track'));
objectMovedBtn.addEventListener('click', confirmObjectMoved);
stopBtn.addEventListener('click', stopMode);
homeTabBtn.addEventListener('click', () => setActiveTab('home'));
promptTabBtn.addEventListener('click', () => {
  setActiveTab('prompt');
  refreshPrompt().catch(() => undefined);
});

fetchState().catch(() => undefined);
fetchMessages().catch(() => undefined);
initLiveStream().catch(() => undefined);
startPromptPolling();
setActiveTab(activeTab);

