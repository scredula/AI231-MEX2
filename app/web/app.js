/* Hey Mason — smart-home simulator front-end.
   Talks to the Python bridge: GET /events (SSE) + small JSON POSTs.          */
'use strict';

const $ = (id) => document.getElementById(id);
const DIAL_LEN = 289.03;                 // circumference of the r=46 dial
const esc = (s) => String(s).replace(/[&<>"]/g,
  (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const frac = (v, lo, hi) => Math.max(0, Math.min(1, (v - lo) / (hi - lo)));

let STATE = null;
let MIC = false;

function post(path, body) {
  return fetch(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body || {}),
  }).then((r) => r.json()).catch(() => null);
}

function setStatus(text, live) {
  $('status').textContent = text;
  $('pulse').classList.toggle('on', !!live);
}

/* --------------------------------------------------------------- rendering */
function renderLights(L) {
  const bulb = $('bulb');
  bulb.style.setProperty('--bulb', L.color);
  bulb.style.setProperty('--bright', (0.18 + 0.62 * (L.brightness / 100)).toFixed(3));
  bulb.classList.toggle('on', !!L.on);
  $('light-state').textContent = L.on ? 'on' : 'off';
  $('light-bright').textContent = L.brightness + ' %';
  $('light-color').textContent = L.color_name;
}

function renderWeather(w) {
  const d = w.data || {};
  $('w-temp').textContent = d.ok ? d.temp_c : '--';
  $('w-icon').textContent = d.icon || '\u26a0\ufe0f';
  $('w-desc').textContent = d.description || 'unavailable';
  $('w-feels').textContent = d.ok ? 'feels ' + d.feels_c + '\u00b0' : 'feels --\u00b0';
  $('w-hum').textContent = d.ok ? d.humidity + ' % hum' : '-- % hum';
  $('w-wind').textContent = d.ok ? d.wind_kmh + ' km/h' : '-- km/h';
  $('w-place').textContent = d.place || 'Cebu, PH';
  const bits = ['source: ' + (d.source || '\u2014')];
  if (d.fetched_at) bits.push(d.fetched_at);
  if (d.error) bits.push(d.error);
  $('w-meta').textContent = bits.join(' \u00b7 ');
}

function renderClimate(t) {
  $('thermo-val').textContent = t.target_c;
  $('dial-arc').style.strokeDashoffset =
    (DIAL_LEN * (1 - frac(t.target_c, 16, 28))).toFixed(1);
}

function renderTimer(t) {
  const arc = $('timer-arc');
  if (!t.duration_s) {
    $('timer-val').textContent = '--';
    $('timer-unit').textContent = '';
    arc.style.stroke = '#2b3a6b';
    arc.style.strokeDashoffset = DIAL_LEN;
    return;
  }
  const rem = t.remaining_s;
  if (rem >= 60) {
    const m = Math.floor(rem / 60);
    $('timer-val').textContent = m + ':' + String(Math.floor(rem % 60)).padStart(2, '0');
    $('timer-unit').textContent = '';
  } else {
    $('timer-val').textContent = Math.ceil(rem);
    $('timer-unit').textContent = 's';
  }
  arc.style.strokeDashoffset = (DIAL_LEN * (1 - frac(rem, 0, t.duration_s))).toFixed(1);
  arc.style.stroke = t.running ? 'var(--warn)' : 'var(--accent2)';
}

function renderMusic(m) {
  const active = m.playing || m.paused;
  $('music-track').textContent = active ? (m.track || 'demo track')
                                        : '\u2014 nothing playing \u2014';
  $('music-state').textContent = m.paused ? 'paused' : (m.playing ? 'playing' : 'stopped');
  $('music-icon').textContent = (m.playing && !m.paused) ? '\ud83c\udfb6' : '\u266b';
  $('vol-fill').style.width = Math.round(m.volume * 100) + '%';
  $('vol-val').textContent = Math.round(m.volume * 100) + ' %';
}

function renderAlarms(list) {
  $('alarms').innerHTML = list.length
    ? list.map((a) => '<span class="chip">\u23f0 ' + esc(a.label) + '</span>').join('')
    : '<span class="sub2">no alarms</span>';
}

function renderReminders(list) {
  $('reminders').innerHTML = list.length
    ? list.map((r) => '<li>' + esc(r.text) + '</li>').join('')
    : '<li class="sub2">no reminders</li>';
}

function renderAssistant(s) {
  $('clock').textContent = s.clock.now;
  $('clock').classList.toggle('hot', !!s.clock.highlight);
  $('call-state').textContent = 'call: ' + (s.call.active ? 'active' : 'idle');
  $('msg-state').textContent = 'messages: ' + s.message.unread;
  const b = $('msg-bubble');
  if (s.message.last) { b.textContent = s.message.last; b.classList.remove('hidden'); }
  else b.classList.add('hidden');
}

function renderWake(w) {
  $('wake-prob').textContent = Number(w.prob).toFixed(3);
  $('wake-thr').textContent = w.threshold;
  $('wake-fill').style.width = Math.min(100, w.prob * 100) + '%';
  $('wake-thresh').style.left = (w.threshold * 100) + '%';
  $('wake-state').textContent = w.listening ? 'listening\u2026' : 'idle';
  setStatus(w.listening ? 'listening for a command\u2026'
                        : (MIC ? 'mic live \u00b7 idle' : 'simulation mode'),
            w.listening || MIC);
}

function renderModels(m) {
  $('wake-model').textContent = m.wake.name;
  $('clf-pill').textContent = 'classifier ' + m.classifier.name +
    (m.classifier.real ? '' : ' (placeholder)');
  $('foot-intents').textContent = m.intents.length + ' intents \u00b7 static server: webui.py';
}

function renderTopk(top, source, intent, conf) {
  if (top && top.length) {
    $('topk').innerHTML = top.map((t, i) =>
      '<li class="' + (i === 0 ? 'top' : '') + '"><span>' + esc(t.intent) +
      '</span><span class="tk-p">' + (t.p * 100).toFixed(1) + '%</span></li>').join('');
  }
  if (intent && conf != null) {
    $('last-cmd').textContent = intent + ' \u00b7 ' + (conf * 100).toFixed(1) +
      '% (' + source + ')';
  }
}

function renderLog(entries) {
  const el = $('log');
  el.innerHTML = (entries || []).slice(-40).map((e) =>
    '<div><span class="t">' + esc(e.t) + '</span>' + esc(e.msg) + '</div>').join('');
  el.scrollTop = el.scrollHeight;
}

function render(state) {
  STATE = state;
  renderLights(state.lights);
  renderWeather(state.weather);
  renderClimate(state.thermostat);
  renderTimer(state.timer);
  renderMusic(state.music);
  renderAlarms(state.alarms);
  renderReminders(state.reminders);
  renderAssistant(state);
  renderWake(state.wake);
  renderModels(state.models);
  renderLog(state.log);
  const last = state.last || {};
  if (last.intent) {
    $('last-cmd').textContent = last.intent +
      (last.confidence ? ' \u00b7 ' + (last.confidence * 100).toFixed(1) + '%' : '') +
      (last.text ? ' \u00b7 "' + last.text + '"' : '');
  }
}

function setMic(on) {
  MIC = !!on;
  $('mic-btn').textContent = MIC ? 'Stop mic' : 'Start mic';
  $('mic-pill').textContent = MIC ? 'mic live' : 'mic off';
}

/* -------------------------------------------------------------------- SSE  */
function connect() {
  const es = new EventSource('/events');
  es.onopen = () => setStatus('connected', MIC);
  es.onerror = () => setStatus('reconnecting\u2026', false);
  es.onmessage = (ev) => {
    if (!ev.data) return;
    let msg;
    try { msg = JSON.parse(ev.data); } catch (e) { return; }
    if (msg.type === 'hello' || msg.type === 'state') {
      render(msg.state);
      setMic(msg.state.mic);
    } else if (msg.type === 'wake') {
      renderWake({
        prob: msg.prob, threshold: msg.threshold,
        listening: msg.listening || msg.triggered || false,
      });
    } else if (msg.type === 'prediction') {
      renderTopk(msg.top, msg.source, msg.intent, msg.confidence);
    } else if (msg.type === 'mic') {
      setMic(msg.on);
    }
  };
}

/* ----------------------------------------------------------------- wiring */
document.querySelectorAll('[data-intent]').forEach((b) => {
  b.addEventListener('click', () => post('/command', { intent: b.dataset.intent }));
});

$('cmd-form').addEventListener('submit', (e) => {
  e.preventDefault();
  const t = $('cmd-input').value.trim();
  if (!t) return;
  post('/command', { text: t });
  $('cmd-input').value = '';
});

$('mic-btn').addEventListener('click', () => {
  post('/mic', { on: !MIC }).then((r) => { if (r && r.mic != null) setMic(r.mic); });
});
$('wake-btn').addEventListener('click', () => {
  post('/wake').then((r) => { if (r && !r.ok && r.reason) setStatus(r.reason, false); });
});
$('reset-btn').addEventListener('click', () => post('/reset'));

/* refresh the live weather in the background every 10 minutes */
setInterval(() => {
  fetch('/weather?force=1').then((r) => r.json()).then((w) => {
    if (STATE) { STATE.weather.data = w; renderWeather(STATE.weather); }
  }).catch(() => {});
}, 600000);

setStatus('connecting\u2026', false);
connect();

