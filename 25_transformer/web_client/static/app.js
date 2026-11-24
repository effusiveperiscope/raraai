/* Stage-2 Studio — single-page client for the web_client server.
 *
 * No framework, no build step.  Everything is driven by /api/options (models,
 * variants, capabilities), then one job at a time: POST /api/generate or
 * /api/infill, poll /api/jobs/{id}, point <audio> at .../audio.wav.
 */
"use strict";

const $ = (id) => document.getElementById(id);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const state = {
  session: null,
  apiKey: null,
  authRequired: false,
  options: null,
  defaults: {},
  caps: {},
  speakers: [],
  current: null,
  history: [],
  log: [],
  busy: false,
  polling: false,
  initialized: false,
};

/* ------------------------------------------------------------------ util */
function newSession() {
  try {
    if (crypto && crypto.randomUUID) return crypto.randomUUID();
  } catch (_) { /* insecure context */ }
  return "s2-" + Math.random().toString(36).slice(2) + Date.now().toString(36);
}

function toast(message, isError) {
  const el = $("toast");
  el.textContent = String(message);
  el.classList.toggle("err", !!isError);
  el.classList.add("show");
  clearTimeout(toast._t);
  toast._t = setTimeout(() => el.classList.remove("show"), isError ? 6000 : 2600);
}

function logLine(message) {
  const stamp = new Date().toLocaleTimeString([], { hour: "2-digit",
    minute: "2-digit", second: "2-digit" });
  state.log.push(`[${stamp}] ${message}`);
  if (state.log.length > 200) state.log.shift();
  const card = $("logCard");
  card.hidden = false;
  const pre = $("logText");
  pre.textContent = state.log.join("\n");
  pre.scrollTop = pre.scrollHeight;
}

function withSession(url, extra) {
  const u = new URL(url, location.origin);
  u.searchParams.set("session", state.session);
  if (state.apiKey) u.searchParams.set("key", state.apiKey);
  for (const [k, v] of Object.entries(extra || {})) u.searchParams.set(k, v);
  return u.pathname + u.search;
}

async function api(path, opts) {
  const options = opts || {};
  const headers = { "X-Studio-Session": state.session };
  if (state.apiKey) headers["X-API-Key"] = state.apiKey;
  let body;
  if (options.body !== undefined) {
    headers["Content-Type"] = "application/json";
    body = JSON.stringify(options.body);
  }
  let res;
  try {
    res = await fetch(path, { method: options.method || "GET", headers, body });
  } catch (err) {
    throw new Error("cannot reach the server (" + err.message + ")");
  }
  if (res.status === 401) {
    state.authRequired = true;
    showAuth();
    throw new Error("access key required");
  }
  let data = null;
  try { data = await res.json(); } catch (_) { /* non-JSON error page */ }
  if (!res.ok) {
    const detail = data && (data.detail || data.message);
    throw new Error(detail || `${res.status} ${res.statusText}`);
  }
  return data;
}

/* ------------------------------------------------------------------ auth */
function showAuth() {
  $("authModal").hidden = false;
  setTimeout(() => $("authKey").focus(), 30);
}

function hideAuth() { $("authModal").hidden = true; }

/* --------------------------------------------------------------- options */
async function bootstrap() {
  state.session = localStorage.getItem("s2s_session") || newSession();
  localStorage.setItem("s2s_session", state.session);
  state.apiKey = sessionStorage.getItem("s2s_key") || null;

  let ping;
  try {
    const res = await fetch("/api/ping");
    ping = await res.json();
  } catch (err) {
    setStatus("err", "server offline");
    toast("Cannot reach the server", true);
    return;
  }
  state.authRequired = !!ping.auth_required;
  $("footerTitle").textContent = ping.title || "Stage-2 Studio";
  document.title = ping.title || "Stage-2 Studio";
  $("modePill").hidden = !ping.single_client;
  if (ping.single_client) $("modePill").textContent = "single user";

  if (state.authRequired && !state.apiKey) {
    showAuth();
    return;
  }
  hideAuth();
  await loadOptions();
  state.initialized = true;
}

async function loadOptions() {
  const data = await api("/api/options");
  state.options = data;
  state.defaults = data.defaults || {};
  const app = data.app || {};
  $("footerTitle").textContent = app.title || "Stage-2 Studio";
  $("subtitle").textContent = app.subtitle || "phoneme-driven neural speech";
  $("deviceText").textContent = data.device || "cpu";
  document.title = app.title || "Stage-2 Studio";

  const select = $("model");
  select.innerHTML = "";
  for (const model of data.models || []) {
    const group = document.createElement("optgroup");
    group.label = model.label;
    for (const variant of model.variants || []) {
      const opt = document.createElement("option");
      opt.value = `${model.id}::${variant.id}`;
      opt.textContent = variant.label;
      opt.title = variant.notes || model.notes || "";
      group.appendChild(opt);
    }
    select.appendChild(group);
  }
  if (!select.options.length) {
    toast("No models are configured on the server", true);
    setStatus("err", "no models");
    return;
  }
  const want = `${state.defaults.model}::${state.defaults.variant}`;
  select.value = [...select.options].some((o) => o.value === want)
    ? want : select.options[0].value;
  await applySelection({ initial: true });
  setStatus("ok", "ready");
}

function selection() {
  const value = $("model").value || "";
  const [modelId, variantId] = value.split("::");
  const model = (state.options.models || []).find((m) => m.id === modelId);
  const variant = model ? (model.variants || []).find((v) => v.id === variantId) : null;
  return { modelId, variantId, model, variant, caps: (variant && variant.caps) || {} };
}

async function applySelection({ initial } = {}) {
  const sel = selection();
  if (!sel.model || !sel.variant) return;
  const caps = sel.caps || {};
  state.caps = caps;

  $("modelNote").textContent = sel.variant.notes || sel.model.notes || "";
  renderCaps(caps);

  const frames = $("frames");
  const maxFrames = Math.max(25, Math.min(512, Number(caps.max_frames) || 512));
  frames.max = String(maxFrames);
  if (initial) frames.value = String(Math.min(maxFrames, state.defaults.frames || 200));
  if (Number(frames.value) > maxFrames) frames.value = String(maxFrames);
  updateFrameLabel();

  if (initial) {
    $("temperature").value = String(state.defaults.temperature ?? 0.9);
    $("topk").value = String(state.defaults.top_k ?? 20);
    $("intensity").value = String(state.defaults.intensity ?? 0);
    $("register").value = String(state.defaults.register ?? 0);
    $("pitchShift").value = String(state.defaults.pitch_shift ?? 0);
    $("slotShift").value = String(state.defaults.slot_shift ?? 0);
    $("wordSem").value = state.defaults.word_semantics || "null";
    $("hintBase").value = "null";
    $("fillBestOf").value = "1";
  }
  updateControlLabels();

  // Hint panel
  const hintPanel = $("hintPanel");
  hintPanel.hidden = !caps.hints;
  hintPanel.classList.toggle("disabled", !$("hintsEnable").checked);
  $("intensity").max = String(caps.n_intensity || 8);
  $("register").max = String(caps.n_register || 5);
  $("wordSemField").hidden = !caps.word_sem;

  // FIM
  if (!caps.fim || !state.current
      || state.current.model !== sel.modelId
      || state.current.variant !== sel.variantId) {
    $("fillPanel").hidden = true;
  }
  $("fillBestOfField").hidden = !(Number(caps.fim_lookahead) > 0);
  if ($("fillBestOfField").hidden) $("fillBestOf").value = "1";

  // Voices
  await loadSpeakers(sel, initial);
  $("sayBtn").disabled = state.busy || !$("text").value.trim();
  $("sayBtnMobile").disabled = $("sayBtn").disabled;
}

function renderCaps(caps) {
  const box = $("capsChips");
  box.innerHTML = "";
  const chips = [];
  const add = (label, on, cls) => chips.push({ label, on, cls });
  add(caps.eos ? "EOS" : "fixed length", !!caps.eos);
  add("FIM", !!caps.fim);
  add("RoPE", !!caps.rope);
  if (caps.hints) {
    add("hints", true);
    if (caps.pitch_mode === "anchors") add(`${caps.pitch_anchors} pitch anchors`, true);
    if (caps.word_sem) add("word BERT", true);
  }
  add(caps.audio ? "vocoder" : "codes only", !!caps.audio);
  if (caps.lora) add(`LoRA r${caps.lora_rank}`, true, "lora");
  for (const chip of chips) {
    const el = document.createElement("span");
    el.className = "chip" + (chip.on ? " on" : "") + (chip.cls ? " " + chip.cls : "");
    el.textContent = chip.label;
    box.appendChild(el);
  }
}

/* ---------------------------------------------------------------- voices */
async function loadSpeakers(sel, initial) {
  const select = $("voice");
  const previous = initial ? Number(state.defaults.sid)
    : Number(select.value || select.dataset.sid || 0);
  $("voiceSearch").value = "";
  select.innerHTML = "";
  const loading = document.createElement("option");
  loading.textContent = "loading voices…";
  select.appendChild(loading);
  try {
    const data = await api(`/api/speakers?model=${encodeURIComponent(sel.modelId)}&variant=${encodeURIComponent(sel.variantId)}`);
    state.speakers = data.speakers || [];
  } catch (err) {
    state.speakers = [];
    toast("Could not load voices: " + err.message, true);
  }
  renderVoiceOptions();
  const exists = state.speakers.some((s) => Number(s.id) === previous);
  const chosen = exists ? previous : (state.speakers[0] ? Number(state.speakers[0].id) : 0);
  renderVoiceOptions("", chosen);
}

function renderVoiceOptions(filter, forceSid) {
  const select = $("voice");
  const current = forceSid !== undefined ? String(forceSid)
    : (select.value || select.dataset.sid || "");
  const needle = String(filter || "").trim().toLowerCase();
  select.innerHTML = "";
  let matched = 0;
  for (const s of state.speakers) {
    const hay = `${s.label} ${s.id}`.toLowerCase();
    if (needle && !hay.includes(needle)) continue;
    const opt = document.createElement("option");
    opt.value = String(s.id);
    opt.textContent = `${s.label} · ${s.count}`;
    opt.title = `speaker ${s.id} — ${s.count} validation clip(s)`;
    select.appendChild(opt);
    matched += 1;
  }
  if (!matched && state.speakers.length) {
    // Filter hid everything: keep the current voice selectable rather than
    // silently switching it.
    const s = state.speakers.find((x) => String(x.id) === String(current));
    if (s) {
      const opt = document.createElement("option");
      opt.value = String(s.id);
      opt.textContent = `${s.label} · ${s.count}`;
      select.appendChild(opt);
    }
  }
  if (current && [...select.options].some((o) => o.value === String(current))) {
    select.value = String(current);
  } else if (select.options.length) {
    select.selectedIndex = 0;
  }
  select.dataset.sid = select.value;
}

/* ------------------------------------------------------------ controls */
function updateFrameLabel() {
  const frames = Number($("frames").value);
  const fr = Number(state.caps.frame_rate) || 25;
  $("framesOut").textContent = `${frames} frames · ${(frames / fr).toFixed(1)} s`;
}

function updateControlLabels() {
  const temp = Number($("temperature").value);
  const topk = Number($("topk").value);
  $("tempOut").textContent = temp.toFixed(2);
  $("topkOut").textContent = topk === 0 ? "off" : String(topk);
  $("advSummary").textContent = `temp ${temp.toFixed(2)} · top-k ${topk === 0 ? "off" : topk}`;

  const intOn = $("hintsEnable").checked;
  const int = Number($("intensity").value);
  const reg = Number($("register").value);
  const pitch = Number($("pitchShift").value);
  const slot = Number($("slotShift").value);
  $("intOut").textContent = !intOn ? "off" : (int === 0 ? "off" : String(int));
  $("regOut").textContent = !intOn ? "off" : (reg === 0 ? "off" : String(reg));
  $("pitchOut").textContent = !intOn ? "off" : `${pitch > 0 ? "+" : ""}${pitch} st`;
  $("slotOut").textContent = !intOn ? "off" : `${slot > 0 ? "+" : ""}${slot}`;
}

function setBusy(on, label) {
  state.busy = on;
  const text = label || "working…";
  for (const id of ["sayBtn", "sayBtnMobile"]) {
    const btn = $(id);
    if (!btn) continue;
    btn.classList.toggle("loading", on);
    btn.disabled = on || !$("text").value.trim();
    btn.querySelector(".btn-label").textContent = on ? text : "Say it";
  }
  if (on) setStatus("busy", text);
  else setStatus("ok", "ready");
}

function setStatus(kind, text) {
  const dot = $("statusDot");
  dot.className = "dot" + (kind === "ok" ? " ok" : kind === "busy" ? " busy"
    : kind === "err" ? " err" : "");
  $("statusText").textContent = text;
}

function shortStage(stage) {
  const s = String(stage || "").toLowerCase();
  if (s.includes("loading")) return "loading model…";
  if (s.includes("phonem")) return "phonemizing…";
  if (s.includes("sampl") || s.includes("regenerat")) return "sampling…";
  if (s.includes("decod")) return "decoding…";
  if (s.includes("packag")) return "finishing…";
  if (s.includes("queued")) return "queued…";
  if (s.includes("start")) return "starting…";
  return s ? s.slice(0, 26) : "working…";
}

/* -------------------------------------------------------------- request */
function currentParams() {
  const sel = selection();
  const caps = state.caps;
  const body = {
    text: $("text").value.trim(),
    model: sel.modelId,
    variant: sel.variantId,
    sid: Number($("voice").value),
    frames: Number($("frames").value),
    temperature: Number($("temperature").value),
    top_k: Number($("topk").value),
  };
  const seed = $("seed").value.trim();
  if (seed !== "") body.seed = Number(seed);
  if (caps.hints && $("hintsEnable").checked) {
    body.intensity = Number($("intensity").value);
    body.register = Number($("register").value);
    body.pitch_shift = Number($("pitchShift").value);
    body.slot_shift = Number($("slotShift").value);
    body.base = $("hintBase").value;
    if (caps.word_sem) body.word_semantics = $("wordSem").value;
  }
  return body;
}

async function waitFor(jobId) {
  state.polling = true;
  try {
    let last = "";
    for (;;) {
      const job = await api(`/api/jobs/${jobId}`);
      if (job.status === "queued") {
        const label = job.queue_position > 0
          ? `queued (#${job.queue_position + 1})` : "queued…";
        if (label !== last) { setBusy(true, label); logLine(label); last = label; }
      } else if (job.status === "running") {
        const label = shortStage(job.stage);
        if (job.stage !== last) { setBusy(true, label); logLine(job.stage); last = job.stage; }
        else setStatus("busy", job.stage);
      } else if (job.status === "done") {
        return job.result;
      } else if (job.status === "error") {
        throw new Error(job.error || "generation failed");
      }
      await sleep(450);
    }
  } finally {
    state.polling = false;
  }
}

async function doGenerate() {
  if (state.busy) return;
  const text = $("text").value.trim();
  if (!text) { toast("Type something for it to say first"); $("text").focus(); return; }
  if (!$("voice").value) { toast("Pick a voice first", true); return; }
  try {
    logLine(`generate · ${text.slice(0, 80)}`);
    setBusy(true, "queued…");
    const { job_id } = await api("/api/generate", {
      method: "POST", body: currentParams(),
    });
    const result = await waitFor(job_id);
    logLine(`done in ${result.timings ? `${Math.round((result.timings.generate_ms || 0) + (result.timings.decode_ms || 0))} ms` : "?"} · ${result.frames} frames`);
    addHistory(result);
    showResult(result);
  } catch (err) {
    logLine(`error: ${err.message}`);
    toast(err.message, true);
  } finally {
    setBusy(false);
    syncSayDisabled();
  }
}

async function doInfill() {
  if (state.busy) return;
  const current = state.current;
  if (!current) { toast("Generate a clip first", true); return; }
  if (!state.caps.fim) { toast("This checkpoint has no FIM support", true); return; }
  try {
    const params = currentParams();
    delete params.text;
    delete params.sid;
    delete params.frames;
    const body = {
      ...params,
      source_job: current.job_id,
      start: Number($("fillStart").value),
      frames: Number($("fillSpan").value),
      best_of: Number($("fillBestOf").value),
    };
    logLine(`infill · frames ${body.start}..${body.start + body.frames - 1} of ${current.frames}`);
    setBusy(true, "queued…");
    const { job_id } = await api("/api/infill", { method: "POST", body });
    const result = await waitFor(job_id);
    addHistory(result);
    showResult(result);
    const st = result.span ? `frames ${result.span.start}..${result.span.start + result.span.frames - 1}` : "span";
    toast(`Regenerated ${st}`);
  } catch (err) {
    logLine(`error: ${err.message}`);
    toast(err.message, true);
  } finally {
    setBusy(false);
    syncSayDisabled();
  }
}

function syncSayDisabled() {
  const disabled = state.busy || !$("text").value.trim();
  $("sayBtn").disabled = disabled;
  $("sayBtnMobile").disabled = disabled;
  if (!state.busy) setStatus("ok", "ready");
}

/* --------------------------------------------------------------- result */
function modelLabel(result) {
  const model = (state.options && state.options.models || []).find((m) => m.id === result.model);
  if (!model) return result.model;
  const variant = (model.variants || []).find((v) => v.id === result.variant);
  return `${model.label}${variant ? " / " + variant.label : ""}`;
}

function showResult(r) {
  state.current = r;
  const card = $("resultCard");
  card.hidden = false;

  $("resultKind").textContent = r.kind === "infill" && r.span
    ? `regenerated span · ${r.span.start}–${r.span.start + r.span.frames - 1}`
    : "generated";
  $("resultVoice").textContent = `${r.sid_label || "Speaker"} (${r.sid})`;
  $("resultDuration").textContent = `${Number(r.duration).toFixed(2)} s`;

  const player = $("player");
  if (r.audio_url) {
    player.src = withSession(r.audio_url);
    player.hidden = false;
    const dl = $("downloadBtn");
    dl.href = withSession(r.audio_url, { download: 1 });
    dl.hidden = false;
  } else {
    player.removeAttribute("src");
    player.load();
    player.hidden = true;
    $("downloadBtn").hidden = true;
  }
  $("codesBtn").href = withSession(r.codes_url);

  renderMeta(r);
  $("phonesText").textContent = r.phones || "—";
  const sel = selection();
  const sameModel = sel.modelId === r.model && sel.variantId === r.variant;
  setupFill(sameModel ? r : null);
  renderHistory();

  if (window.innerWidth < 720) {
    card.scrollIntoView({ behavior: "smooth", block: "start" });
  }
}

function renderMeta(r) {
  const rows = [];
  rows.push(["Model", modelLabel(r)]);
  rows.push(["Text", r.text]);
  rows.push(["Frames", `${r.frames} @ ${r.frame_rate} Hz`]);
  rows.push(["Temperature", Number(r.temperature).toFixed(2)]);
  rows.push(["Top-k", r.top_k == null ? "off" : String(r.top_k)]);
  rows.push(["Seed", String(r.seed)]);
  if (r.f0) {
    rows.push(["Voiced", `${Math.round((r.f0.voiced_frac || 0) * 100)} %`]);
    rows.push(["Median f0", r.f0.median_hz ? `${Math.round(r.f0.median_hz)} Hz` : "—"]);
  }
  if (r.timings) {
    const gen = r.timings.generate_ms != null ? `${Math.round(r.timings.generate_ms)} ms` : "—";
    const dec = r.timings.decode_ms != null ? `${Math.round(r.timings.decode_ms)} ms` : "—";
    rows.push(["Sample / decode", `${gen} / ${dec}`]);
  }
  if (r.eos_length != null) rows.push(["EOS length", `${r.eos_length} frames`]);
  if (r.hints_state && !r.hints_state.startsWith("n/a")) {
    rows.push(["Hints", r.hints_state]);
  }
  if (r.span) rows.push(["Span", `${r.span.start}..${r.span.start + r.span.frames - 1} of ${r.frames}`]);
  if (r.seam_ce != null) rows.push(["Seam CE", `${Number(r.seam_ce).toFixed(3)} nats`]);
  if (r.best_of && r.best_of > 1) rows.push(["Candidates", `${r.best_of}`]);
  if (r.notes && r.notes.length) rows.push(["Notes", r.notes.join(" · ")]);
  rows.push(["Unique codes", (r.unique_codes || []).join(" ")]);

  const grid = $("metaGrid");
  grid.innerHTML = "";
  for (const [label, value] of rows) {
    const div = document.createElement("div");
    const dt = document.createElement("dt");
    dt.textContent = label;
    const dd = document.createElement("dd");
    dd.textContent = value == null || value === "" ? "—" : String(value);
    div.appendChild(dt);
    div.appendChild(dd);
    grid.appendChild(div);
  }
}

/* ------------------------------------------------------------------ fill */
function setupFill(r) {
  if (!r) { $("fillPanel").hidden = true; return; }
  const n = Number(r.frames) || 0;
  if (!state.caps.fim || n < 3) { $("fillPanel").hidden = true; return; }
  $("fillPanel").hidden = false;
  const span = Math.max(2, Math.min(n - 1, Math.round(n / 3)));
  const start = Math.max(0, Math.min(n - span, Math.round((n - span) / 2)));
  $("fillSpan").value = String(span);
  $("fillStart").value = String(start);
  syncFillSliders();
}

function syncFillSliders(changed) {
  const r = state.current;
  const n = r ? Number(r.frames) : 100;
  let start = Number($("fillStart").value);
  let span = Number($("fillSpan").value);

  if (changed === "start") {
    span = Math.min(span, Math.max(1, n - start));
  } else if (changed === "span") {
    start = Math.min(start, Math.max(0, n - span));
  }
  start = Math.max(0, Math.min(start, n - 1));
  span = Math.max(1, Math.min(span, n - start));
  $("fillStart").max = String(Math.max(0, n - span));
  $("fillSpan").max = String(Math.max(1, n - start));
  $("fillStart").value = String(start);
  $("fillSpan").value = String(span);

  $("fillStartOut").textContent = `${start} (${(start / (Number(state.caps.frame_rate) || 25)).toFixed(2)} s)`;
  $("fillSpanOut").textContent = `${span} (${(span / (Number(state.caps.frame_rate) || 25)).toFixed(2)} s)`;

  const w = Math.max(1, n);
  $("fillKept").style.flex = `${start + 0.6}`;
  $("fillHole").style.flex = `${span}`;
  $("fillKept2").style.flex = `${Math.max(0, n - start - span) + 0.6}`;
}

/* --------------------------------------------------------------- history */
function addHistory(r) {
  state.history = state.history.filter((x) => x.job_id !== r.job_id);
  state.history.unshift(r);
  if (state.history.length > 12) state.history.length = 12;
  $("historyCard").hidden = state.history.length === 0;
}

function renderHistory() {
  const list = $("historyList");
  list.innerHTML = "";
  for (const r of state.history) {
    const li = document.createElement("li");
    if (state.current && state.current.job_id === r.job_id) li.className = "active";
    const btn = document.createElement("button");
    btn.type = "button";
    const text = document.createElement("span");
    text.className = "h-text";
    text.textContent = r.text || "(no text)";
    const meta = document.createElement("span");
    meta.className = "h-meta";
    meta.textContent = `${Number(r.duration).toFixed(1)}s · ${r.sid_label || r.sid}`;
    btn.appendChild(text);
    btn.appendChild(meta);
    btn.addEventListener("click", () => showResult(r));
    li.appendChild(btn);
    list.appendChild(li);
  }
  $("historyCard").hidden = state.history.length === 0;
}

/* ----------------------------------------------------------------- ideas */
const IDEAS = [
  "The stars are beautiful tonight. Shall we go for a walk?",
  "I told you, Paimon is not emergency food!",
  "Well… that could have gone better.",
  "Use the tool before you lose it.",
  "Hello there, traveller. You don't sound too excited.",
  "Monday, Tuesday, Wednesday, Thursday. Five days, one adventure.",
];
let ideaIndex = 0;

/* ------------------------------------------------------------------ wire */
function wire() {
  const text = $("text");
  text.addEventListener("input", () => {
    $("chars").textContent = String(text.value.length);
    syncSayDisabled();
  });
  text.addEventListener("keydown", (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
      e.preventDefault();
      doGenerate();
    }
  });
  $("clearBtn").addEventListener("click", () => {
    text.value = "";
    $("chars").textContent = "0";
    text.focus();
    syncSayDisabled();
  });
  $("sayBtn").addEventListener("click", doGenerate);
  $("sayBtnMobile").addEventListener("click", doGenerate);
  $("againBtn").addEventListener("click", doGenerate);
  $("fillBtn").addEventListener("click", doInfill);
  $("ideaBtn").addEventListener("click", () => {
    text.value = IDEAS[ideaIndex++ % IDEAS.length];
    $("chars").textContent = String(text.value.length);
    syncSayDisabled();
    text.focus();
  });

  $("model").addEventListener("change", () => {
    applySelection({}).catch((err) => toast(err.message, true));
  });
  $("voiceSearch").addEventListener("input", () => renderVoiceOptions($("voiceSearch").value));
  $("voice").addEventListener("change", () => { $("voice").dataset.sid = $("voice").value; });

  $("frames").addEventListener("input", updateFrameLabel);
  $("temperature").addEventListener("input", updateControlLabels);
  $("topk").addEventListener("input", updateControlLabels);
  $("intensity").addEventListener("input", updateControlLabels);
  $("register").addEventListener("input", updateControlLabels);
  $("pitchShift").addEventListener("input", updateControlLabels);
  $("slotShift").addEventListener("input", updateControlLabels);
  $("hintsEnable").addEventListener("change", () => {
    $("hintPanel").classList.toggle("disabled", !$("hintsEnable").checked);
    updateControlLabels();
  });
  $("seedBtn").addEventListener("click", () => {
    $("seed").value = String(Math.floor(Math.random() * 2 ** 31));
  });

  $("fillStart").addEventListener("input", () => syncFillSliders("start"));
  $("fillSpan").addEventListener("input", () => syncFillSliders("span"));

  $("clearHistory").addEventListener("click", async () => {
    state.history = [];
    state.current = null;
    renderHistory();
    $("resultCard").hidden = true;
    $("player").removeAttribute("src");
    $("player").load();
    try { await api("/api/history", { method: "DELETE" }); } catch (_) { /* ignore */ }
  });
  $("clearLog").addEventListener("click", () => {
    state.log = [];
    $("logText").textContent = "";
    $("logCard").hidden = true;
  });

  $("authForm").addEventListener("submit", async (e) => {
    e.preventDefault();
    const key = $("authKey").value.trim();
    if (!key) return;
    state.apiKey = key;
    sessionStorage.setItem("s2s_key", key);
    hideAuth();
    try {
      await loadOptions();
      state.initialized = true;
      setStatus("ok", "ready");
    } catch (err) {
      sessionStorage.removeItem("s2s_key");
      state.apiKey = null;
      showAuth();
      toast(err.message, true);
    }
  });
}

/* ------------------------------------------------------------------ init */
wire();
updateControlLabels();
bootstrap().catch((err) => {
  setStatus("err", "failed");
  toast(err.message, true);
});
