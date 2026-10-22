import {createSimulator, PRESETS, initialAreas, FREQUENCIES} from './simulator/index.js';

const $ = id => document.getElementById(id);
const NS = 'http://www.w3.org/2000/svg';
function element(tag, attributes = {}, text) {
  const node = document.createElementNS(NS, tag);
  for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, value);
  if (text !== undefined) node.textContent = text;
  return node;
}
function linePath(values, x, y) {
  return values.map((value, i) => `${i ? 'L' : 'M'}${x(i).toFixed(2)},${y(value).toFixed(2)}`).join(' ');
}
const state = {sim: null, areas: initialAreas(), target: PRESETS[0], length: PRESETS[0].length, pitch: 120, selected: 15, running: false, worker: null, iteration: 0, history: [], elapsedMs: 0, audio: null, source: null, audioButton: null, audioRequest: 0};
const palette = {current: '#146f68', target: '#b96537'};
let targetDb = [], currentDb = [], renderPending = false;

function setStatus(message) { $('fit-status').textContent = message; }
function syncControls() {
  $('section').value = state.selected + 1;
  $('section-value').textContent = `${state.selected + 1} / 32`;
  $('area').value = state.areas[state.selected];
  $('area-value').textContent = `${state.areas[state.selected].toFixed(2)} cm²`;
  $('tract-length').value = state.length;
  $('length-value').textContent = `${state.length} cm`;
  $('pitch-value').textContent = `${state.pitch} Hz`;
}
function drawTube() {
  const svg = $('tube-plot');
  const x = i => 43 + i / 31 * 690;
  const y = area => 148 - area / 16 * 126;
  svg.replaceChildren(element('title', {id: 'tube-title'}, 'Vocal tract cross-sectional area'), element('desc', {id: 'tube-description'}, 'Edit the profile by dragging, or use the section and area sliders. Glottis on the left, lips on the right.'));
  for (const area of [0, 4, 8, 12, 16]) {
    svg.append(element('line', {x1: 43, y1: y(area), x2: 733, y2: y(area), class: 'plot-grid'}));
    svg.append(element('text', {x: 31, y: y(area) + 4, 'text-anchor': 'end', class: 'plot-axis'}, area));
  }
  const current = linePath(state.areas, x, y);
  svg.append(element('path', {d: `${current} L733,148 L43,148 Z`, fill: '#dcebe3', opacity: '.85'}));
  if ($('show-target').checked) svg.append(element('path', {d: linePath(state.target.areas, x, y), class: 'plot-target'}));
  svg.append(element('path', {d: current, class: 'plot-current'}));
  svg.append(element('line', {x1: x(state.selected), x2: x(state.selected), y1: 18, y2: 148, stroke: palette.current, 'stroke-width': 1, opacity: .3}));
  svg.append(element('circle', {cx: x(state.selected), cy: y(state.areas[state.selected]), r: 4.5, fill: palette.current, stroke: '#fff', 'stroke-width': 2}));
  svg.append(element('text', {x: 43, y: 174, class: 'plot-axis'}, 'Glottis'));
  svg.append(element('text', {x: 733, y: 174, class: 'plot-axis', 'text-anchor': 'end'}, 'Lips'));
  svg.append(element('text', {x: 390, y: 174, class: 'plot-axis', 'text-anchor': 'middle'}, '32 cross-sectional areas · cm²'));
}
function drawSpectrum() {
  if (!targetDb.length) return;
  const svg = $('spectrum-plot');
  const reference = Math.max(...targetDb);
  const x = frequency => 43 + frequency / 5000 * 690;
  // Both curves use the target peak as the same dB reference.
  const level = db => 185 - (Math.max(-70, Math.min(15, db - reference)) + 70) / 85 * 165;
  svg.replaceChildren(element('title', {}, 'Current and target acoustic responses'));
  for (const tick of [-60, -40, -20, 0]) {
    const yy = level(reference + tick);
    svg.append(element('line', {x1: 43, x2: 733, y1: yy, y2: yy, class: 'plot-grid'}));
    svg.append(element('text', {x: 31, y: yy + 4, 'text-anchor': 'end', class: 'plot-axis'}, tick));
  }
  for (let frequency = 0; frequency <= 5000; frequency += 1000) {
    svg.append(element('text', {x: x(frequency), y: 207, 'text-anchor': 'middle', class: 'plot-axis'}, frequency === 0 ? '0' : `${frequency / 1000}k`));
  }
  svg.append(element('path', {d: linePath(targetDb, i => x(FREQUENCIES[i]), level), class: 'plot-target'}));
  svg.append(element('path', {d: linePath(currentDb, i => x(FREQUENCIES[i]), level), class: 'plot-current'}));
  svg.append(element('text', {x: 43, y: 229, class: 'plot-axis'}, 'Magnitude (dB, target peak = 0)'));
  svg.append(element('text', {x: 733, y: 229, class: 'plot-axis', 'text-anchor': 'end'}, 'Frequency (Hz)'));
}
function drawLoss() {
  const svg = $('loss-plot');
  if (state.history.length < 2) {
    svg.replaceChildren(element('text', {x: 0, y: 51, class: 'plot-empty'}, 'Start fitting to watch the loss change.'));
    return;
  }
  const values = state.history.map(value => Math.log10(Math.max(1e-10, value)));
  const maximum = Math.max(...values), minimum = Math.min(...values, maximum - .1);
  const x = i => 5 + i / Math.max(239, values.length - 1) * 740;
  const y = value => 67 - (value - minimum) / (maximum - minimum) * 53;
  svg.replaceChildren(element('title', {}, 'Measured spectral loss, logarithmic scale'), element('line', {x1: 5, x2: 745, y1: 68, y2: 68, class: 'plot-grid'}));
  svg.append(element('path', {d: linePath(values, x, y), class: 'plot-current', 'stroke-width': 1.7}));
  svg.append(element('text', {x: 5, y: 96, class: 'plot-axis'}, 'Loss · log scale'));
  svg.append(element('text', {x: 745, y: 96, class: 'plot-axis', 'text-anchor': 'end'}, `${state.iteration} gradient updates`));
}
function render() {
  renderPending = false;
  syncControls(); drawTube(); drawSpectrum(); drawLoss();
  $('fit-step').textContent = state.iteration;
  const loss = state.history.at(-1);
  $('fit-loss').textContent = loss === undefined ? '—' : loss < .001 ? loss.toExponential(2) : loss.toFixed(3);
  $('fit-time').textContent = `${(state.elapsedMs / 1000).toFixed(1)} s`;
}
function queueRender() {
  if (!renderPending) { renderPending = true; requestAnimationFrame(render); }
}
function responseUpdated() {
  if (!state.sim) return;
  currentDb = state.sim.response(state.areas, state.length).magnitudeDb;
  queueRender();
}
function cancelFit(message) {
  if (state.worker) { state.worker.terminate(); state.worker = null; }
  state.running = false;
  $('fit').textContent = state.iteration ? 'Fit again →' : 'Fit to target →';
  if (message) setStatus(message);
}
function clearFit() {
  state.iteration = 0; state.history = []; state.elapsedMs = 0;
  $('fit').textContent = 'Fit to target →';
}
function editAreas() {
  stopAudio(); cancelFit(); clearFit();
  setStatus('Shape edited. Listen to the result, or fit this shape to the target.');
  responseUpdated();
}
function stopAudio() {
  state.audioRequest++;
  if (state.source) { state.source.onended = null; state.source.stop(); state.source = null; }
  if (state.audioButton) { state.audioButton.textContent = state.audioButton.id === 'listen-current' ? '▷ Listen to current' : '▷ Listen to target'; state.audioButton.removeAttribute('aria-pressed'); state.audioButton = null; }
}
async function playSound(which) {
  const button = $(`listen-${which}`);
  if (state.audioButton === button) { stopAudio(); return; }
  stopAudio();
  const request = state.audioRequest;
  for (const media of document.querySelectorAll('audio, video')) media.pause();
  try {
    const AudioContext = window.AudioContext || window.webkitAudioContext;
    if (!AudioContext) throw new Error('Audio playback is not supported by this browser.');
    state.audio ??= new AudioContext();
    await state.audio.resume();
    if (request !== state.audioRequest) return;
    const {samples, sampleRate} = state.sim.synthesize(which === 'target' ? state.target.areas : state.areas, state.length, state.pitch, 1.2);
    const buffer = state.audio.createBuffer(1, samples.length, sampleRate);
    buffer.copyToChannel(samples, 0);
    const source = state.audio.createBufferSource();
    source.buffer = buffer; source.connect(state.audio.destination);
    state.source = source; state.audioButton = button;
    button.textContent = '■ Stop sound'; button.setAttribute('aria-pressed', 'true');
    source.onended = () => { if (state.source === source) { state.source = null; stopAudio(); } };
    source.start();
  } catch (error) { if (request === state.audioRequest) { setStatus(error.message); stopAudio(); } }
}
function startFit() {
  if (!state.sim) return;
  if (state.running) { cancelFit('Fitting paused. The current shape is preserved. Fit again to continue with a fresh optimizer.'); return; }
  stopAudio(); cancelFit(); clearFit();
  state.running = true; $('fit').textContent = 'Pause fitting';
  setStatus('Backpropagating through the acoustic model…');
  let worker;
  try { worker = new Worker(new URL('./simulator/fit-worker.js', import.meta.url), {type: 'module'}); }
  catch { cancelFit('This browser could not start the fitting worker. You can still reshape the tube and listen.'); return; }
  state.worker = worker;
  worker.onmessage = ({data}) => {
    if (state.worker !== worker) return;
    if (data.type === 'error') { cancelFit(`Fitting stopped: ${data.message}`); return; }
    if (data.type !== 'progress' && data.type !== 'done') return;
    state.areas = data.areas; state.iteration = data.iteration; state.history = data.history; state.elapsedMs = data.elapsedMs;
    currentDb = data.magnitudeDb; queueRender();
    if (data.type === 'done') {
      const first = data.history[0];
      const reduction = first > 0 ? 100 * (1 - data.loss / first) : 0;
      let change = 'Spectral loss changed by less than 0.1%.';
      if (first === 0 && data.loss > 0) change = 'Spectral loss increased from zero. Reset the tube to begin a new experiment.';
      else if (reduction >= .05) change = `Spectral loss decreased by ${reduction >= 99.9 ? '>99.9' : reduction.toFixed(1)}%.`;
      else if (reduction <= -.05) change = `Spectral loss increased by ${(-reduction).toFixed(1)}%. Try resetting the tube before fitting again.`;
      cancelFit(`Finished ${data.iteration} updates. ${change} Listen to the fit or reveal the target shape.`);
    }
  };
  worker.onerror = () => { if (state.worker === worker) cancelFit('The fitting worker could not run. Reload the page to try again.'); };
  worker.postMessage({type: 'start', areas: [...state.areas], targetAreas: state.target.areas, length: state.length, iterations: 240, learningRate: .035});
}

for (const preset of PRESETS) {
  const button = document.createElement('button');
  button.type = 'button'; button.className = 'target-option'; button.textContent = preset.label; button.dataset.target = preset.id;
  button.setAttribute('aria-pressed', preset === state.target ? 'true' : 'false');
  button.disabled = true;
  button.addEventListener('click', () => {
    stopAudio(); cancelFit(); clearFit();
    state.target = preset; state.length = preset.length; state.areas = initialAreas();
    for (const option of $('target-options').children) option.setAttribute('aria-pressed', option === button ? 'true' : 'false');
    targetDb = state.sim.response(preset.areas, state.length).magnitudeDb;
    $('fit').textContent = 'Fit to target →';
    setStatus(`${preset.name} target selected. Start fitting from this neutral tube, or edit it first.`);
    responseUpdated();
  });
  $('target-options').append(button);
}
$('section').addEventListener('input', event => { state.selected = +event.target.value - 1; syncControls(); drawTube(); });
$('area').addEventListener('input', event => { state.areas[state.selected] = +event.target.value; editAreas(); });
$('tract-length').addEventListener('input', event => {
  stopAudio(); cancelFit(); clearFit(); state.length = +event.target.value;
  if (state.sim) targetDb = state.sim.response(state.target.areas, state.length).magnitudeDb;
  setStatus('Tract length changes both the current and target resonances.'); responseUpdated();
});
$('pitch').addEventListener('input', event => { stopAudio(); state.pitch = +event.target.value; syncControls(); });
$('show-target').addEventListener('change', drawTube);
$('listen-current').addEventListener('click', () => playSound('current'));
$('listen-target').addEventListener('click', () => playSound('target'));
$('fit').addEventListener('click', startFit);
$('reset').addEventListener('click', () => { stopAudio(); cancelFit(); clearFit(); state.areas = initialAreas(); $('fit').textContent = 'Fit to target →'; setStatus('Reset to a neutral tube. Choose a target and start fitting.'); responseUpdated(); });

let dragging = false;
function movePoint(event) {
  const svg = $('tube-plot');
  const bounds = svg.getBoundingClientRect();
  const x = (event.clientX - bounds.left) / bounds.width * 760;
  const y = (event.clientY - bounds.top) / bounds.height * 190;
  state.selected = Math.min(31, Math.max(0, Math.round((x - 43) / 690 * 31)));
  state.areas[state.selected] = Math.min(16, Math.max(.12, (148 - y) / 126 * 16));
  editAreas();
}
$('tube-plot').addEventListener('pointerdown', event => {
  if (!state.sim || event.button !== 0) return;
  dragging = true; $('tube-plot').setPointerCapture(event.pointerId); movePoint(event);
});
$('tube-plot').addEventListener('pointermove', event => { if (dragging) movePoint(event); });
$('tube-plot').addEventListener('pointerup', () => { dragging = false; });
$('tube-plot').addEventListener('pointercancel', () => { dragging = false; });

// Only one audio source plays at a time; all sounds require a visitor action.
for (const media of document.querySelectorAll('audio, video')) media.addEventListener('play', () => {
  stopAudio();
  for (const other of document.querySelectorAll('audio, video')) if (other !== media) other.pause();
});
document.addEventListener('visibilitychange', () => { if (document.hidden) { stopAudio(); if (state.running) cancelFit('Fitting paused while this page is in the background.'); } });
$('copy-citation').addEventListener('click', async () => {
  try {
    await navigator.clipboard.writeText($('bibtex').textContent);
    $('copy-citation').textContent = 'Copied'; $('copy-status').textContent = 'Citation copied to clipboard.';
    setTimeout(() => { $('copy-citation').textContent = 'Copy BibTeX'; }, 2000);
  } catch {
    const range = document.createRange(); range.selectNodeContents($('bibtex'));
    const selection = window.getSelection(); selection.removeAllRanges(); selection.addRange(range);
    $('copy-status').textContent = 'Citation selected. Copy it using your keyboard.';
  }
});
function updateReadingProgress() {
  const total = document.documentElement.scrollHeight - innerHeight;
  $('reading-progress').style.width = `${total > 0 ? scrollY / total * 100 : 0}%`;
}
addEventListener('scroll', updateReadingProgress, {passive: true});
addEventListener('resize', updateReadingProgress);
render();
try {
  state.sim = await createSimulator();
  targetDb = state.sim.response(state.target.areas, state.length).magnitudeDb;
  currentDb = state.sim.response(state.areas, state.length).magnitudeDb;
  $('engine-state').textContent = 'Rust / WASM · ready'; $('engine-state').classList.add('ready');
  $('lab').setAttribute('aria-busy', 'false');
  for (const button of $('lab').querySelectorAll('button')) button.disabled = false;
  render();
} catch (error) {
  $('engine-state').textContent = 'Simulator unavailable'; $('lab').setAttribute('aria-busy', 'false');
  setStatus(`${error.message} You can still explore the paper figures and recorded results below.`);
}
