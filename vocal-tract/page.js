import {PRESETS, initialAreas, FREQUENCIES} from './simulator/index.js?v=vowel-presets-e-1';
import {createWebGpuSimulator} from './simulator/webgpu.js?v=vowel-mfn-e-1';
import {setupPitchEstimator} from './pitch-ui.js';
import {setupSpeechExperiment} from './speech-ui.js?v=vowel-tubes-1';
import {renderTractView} from './speech/tract-view.js?v=vowel-ground-truth-1';

const $ = id => document.getElementById(id);
const audioLabels = new Map(['listen-current', 'listen-target'].map(id => [id, $(id).textContent.trim()]));
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
let targetDb = [], currentDb = [], renderPending = false;
let tubePlot, responseVersion = 0, responseTask = null, backendVersion = 0;

function setStatus(message) { $('fit-status').textContent = message; }
function syncControls() {
  $('section').value = state.selected + 1;
  $('section-value').textContent = `${state.selected + 1} / 32`;
  $('area').value = state.areas[state.selected];
  $('area-value').textContent = `${state.areas[state.selected].toFixed(2)} cm²`;
  $('tract-length').value = state.length;
  $('length-value').textContent = `${state.length} cm`;
  $('pitch').value = state.pitch;
  $('pitch-value').textContent = `${state.pitch} Hz`;
}
function drawTube() {
  ({plot: tubePlot} = renderTractView($('tube-plot'), state.areas, {
    lengthCm: state.length, radiusLimitCm: 2.5,
    targetAreas: $('show-target').checked ? state.target.areas : undefined,
    selectedIndex: state.selected,
  }));
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
async function responseUpdated() {
  responseVersion++; queueRender();
  const sim = state.sim, generation = backendVersion;
  if (!sim || responseTask?.sim === sim || state.running) return;
  const task = {sim}; responseTask = task;
  try {
    let version;
    do {
      version = responseVersion;
      const areas = [...state.areas], target = state.target, length = state.length;
      const current = await sim.response(areas, length);
      if (generation !== backendVersion) return false;
      const reference = await sim.response(target.areas, length);
      if (generation !== backendVersion) return false;
      if (version === responseVersion && !state.running) {
        currentDb = current.magnitudeDb; targetDb = reference.magnitudeDb; queueRender();
      }
    } while (version !== responseVersion && !state.running);
    return true;
  } catch (error) { backendUnavailable(error.message, generation); return false; }
  finally { if (responseTask === task) responseTask = null; }
}
function backendUnavailable(message, generation) {
  if (generation !== backendVersion) return;
  backendVersion++; responseVersion++;
  const sim = state.sim; state.sim = null; sim?.dispose?.();
  stopAudio(); cancelFit();
  $('engine-state').textContent = 'WebGPU unavailable'; $('engine-state').classList.remove('ready');
  $('lab').setAttribute('aria-busy', 'false');
  for (const id of ['fit', 'listen-current', 'listen-target']) $(id).disabled = true;
  setStatus(message);
}
async function initializeSimulator() {
  const generation = ++backendVersion;
  stopAudio(); cancelFit(); clearFit();
  const previous = state.sim; state.sim = null;
  responseVersion++; previous?.dispose?.();
  $('lab').setAttribute('aria-busy', 'true');
  $('engine-state').textContent = 'Loading WebGPU…'; $('engine-state').classList.remove('ready');
  for (const id of ['fit', 'listen-current', 'listen-target']) $(id).disabled = true;
  for (const button of $('lab').querySelectorAll('.target-option, #reset')) button.disabled = false;
  setStatus('Preparing WebGPU…'); queueRender();
  try {
    const sim = await createWebGpuSimulator({onDeviceLost: message => backendUnavailable(message, generation)});
    if (generation !== backendVersion) { sim.dispose?.(); return; }
    state.sim = sim;
    if (!await responseUpdated() || generation !== backendVersion) return;
    $('engine-state').textContent = 'WebGPU · ready'; $('engine-state').classList.add('ready');
    $('lab').setAttribute('aria-busy', 'false');
    for (const id of ['fit', 'listen-current', 'listen-target']) $(id).disabled = false;
    setStatus('MFN field ready on WebGPU. Fit the current shape, or Reset to start from a neutral tube.');
  } catch (error) { backendUnavailable(error.message, generation); }
}
function cancelFit(message) {
  if (state.worker) {
    const worker = state.worker; state.worker = null;
    worker.postMessage({type: 'stop'});
    // Allow the worker's finally block to destroy its GPU buffers/device.
    setTimeout(() => worker.terminate(), 2000);
  }
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
  if (state.audioButton) { state.audioButton.textContent = audioLabels.get(state.audioButton.id); state.audioButton.removeAttribute('aria-pressed'); state.audioButton = null; }
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
    const {samples, sampleRate} = await state.sim.synthesize(which === 'target' ? state.target.areas : state.areas, state.length, state.pitch, 1.2);
    if (request !== state.audioRequest) return;
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
  speechExperiment.cancel('Speech fitting stopped to run the single-vowel experiment.');
  if (state.running) { cancelFit('Fitting paused. The current shape is preserved. Fit again to continue with a fresh optimizer.'); return; }
  stopAudio(); cancelFit(); clearFit();
  state.running = true; responseVersion++; $('fit').textContent = 'Pause fitting';
  setStatus('Preparing the MFN field on WebGPU…');
  let worker;
  try { worker = new Worker(new URL('./simulator/fit-worker.js?v=vowel-mfn-e-1', import.meta.url), {type: 'module'}); }
  catch { cancelFit('This browser could not start the fitting worker. You can still reshape the tube and listen.'); return; }
  state.worker = worker;
  worker.onmessage = ({data}) => {
    if (state.worker !== worker) return;
    if (data.type === 'error') { cancelFit(`Fitting stopped: ${data.message}`); return; }
    if (data.type !== 'progress' && data.type !== 'done') return;
    state.areas = data.areas; state.iteration = data.iteration; state.history = data.history; state.elapsedMs = data.elapsedMs;
    currentDb = data.magnitudeDb; targetDb = data.targetDb; queueRender();
    if (data.type === 'progress') setStatus('Fitting the MFN neural field with WebGPU…');
    if (data.type === 'done') {
      const first = data.history[0];
      const reduction = first > 0 ? 100 * (1 - data.loss / first) : 0;
      let change = 'Spectral loss changed by less than 0.1%.';
      if (first === 0 && data.loss > 0) change = 'Spectral loss increased from zero. Reset the tube to begin a new experiment.';
      else if (reduction >= .05) change = `Spectral loss decreased by ${reduction >= 99.9 ? '>99.9' : reduction.toFixed(1)}%.`;
      else if (reduction <= -.05) change = `Spectral loss increased by ${(-reduction).toFixed(1)}%. Try resetting the tube before fitting again.`;
      cancelFit(`Finished ${data.iteration} updates. ${change} ${$('show-target').checked ? 'Compare the fit with the ground-truth outline.' : 'Enable Ground truth to compare the shapes.'}`);
    }
  };
  worker.onerror = () => { if (state.worker === worker) cancelFit('The fitting worker could not run. Reload the page to try again.'); };
  worker.postMessage({type: 'start', backend: 'webgpu', areas: [...state.areas], targetAreas: state.target.areas, length: state.length, iterations: 240, learningRate: .005});
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
  setStatus('Tract length changes both the current and target resonances.'); responseUpdated();
});
const pitchEstimator = setupPitchEstimator({
  onPitch: pitch => { stopAudio(); state.pitch = pitch; syncControls(); },
});
const speechExperiment = setupSpeechExperiment({
  parameterization: 'faithful-rff',
  iterations: 100,
  onStart: () => { stopAudio(); cancelFit(); pitchEstimator.cancel('Pitch analysis stopped to fit the full recording.'); },
});
$('pitch').addEventListener('input', event => { pitchEstimator.cancel('Analysis canceled because the source pitch was adjusted.'); stopAudio(); state.pitch = +event.target.value; syncControls(); });
$('show-target').addEventListener('change', drawTube);
$('listen-current').addEventListener('click', () => playSound('current'));
$('listen-target').addEventListener('click', () => playSound('target'));
$('fit').addEventListener('click', startFit);
$('reset').addEventListener('click', () => { stopAudio(); cancelFit(); clearFit(); state.areas = initialAreas(); $('fit').textContent = 'Fit to target →'; setStatus('Reset to a neutral tube. Choose a target and start fitting.'); responseUpdated(); });

let dragging = false;
function tubePoint(event) {
  return new DOMPoint(event.clientX, event.clientY).matrixTransform($('tube-plot').getScreenCTM().inverse());
}
function movePoint(point) {
  state.selected = tubePlot.sectionAt(point.x);
  state.areas[state.selected] = Math.min(16, Math.max(.12, tubePlot.areaAt(point.y)));
  editAreas();
}
$('tube-plot').addEventListener('pointerdown', event => {
  if (!state.sim || event.button !== 0) return;
  const point = tubePoint(event);
  if (!tubePlot.contains(point.x, point.y)) return;
  dragging = true; $('tube-plot').setPointerCapture(event.pointerId); movePoint(point);
});
$('tube-plot').addEventListener('pointermove', event => { if (dragging) movePoint(tubePoint(event)); });
$('tube-plot').addEventListener('pointerup', () => { dragging = false; });
$('tube-plot').addEventListener('pointercancel', () => { dragging = false; });

// Only one audio source plays at a time; all sounds require a visitor action.
for (const media of document.querySelectorAll('audio, video')) media.addEventListener('play', () => {
  stopAudio();
  for (const other of document.querySelectorAll('audio, video')) if (other !== media) other.pause();
});
document.addEventListener('visibilitychange', () => { if (document.hidden) { stopAudio(); pitchEstimator.cancel('Pitch analysis paused. Analyze the recording again to continue.'); if (state.running) cancelFit('Fitting paused while this page is in the background.'); } });
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
addEventListener('pagehide', () => { backendVersion++; stopAudio(); cancelFit(); state.sim?.dispose?.(); state.sim = null; });
await initializeSimulator();
