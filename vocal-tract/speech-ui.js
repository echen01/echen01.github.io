import {estimatePitch} from './pitch/swift-f0.js';
import {initialAreas} from './simulator/index.js';

const $ = id => document.getElementById(id);
const NS = 'http://www.w3.org/2000/svg';
const ITERATIONS = 32;
const MAX_SECONDS = 10;
const MAX_BYTES = 20 * 1024 * 1024;
const SAMPLE_RATE = 16000;
const SAMPLE_URL = new URL('./assets/speech-original.wav', import.meta.url);
function node(tag, attributes = {}, text) {
  const value = document.createElementNS(NS, tag);
  for (const [key, attr] of Object.entries(attributes)) value.setAttribute(key, attr);
  if (text !== undefined) value.textContent = text;
  return value;
}

async function decode(blob) {
  if (blob.size > MAX_BYTES) throw new Error('Choose an audio file smaller than 20 MB.');
  const Context = window.OfflineAudioContext || window.webkitOfflineAudioContext;
  if (!Context) throw new Error('This browser cannot decode audio. Try a recent browser.');
  const context = new Context(1, 1, SAMPLE_RATE);
  let buffer;
  try { buffer = await context.decodeAudioData(await blob.arrayBuffer()); }
  catch { throw new Error('This recording could not be decoded. Try WAV, MP3, or M4A audio.'); }
  if (buffer.duration > MAX_SECONDS) throw new Error('Choose a clip of 10 seconds or less. The entire clip will be fitted.');
  if (buffer.duration < .2) throw new Error('Choose a recording at least 0.2 seconds long.');
  const samples = new Float32Array(buffer.length);
  for (let channel = 0; channel < buffer.numberOfChannels; channel++) {
    const source = buffer.getChannelData(channel);
    for (let i = 0; i < samples.length; i++) samples[i] += source[i] / buffer.numberOfChannels;
  }
  if (!samples.every(Number.isFinite)) throw new Error('The recording contains invalid samples.');
  return samples;
}

function wav(samples, sampleRate) {
  const bytes = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(bytes);
  const text = (offset, value) => { for (let i = 0; i < value.length; i++) view.setUint8(offset + i, value.charCodeAt(i)); };
  text(0, 'RIFF'); view.setUint32(4, 36 + samples.length * 2, true); text(8, 'WAVE');
  text(12, 'fmt '); view.setUint32(16, 16, true); view.setUint16(20, 1, true); view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true); view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true); view.setUint16(34, 16, true);
  text(36, 'data'); view.setUint32(40, samples.length * 2, true);
  for (let i = 0; i < samples.length; i++) view.setInt16(44 + i * 2, Math.round(Math.max(-1, Math.min(1, samples[i])) * 32767), true);
  return new Blob([bytes], {type: 'audio/wav'});
}

export function setupSpeechExperiment({onStart}) {
  const original = $('speech-original'), reconstructed = $('speech-reconstructed');
  const canvas = $('speech-map'), ctx = canvas.getContext('2d');
  const map = document.createElement('canvas'); map.width = canvas.width; map.height = canvas.height;
  const plot = map.getContext('2d');
  const neutral = initialAreas();
  let run = null, file = null, waveform = null, duration = 0, trajectory = null;
  let originalUrl = null, reconstructedUrl = null, selectedTime = 0, animation = 0, playing = null;
  const pendingSeeks = new Map();
  let lastFrame = -1, lastTrajectory = null;
  const status = text => { $('speech-status').textContent = text; };
  function busy(value, label = '') {
    $('speech-fit').disabled = value;
    $('speech-file').disabled = value;
    $('speech-example').disabled = value;
    $('speech-cancel').hidden = !value;
    $('speech-shape').setAttribute('aria-busy', String(value));
    $('speech-map').setAttribute('aria-busy', String(value));
    $('speech-engine').textContent = label || (value ? 'Fitting locally…' : 'Ready to fit');
    $('speech-engine').classList.toggle('ready', !value);
  }
  function cancel(message = 'Fitting stopped. You can inspect the latest area trajectory or start again.') {
    if (!run) return;
    const old = run; run = null;
    old.controller.abort(); old.worker?.terminate();
    busy(false, 'Stopped'); status(message);
  }
  function stopPlayback() {
    original.pause(); reconstructed.pause();
    playing = null; cancelAnimationFrame(animation); animation = 0;
  }
  function clearReconstruction() {
    reconstructed.pause(); reconstructed.removeAttribute('src'); reconstructed.load();
    reconstructed.hidden = true;
    if (reconstructedUrl) URL.revokeObjectURL(reconstructedUrl);
    reconstructedUrl = null;
    $('speech-audio-note').textContent = 'Available after fitting.';
  }
  function resetView() {
    pendingSeeks.clear();
    trajectory = null; waveform = null; duration = 0; selectedTime = 0;
    lastFrame = -1; lastTrajectory = null;
    $('speech-progress').value = 0;
    $('speech-step').textContent = `0 / ${ITERATIONS}`;
    $('speech-loss').textContent = '—'; $('speech-elapsed').textContent = '0.0 s';
    $('speech-scrub').disabled = true; $('speech-scrub').value = 0;
    clearReconstruction(); drawMap(); renderTime(0);
  }
  function setRecording(nextFile) {
    if (nextFile?.size > MAX_BYTES) { status('Choose an audio file smaller than 20 MB.'); return; }
    cancel(); stopPlayback(); file = nextFile;
    if (originalUrl) URL.revokeObjectURL(originalUrl);
    originalUrl = file ? URL.createObjectURL(file) : null;
    original.src = originalUrl || SAMPLE_URL.href;
    $('speech-caption').textContent = file ? file.name : 'Example · “On November 8, 1963”';
    $('speech-example').hidden = !file;
    $('speech-fit').textContent = file ? 'Fit entire recording →' : 'Fit entire speech sample →';
    busy(false); resetView();
    status('Ready. Fit the entire clip to estimate its moving area function.');
  }

  function drawShape(areas) {
    const svg = $('speech-shape');
    for (const child of [...svg.children]) if (!['title', 'desc'].includes(child.tagName)) child.remove();
    const x = i => 48 + i / 31 * 685;
    const y = area => 151 - Math.min(16, Math.max(0, area)) / 16 * 128;
    for (const value of [0, 4, 8, 12, 16]) {
      svg.append(node('line', {x1: 48, x2: 733, y1: y(value), y2: y(value), class: 'plot-grid'}));
      svg.append(node('text', {x: 35, y: y(value) + 4, 'text-anchor': 'end', class: 'plot-axis'}, value));
    }
    const path = Array.from(areas, (value, i) => `${i ? 'L' : 'M'}${x(i).toFixed(2)},${y(value).toFixed(2)}`).join(' ');
    svg.append(node('path', {d: `${path} L733,151 L48,151 Z`, fill: '#dcebe3'}));
    svg.append(node('path', {d: path, class: 'plot-current'}));
    svg.append(node('text', {x: 48, y: 181, class: 'plot-axis'}, 'Glottis'));
    svg.append(node('text', {x: 733, y: 181, class: 'plot-axis', 'text-anchor': 'end'}, 'Lips'));
    svg.append(node('text', {x: 390, y: 181, class: 'plot-axis', 'text-anchor': 'middle'}, 'Cross-sectional area · cm²'));
  }

  function drawMap() {
    plot.clearRect(0, 0, map.width, map.height);
    plot.fillStyle = '#f4f5ef'; plot.fillRect(54, 16, 684, 158);
    if (trajectory) {
      const frames = trajectory.frames;
      for (let frame = 0; frame < frames; frame++) {
        for (let area = 0; area < 32; area++) {
          const value = trajectory.areas[frame * 32 + area];
          const amount = Math.max(0, Math.min(1, Math.log(value / .12) / Math.log(16 / .12)));
          const quiet = trajectory.frameActive && !trajectory.frameActive[frame];
          const t = quiet ? amount * .25 : amount;
          plot.fillStyle = `rgb(${Math.round(242 - 222 * t)},${Math.round(238 - 127 * t)},${Math.round(229 - 125 * t)})`;
          plot.fillRect(54 + frame / frames * 684, 16 + (31 - area) / 32 * 158, Math.ceil(684 / frames), 5.1);
        }
      }
    } else {
      plot.fillStyle = '#7a827e'; plot.font = '14px system-ui, sans-serif'; plot.textAlign = 'center';
      plot.fillText(waveform ? 'Preparing the complete recording…' : 'Fit the recording to reveal its area trajectory.', 396, 98);
    }
    plot.font = '13px system-ui, sans-serif'; plot.fillStyle = '#626b6b'; plot.textAlign = 'right';
    plot.fillText('Lips', 45, 28); plot.fillText('Glottis', 45, 170);
    if (waveform) {
      let max = 0; for (const value of waveform) max = Math.max(max, Math.abs(value));
      plot.strokeStyle = '#92a7a0'; plot.lineWidth = 1;
      plot.beginPath();
      for (let pixel = 0; pixel < 684; pixel++) {
        const from = Math.floor(pixel / 684 * waveform.length), to = Math.floor((pixel + 1) / 684 * waveform.length);
        let peak = 0; for (let i = from; i < to; i++) peak = Math.max(peak, Math.abs(waveform[i]));
        const height = peak / Math.max(max, 1e-6) * 16;
        plot.moveTo(54 + pixel, 200 - height); plot.lineTo(54 + pixel, 200 + height);
      }
      plot.stroke();
    }
    plot.textAlign = 'left'; plot.fillText('0 s', 54, 237);
    plot.textAlign = 'right'; plot.fillText(`${duration.toFixed(2)} s`, 738, 237);
    drawCursor();
  }
  function drawCursor() {
    ctx.clearRect(0, 0, canvas.width, canvas.height); ctx.drawImage(map, 0, 0);
    if (duration > 0) {
      const x = 54 + selectedTime / duration * 684;
      ctx.strokeStyle = '#b96537'; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(x, 12); ctx.lineTo(x, 218); ctx.stroke();
    }
  }
  function frameAt(time) {
    const times = trajectory.frameTimes;
    let lo = 0, hi = times.length - 1;
    while (lo < hi) { const middle = Math.floor((lo + hi) / 2); if (times[middle] < time) lo = middle + 1; else hi = middle; }
    return lo > 0 && Math.abs(times[lo - 1] - time) < Math.abs(times[lo] - time) ? lo - 1 : lo;
  }
  function renderTime(time) {
    selectedTime = Math.min(duration, Math.max(0, time));
    $('speech-time').textContent = `${selectedTime.toFixed(2)} s`;
    $('speech-position').textContent = `${selectedTime.toFixed(2)} / ${duration.toFixed(2)} s`;
    $('speech-scrub').value = selectedTime;
    $('speech-scrub').setAttribute('aria-valuetext', `${selectedTime.toFixed(2)} of ${duration.toFixed(2)} seconds`);
    if (trajectory) {
      const frame = frameAt(selectedTime);
      if (frame !== lastFrame || trajectory !== lastTrajectory) {
        drawShape(trajectory.areas.subarray(frame * 32, (frame + 1) * 32));
        const active = !trajectory.frameActive || trajectory.frameActive[frame];
        const voiced = trajectory.frameVoiced?.[frame];
        $('speech-frame-label').textContent = `Frame ${frame + 1} / ${trajectory.frames} · ${!active ? 'Quiet · smoothed' : voiced ? 'Voiced' : 'Unvoiced'}`;
        const pitch = trajectory.framePitchHz?.[frame];
        $('speech-frame-pitch').textContent = active && voiced && Number.isFinite(pitch) ? `F₀ ${Math.round(pitch)} Hz` : 'No voiced pitch';
        lastFrame = frame; lastTrajectory = trajectory;
      }
    } else {
      drawShape(neutral);
      $('speech-frame-label').textContent = 'No fitted frames yet'; $('speech-frame-pitch').textContent = 'Pitch —';
    }
    drawCursor();
  }
  function tick() {
    if (!playing || playing.paused || playing.ended) { animation = 0; return; }
    if (playing.readyState > 0) renderTime(playing.currentTime);
    animation = requestAnimationFrame(tick);
  }
  function seek(time) {
    renderTime(time);
    for (const media of [original, reconstructed]) {
      if (media.readyState > 0) media.currentTime = Math.min(selectedTime, media.duration || selectedTime);
      else pendingSeeks.set(media, selectedTime);
    }
  }

  async function start() {
    cancel(); stopPlayback(); onStart?.();
    for (const media of document.querySelectorAll('audio, video')) media.pause();
    resetView();
    const current = {controller: new AbortController(), worker: null}; run = current;
    const valid = () => run === current && !current.controller.signal.aborted;
    busy(true, 'Reading audio…'); status('Decoding the entire recording…');
    const started = performance.now();
    try {
      let blob = file;
      if (!blob) {
        const response = await fetch(SAMPLE_URL, {signal: current.controller.signal});
        if (!response.ok) throw new Error('The example could not be loaded. Choose a local recording to continue.');
        blob = await response.blob();
      }
      if (!valid()) return;
      const decoded = await decode(blob);
      if (!valid()) return;
      waveform = decoded;
      duration = waveform.length / SAMPLE_RATE;
      $('speech-scrub').max = duration; $('speech-scrub').disabled = false;
      drawMap(); renderTime(0);
      busy(true, 'Tracking pitch…');
      status('SwiftF0 is estimating pitch and voicing across the full recording…');
      const pitch = await estimatePitch(waveform, {
        signal: current.controller.signal,
        onStage: stage => { if (valid() && stage === 'loading') status('Loading SwiftF0’s local model and WASM runtime…'); },
      });
      if (!valid()) return;
      busy(true, 'Backpropagating…');
      status('Fitting every active frame and smoothing the area trajectory across time…');
      const worker = new Worker(new URL('./speech/fit-worker.js', import.meta.url), {type: 'module'});
      current.worker = worker;
      worker.onerror = () => { if (valid()) cancel('The speech fitting worker could not run. Reload the page and try again.'); };
      worker.onmessage = ({data}) => {
        if (!valid()) return;
        if (data.type === 'error') { cancel(`Fitting stopped: ${data.message}`); return; }
        if (data.type === 'stage') { status(data.message || (data.stage === 'analyzing' ? 'Measuring the recording’s short-time spectra…' : 'Synthesizing the moving tract…')); return; }
        if (data.type !== 'progress' && data.type !== 'done') return;
        if (data.areas && data.frameTimes) trajectory = data;
        $('speech-step').textContent = `${data.iteration} / ${ITERATIONS}`;
        $('speech-progress').value = data.iteration;
        $('speech-loss').textContent = Number.isFinite(data.loss) ? data.loss.toFixed(4) : '—';
        $('speech-elapsed').textContent = `${((performance.now() - started) / 1000).toFixed(1)} s`;
        drawMap(); renderTime(selectedTime);
        if (data.type === 'progress') {
          status(`Pass ${data.iteration} / ${ITERATIONS} · ${data.frames} frames across ${duration.toFixed(2)} seconds. Play the original or scrub to inspect the current fit.`);
          return;
        }
        try {
          if (!data.samples?.length || !data.samples.every(Number.isFinite)) throw new Error('The reconstructed audio is invalid.');
          reconstructedUrl = URL.createObjectURL(wav(data.samples, data.sampleRate));
          reconstructed.src = reconstructedUrl;
          reconstructed.hidden = false;
          $('speech-audio-note').textContent = 'Generated here from the fitted areas, pitch, and excitation model.';
          const improvement = data.initialLoss > 0 ? 100 * (1 - data.loss / data.initialLoss) : 0;
          const change = improvement >= 0 ? `Objective decreased ${improvement.toFixed(1)}%.` : `Objective increased ${(-improvement).toFixed(1)}%.`;
          status(`Fitted all ${duration.toFixed(2)} seconds. ${change} Play either recording to watch the areas move, or scrub the timeline.`);
          worker.terminate(); run = null; busy(false, 'Fit complete');
          $('speech-fit').textContent = 'Fit again →';
        } catch (error) { cancel(error.message); }
      };
      worker.postMessage({type: 'start', samples: waveform.slice(), sampleRate: SAMPLE_RATE, pitch, iterations: ITERATIONS, length: 17.5});
    } catch (error) {
      if (valid()) cancel(error.message || 'This recording could not be fitted. Try another clip.');
    }
  }

  $('speech-fit').addEventListener('click', start);
  $('speech-cancel').addEventListener('click', () => cancel());
  $('speech-example').addEventListener('click', () => setRecording(null));
  $('speech-file').addEventListener('change', event => { const next = event.target.files?.[0]; event.target.value = ''; if (next) setRecording(next); });
  $('speech-scrub').addEventListener('input', event => seek(+event.target.value));
  let dragging = false;
  function seekMap(event) {
    if (!duration) return;
    const bounds = canvas.getBoundingClientRect();
    const x = (event.clientX - bounds.left) / bounds.width * canvas.width;
    seek((x - 54) / 684 * duration);
  }
  canvas.addEventListener('pointerdown', event => { if (event.button !== 0 || !duration) return; dragging = true; canvas.setPointerCapture(event.pointerId); seekMap(event); });
  canvas.addEventListener('pointermove', event => { if (dragging) seekMap(event); });
  canvas.addEventListener('pointerup', () => { dragging = false; });
  canvas.addEventListener('pointercancel', () => { dragging = false; });
  for (const media of [original, reconstructed]) {
    media.addEventListener('loadedmetadata', () => {
      if (pendingSeeks.has(media)) {
        media.currentTime = Math.min(pendingSeeks.get(media), media.duration);
        pendingSeeks.delete(media);
      }
    });
    media.addEventListener('play', () => { playing = media; cancelAnimationFrame(animation); tick(); });
    media.addEventListener('pause', () => { if (playing === media) { renderTime(media.currentTime); playing = null; cancelAnimationFrame(animation); animation = 0; } });
    media.addEventListener('seeked', () => { if (!playing || playing === media) renderTime(media.currentTime); });
    media.addEventListener('ended', () => { renderTime(duration); playing = null; cancelAnimationFrame(animation); animation = 0; });
  }
  document.addEventListener('visibilitychange', () => { if (document.hidden) { stopPlayback(); cancel('Fitting stopped while the page was in the background. Start again to finish the reconstruction.'); } });
  resetView();
  return {cancel};
}
