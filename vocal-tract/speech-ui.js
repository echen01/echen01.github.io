import {estimatePitch} from './pitch/swift-f0.js';
import {initialAreas} from './simulator/index.js';
import {evaluateSerializedField} from './speech/rff-field.js';
import {createSpectrogramComparison} from './speech/spectrogram.js';
import {renderTractView} from './speech/tract-view.js?v=vowel-tubes-1';
import {createMicrophoneRecorder} from './speech/microphone-recorder.js?v=recording-1';

const $ = id => document.getElementById(id);
const NS = 'http://www.w3.org/2000/svg';
const MAX_SECONDS = 10;
const MAX_BYTES = 20 * 1024 * 1024;
const SAMPLE_RATE = 16000;
const MFN_AREA_SCALE = 1;
const SAMPLE_URL = new URL('./assets/speech-original.wav', import.meta.url);
function node(tag, attributes = {}, text) {
  const value = document.createElementNS(NS, tag);
  for (const [key, attr] of Object.entries(attributes)) value.setAttribute(key, attr);
  if (text !== undefined) value.textContent = text;
  return value;
}

async function decode(blob, sampleRate = SAMPLE_RATE, {trimToLimit = false} = {}) {
  if (blob.size > MAX_BYTES) throw new Error('Choose an audio file smaller than 20 MB.');
  const Context = window.OfflineAudioContext || window.webkitOfflineAudioContext;
  if (!Context) throw new Error('This browser cannot decode audio. Try a recent browser.');
  const context = new Context(1, 1, sampleRate);
  let buffer;
  try { buffer = await context.decodeAudioData(await blob.arrayBuffer()); }
  catch { throw new Error('This recording could not be decoded. Try WAV, MP3, or M4A audio.'); }
  if (buffer.duration > MAX_SECONDS && !trimToLimit) throw new Error('Choose a clip of 10 seconds or less. The entire clip will be fitted.');
  if (buffer.duration < .2) throw new Error('Choose a recording at least 0.2 seconds long.');
  // Recorder containers can include a short codec tail beyond the stop time.
  const samples = new Float32Array(trimToLimit ? Math.min(buffer.length, MAX_SECONDS * sampleRate) : buffer.length);
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

export function setupSpeechExperiment({onStart, parameterization = 'discrete', iterations = 32, createRecorder = createMicrophoneRecorder} = {}) {
  let ITERATIONS = iterations;
  const faithful = parameterization === 'faithful-rff';
  const isRff = parameterization === 'rff' || faithful;
  const fieldType = () => faithful ? $('speech-field-type')?.value ?? 'mfn' : 'rff';
  const fieldLabel = () => fieldType().toUpperCase();
  const mfnInputScale = () => Number($('speech-mfn-scale')?.value ?? 64);
  let rffWidth = $('speech-hidden-dim')?.value ?? '256';
  const sampleRate = SAMPLE_RATE;
  function updateArchitecture() {
    if (!faithful || !$('speech-hidden-dim')) return;
    const width = Number($('speech-hidden-dim').value);
    const mfn = fieldType() === 'mfn';
    $('speech-architecture').textContent = mfn ? 'Four learned sine filters × three 256-wide affine layers → 1'
      : `512 Fourier features → ${width} → ${width} → ${width} → 1`;
    $('speech-parameter-count').textContent = (mfn ? 200705 : 512 * width + 2 * width * width + 4 * width + 1).toLocaleString('en-US');
    if ($('speech-architecture-kind')) $('speech-architecture-kind').textContent = mfn ? 'Multiplicative filter network' : 'Three tanh hidden layers';
    if ($('speech-area-scale')) $('speech-area-scale').textContent = String(mfn ? MFN_AREA_SCALE : .35);
    if ($('speech-architecture-settings')) $('speech-architecture-settings').textContent = mfn
      ? `FourierNet initialization · learned filters · input scale ${mfnInputScale()} · peak learning rate 0.005`
      : 'NNX initialization at the selected width · frozen Fourier bandwidths (10, 2) · peak learning rate 0.001';
    if ($('speech-width-help')) $('speech-width-help').textContent = mfn
      ? 'MFN uses learned sine filters to represent the moving tract.'
      : 'Smaller RFF networks take faster steps but may need more updates.';
    if ($('speech-hidden-dim-label')) $('speech-hidden-dim-label').hidden = mfn;
    if ($('speech-mfn-scale-label')) $('speech-mfn-scale-label').hidden = !mfn;
    if ($('speech-mfn-scale-help')) $('speech-mfn-scale-help').hidden = !mfn;
    if ($('speech-field-label')) $('speech-field-label').textContent = 'Speech playground';
  }
  $('speech-progress').max = ITERATIONS;
  const original = $('speech-original'), reconstructed = $('speech-reconstructed');
  const spectrograms = $('speech-target-spectrogram') ? createSpectrogramComparison({
    targetCanvas: $('speech-target-spectrogram'), reconstructedCanvas: $('speech-reconstructed-spectrogram'),
    status: $('speech-spectrogram-status'),
    trainingControls: $('speech-spectrogram-training'), stepSlider: $('speech-spectrogram-step'),
    stepLabel: $('speech-spectrogram-step-label'), followLatest: $('speech-spectrogram-follow'),
  }) : null;
  const canvas = $('speech-map'), ctx = canvas.getContext('2d');
  const map = document.createElement('canvas'); map.width = canvas.width; map.height = canvas.height;
  const plot = map.getContext('2d');
  const neutral = initialAreas();
  const radiusExtent = areas => {
    let maximum = 0;
    for (const area of areas) maximum = Math.max(maximum, Math.sqrt(Math.max(0, area) / Math.PI));
    return Math.max(.1, maximum * 1.15);
  };
  let tubeRadiusLimit = radiusExtent(neutral);
  let run = null, file = null, waveform = null, duration = 0, trajectory = null;
  let stopping = Promise.resolve();
  let originalUrl = null, reconstructedUrl = null, selectedTime = 0, animation = 0, playing = null;
  let fittingBusy = false, captureState = 'idle', preparingCapture = false, captureGeneration = 0, microphoneFile = false;
  const recordingPreview = $('speech-recording-preview');
  const captureBusy = () => captureState !== 'idle' || preparingCapture;
  const recordingStatus = text => { if ($('speech-record-status')) $('speech-record-status').textContent = text; };
  const recorder = $('speech-record') ? createRecorder({
    maxSeconds: MAX_SECONDS, maxBytes: MAX_BYTES,
    onState(state) {
      captureState = state;
      if (state === 'requesting') recordingStatus('Allow microphone access to start recording.');
      if (state === 'recording') recordingStatus('Recording… Speak naturally, then stop when you are done.');
      if (state === 'stopping') recordingStatus('Finishing the recording…');
      syncInputControls();
    },
    onTick(seconds) { $('speech-record-time').textContent = `${seconds.toFixed(1)} / ${MAX_SECONDS} s`; },
    onComplete: prepareRecording,
    onError(message) { recordingStatus(message); syncInputControls(); },
  }) : null;
  const pendingSeeks = new Map();
  let lastFrame = -1, lastTrajectory = null;
  const status = text => { $('speech-status').textContent = text; };
  function syncInputControls() {
    const locked = fittingBusy || captureBusy();
    for (const id of ['speech-fit', 'speech-file', 'speech-example', 'speech-backend', 'speech-field-type', 'speech-mfn-scale', 'speech-iterations']) {
      if ($(id)) $(id).disabled = locked;
    }
    if ($('speech-hidden-dim')) $('speech-hidden-dim').disabled = locked || fieldType() === 'mfn';
    if ($('speech-record')) {
      $('speech-record').disabled = locked || !recorder?.supported;
      $('speech-record').textContent = microphoneFile ? 'Record again' : 'Record audio';
      $('speech-record-stop').hidden = captureState !== 'recording';
      $('speech-record-cancel').hidden = !captureBusy();
      $('speech-record-time').hidden = !captureBusy();
      $('speech-recording').classList.toggle('is-recording', captureState === 'recording');
    }
    if (recordingPreview) recordingPreview.hidden = !microphoneFile || captureBusy();
  }
  function busy(value, label = '') {
    fittingBusy = value;
    syncInputControls();
    $('speech-cancel').hidden = !value;
    $('speech-shape').setAttribute('aria-busy', String(value));
    $('speech-map').setAttribute('aria-busy', String(value));
    $('speech-engine').textContent = label || (value ? 'Fitting locally…' : 'Ready to fit');
    $('speech-engine').classList.toggle('ready', !value);
  }
  function cancel(message = 'Fitting stopped. You can inspect the latest area trajectory or start again.') {
    if (!run) return;
    const old = run; run = null;
    old.controller.abort(); stopping = Promise.all([stopping, releaseWorker(old, true)]);
    spectrograms?.stopTraining();
    busy(false, 'Stopped'); status(message);
  }
  function releaseWorker(current, graceful = false) {
    if (!current.worker) return Promise.resolve();
    if (current.releasing) return current.releasing;
    const worker = current.worker;
    current.releasing = new Promise(resolve => {
      let timeout;
      const finish = () => {
        clearTimeout(timeout);
        worker.removeEventListener('message', terminal);
        worker.removeEventListener('error', finish);
        worker.terminate(); resolve();
      };
      const terminal = ({data}) => { if (['stopped', 'done', 'error'].includes(data.type)) finish(); };
      if (!faithful || !graceful) { finish(); return; }
      // Let the worker release GPU buffers/device before termination. Bound the
      // wait in case the browser driver or worker cannot respond to cancellation.
      worker.addEventListener('message', terminal);
      worker.addEventListener('error', finish);
      timeout = setTimeout(finish, 2000);
      worker.postMessage({type: 'stop'});
    });
    return current.releasing;
  }
  function stopPlayback() {
    original.pause(); reconstructed.pause(); recordingPreview?.pause();
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
    tubeRadiusLimit = radiusExtent(neutral);
    lastFrame = -1; lastTrajectory = null;
    $('speech-progress').value = 0;
    $('speech-step').textContent = `0 / ${ITERATIONS}`;
    $('speech-loss').textContent = '—'; $('speech-elapsed').textContent = '0.0 s';
    if ($('speech-speed')) $('speech-speed').textContent = '—';
    if ($('speech-backend-note')) $('speech-backend-note').textContent = '';
    $('speech-scrub').disabled = true; $('speech-scrub').value = 0;
    clearReconstruction(); spectrograms?.reset(); drawMap(); drawLoss([]); renderTime(0);
  }
  function setRecording(nextFile, {microphone = false} = {}) {
    if (nextFile?.size > MAX_BYTES) { status('Choose an audio file smaller than 20 MB.'); return; }
    cancel(); stopPlayback(); file = nextFile; microphoneFile = microphone;
    if (originalUrl) URL.revokeObjectURL(originalUrl);
    originalUrl = file ? URL.createObjectURL(file) : null;
    original.src = originalUrl || SAMPLE_URL.href;
    if (recordingPreview) {
      if (microphoneFile) recordingPreview.src = originalUrl;
      else { recordingPreview.removeAttribute('src'); recordingPreview.load(); }
    }
    $('speech-caption').textContent = file ? file.name : 'Example · “On November 8, 1963”';
    $('speech-example').hidden = !file;
    $('speech-fit').textContent = 'Fit audio →';
    busy(false); resetView();
    status('Ready. Fit the entire clip to estimate its moving area function.');
  }

  async function prepareRecording(blob) {
    const generation = captureGeneration;
    preparingCapture = true; syncInputControls();
    recordingStatus('Preparing the recording for playback…');
    try {
      const samples = await decode(blob, sampleRate, {trimToLimit: true});
      if (generation !== captureGeneration) return;
      if (!samples.some(value => Math.abs(value) > 1e-6)) throw new Error('No audio was captured. Check your microphone and record again.');
      const seconds = samples.length / sampleRate;
      setRecording(new File([wav(samples, sampleRate)], `Microphone recording · ${seconds.toFixed(1)} s.wav`, {type: 'audio/wav'}), {microphone: true});
      recordingStatus(`Recorded ${seconds.toFixed(1)} seconds. Listen below, then click Fit when ready.`);
      status('Your microphone recording is selected. Click Fit to estimate its moving area function.');
    } catch (error) {
      if (generation === captureGeneration) recordingStatus(error.message || 'The recording could not be prepared. Try again.');
    } finally {
      if (generation === captureGeneration) { preparingCapture = false; syncInputControls(); }
    }
  }
  function cancelRecording(message = 'Recording discarded. Your previous audio is still selected.') {
    if (!captureBusy()) return;
    captureGeneration++; preparingCapture = false;
    recorder?.cancel(); captureState = 'idle';
    recordingStatus(message); syncInputControls();
  }

  function drawShape(areas) {
    // Use the entire trajectory's extent, as in diffTLM/viz.py, so the tube
    // doesn't rescale at every frame. Allow an off-grid field peak extra room.
    const currentPeak = Math.max(...Array.from(areas, area => Math.sqrt(Math.max(0, area) / Math.PI)));
    if (currentPeak > tubeRadiusLimit) tubeRadiusLimit = currentPeak * 1.15;
    renderTractView($('speech-shape'), areas, {
      lengthCm: trajectory?.length ?? 17.5, radiusLimitCm: tubeRadiusLimit, faithful,
    });
  }

  function drawLoss(history) {
    const svg = $('speech-loss-plot');
    if (!svg) return;
    svg.replaceChildren(node('title', {}, 'Fitting objective at each optimization step'));
    if (!history.length) return;
    const maximum = Math.max(...history, 1e-6) * 1.05;
    const x = i => 48 + i / ITERATIONS * 685, y = loss => 96 - loss / maximum * 78;
    for (const value of [0, maximum / 2, maximum]) {
      svg.append(node('line', {x1:48, x2:733, y1:y(value), y2:y(value), class:'plot-grid'}));
      svg.append(node('text', {x:38, y:y(value)+4, 'text-anchor':'end', class:'plot-axis'}, value.toFixed(2)));
    }
    const path = history.map((value, i) => `${i ? 'L' : 'M'}${x(i).toFixed(2)},${y(value).toFixed(2)}`).join(' ');
    svg.append(node('path', {d:path, class:'plot-current'}));
    svg.append(node('text', {x:48, y:120, class:'plot-axis'}, '0'));
    svg.append(node('text', {x:733, y:120, 'text-anchor':'end', class:'plot-axis'}, `${ITERATIONS} steps`));
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
      // Query the learned function at the actual playhead time, including times
      // between acoustic training frames. This is not interpolation of free areas.
      if (trajectory.field && !faithful) drawShape(evaluateSerializedField(trajectory.field, selectedTime));
      if (frame !== lastFrame || trajectory !== lastTrajectory) {
        if (!trajectory.field || faithful) drawShape(trajectory.areas.subarray(frame * 32, (frame + 1) * 32));
        const active = !trajectory.frameActive || trajectory.frameActive[frame];
        const voiced = trajectory.frameVoiced?.[frame];
        $('speech-frame-label').textContent = `${trajectory.field ? 'Nearest frame' : 'Frame'} ${frame + 1} / ${trajectory.frames} · ${!active ? 'Quiet' : voiced ? 'Voiced' : 'Unvoiced'}`;
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
    if (captureBusy()) return;
    cancel(); stopPlayback(); onStart?.();
    updateArchitecture();
    ITERATIONS = Math.max(1, Math.min(500, Math.floor(Number($('speech-iterations')?.value || iterations))));
    $('speech-progress').max = ITERATIONS;
    for (const media of document.querySelectorAll('audio, video')) media.pause();
    resetView();
    const current = {controller: new AbortController(), worker: null}; run = current;
    const valid = () => run === current && !current.controller.signal.aborted;
    busy(true, 'Reading audio…'); status('Decoding the entire recording…');
    const started = performance.now();
    try {
      await stopping;
      if (!valid()) return;
      let blob = file, decoded, pitch, uin;
      if (faithful && !file) {
        const {loadFaithfulExample} = await import('./speech/faithful-engine.js');
        const example = await loadFaithfulExample({sampleRate});
        decoded = example.target; uin = example.uin;
        pitch = {pitchHz:example.metadata.pitchHz, confidence:example.metadata.confidence};
      } else if (!blob) {
        const response = await fetch(SAMPLE_URL, {signal: current.controller.signal});
        if (!response.ok) throw new Error('The example could not be loaded. Choose a local recording to continue.');
        blob = await response.blob();
      }
      if (!valid()) return;
      if (!decoded) decoded = await decode(blob, sampleRate);
      if (faithful && file) {
        const {normalizeFaithfulTarget} = await import('./speech/faithful-source.js');
        decoded = normalizeFaithfulTarget(decoded);
      }
      if (!valid()) return;
      waveform = decoded;
      duration = waveform.length / sampleRate;
      spectrograms?.setTarget(waveform, sampleRate);
      $('speech-scrub').max = duration; $('speech-scrub').disabled = false;
      drawMap(); renderTime(0);
      busy(true, 'Tracking pitch…');
      status('SwiftF0 is estimating pitch and voicing across the full recording…');
      // Audio decoding and fitting both use SwiftF0's native 16 kHz rate.
      if (!pitch) pitch = await estimatePitch(waveform, {
        signal: current.controller.signal,
        ...(faithful ? {fmin:65,fmax:500} : {}),
        onStage: stage => { if (valid() && stage === 'loading') status('Loading SwiftF0’s local model and WASM runtime…'); },
      });
      if (!valid()) return;
      busy(true, 'Backpropagating…');
      status(isRff ? `Training shared ${fieldLabel()} network weights through the acoustic simulator…` : 'Fitting every active frame and smoothing the area trajectory across time…');
      const worker = new Worker(new URL(faithful ? './speech/faithful-fit-worker.js?v=playground-simple-1' : './speech/fit-worker.js', import.meta.url), {type: 'module'});
      current.worker = worker;
      worker.onerror = () => { if (valid()) cancel('The speech fitting worker could not run. Reload the page and try again.'); };
      worker.onmessage = ({data}) => {
        if (!valid()) return;
        if (data.type === 'error') { releaseWorker(current); cancel(`Fitting stopped: ${data.message}`); return; }
        if (data.type === 'stopped') { releaseWorker(current); cancel(data.message || 'Fitting stopped.'); return; }
        if (data.type === 'stage') {
          if (data.restart) {
            current.receivedSpectrogram = false;
            spectrograms?.clearReconstruction('Restarting the fit after a GPU reset…');
          }
          status(data.message || (data.stage === 'analyzing' ? 'Measuring the recording’s short-time spectra…' : 'Synthesizing the moving tract…')); return;
        }
        if (data.type === 'spectrogram') {
          spectrograms?.addCheckpoint(data.spectrum, {iteration: data.iteration, iterations: data.iterations});
          current.receivedSpectrogram = true;
          return;
        }
        if (data.type !== 'progress' && data.type !== 'done') return;
        if (data.areas && data.frameTimes) { trajectory = data; tubeRadiusLimit = radiusExtent(data.areas); }
        $('speech-step').textContent = `${data.iteration} / ${ITERATIONS}`;
        $('speech-progress').value = data.iteration;
        $('speech-loss').textContent = Number.isFinite(data.loss) ? data.loss.toFixed(4) : '—';
        $('speech-elapsed').textContent = `${((performance.now() - started) / 1000).toFixed(1)} s`;
        if ($('speech-speed')) $('speech-speed').textContent = data.iterationsPerSecond > 0 ? `${data.iterationsPerSecond.toFixed(1)} it/s` : '—';
        if ($('speech-backend-note') && data.backend) {
          $('speech-backend-note').textContent = data.backend === 'webgpu-f32'
            ? 'WebGPU float32 field · Rust/WASM float64 acoustics'
            : data.backend === 'faithful-full-gpu' ? 'WebGPU'
            : data.backend === 'faithful-cpu' ? 'CPU'
            : data.backend === 'webgpu-full-f32' ? 'WebGPU'
            : `CPU float64 field · Rust/WASM float64 acoustics${data.fallbackReason ? ` · ${data.fallbackReason}` : ''}`;
        }
        drawMap(); drawLoss(data.history || []); renderTime(selectedTime);
        if (data.type === 'progress') {
          status(`${isRff ? `${fieldLabel()} step` : 'Pass'} ${data.iteration} / ${ITERATIONS} · ${data.frames} frames across ${duration.toFixed(2)} seconds. Scrub the timeline to inspect the current fit.`);
          return;
        }
        try {
          if (!data.samples?.length || !data.samples.every(Number.isFinite)) throw new Error('The reconstructed audio is invalid.');
          if (!current.receivedSpectrogram) spectrograms?.setReconstruction(data.samples, {sampleRate: data.sampleRate,
            normalizationGain: faithful ? data.normalizationGain ?? 1 : 1});
          spectrograms?.completeTraining();
          reconstructedUrl = URL.createObjectURL(wav(data.samples, data.sampleRate));
          reconstructed.src = reconstructedUrl;
          reconstructed.hidden = false;
          $('speech-audio-note').textContent = faithful ? 'Original waveform synthesis with key(0); normalized for playback after fitting.' : 'Generated here from the fitted areas, pitch, and excitation model.';
          const improvement = data.initialLoss > 0 ? 100 * (1 - data.loss / data.initialLoss) : 0;
          const change = improvement >= 0 ? `Objective decreased ${improvement.toFixed(1)}%.` : `Objective increased ${(-improvement).toFixed(1)}%.`;
          status(`Fitted all ${duration.toFixed(2)} seconds. ${data.gpuRestarts ? 'Recovered after a GPU reset. ' : ''}${change} Play the reconstruction to watch the areas move, or scrub the timeline.`);
          releaseWorker(current); run = null; busy(false, 'Fit complete');
          $('speech-fit').textContent = 'Fit again →';
        } catch (error) { cancel(error.message); }
      };
      worker.postMessage({type: 'start', samples: waveform.slice(), sampleRate, pitch, uin, iterations: ITERATIONS, length: 17.5, parameterization,
        spectrogramPreviews: faithful && Boolean(spectrograms),
        fieldType: fieldType(), mfnInputScale: mfnInputScale(), mfnAreaScale: MFN_AREA_SCALE, hiddenDim: Number($('speech-hidden-dim')?.value ?? 512),
        backend: faithful ? 'fullgpu' : $('speech-backend')?.value || 'auto'});
    } catch (error) {
      if (valid()) cancel(error.message || 'This recording could not be fitted. Try another clip.');
    }
  }

  $('speech-fit').addEventListener('click', start);
  $('speech-field-type')?.addEventListener('change', () => {
    const width = $('speech-hidden-dim');
    if (fieldType() === 'mfn') { rffWidth = width.value; width.value = '256'; }
    else width.value = rffWidth;
    stopPlayback(); resetView(); updateArchitecture(); busy(false);
    $('speech-fit').textContent = 'Fit audio →';
    status(`Ready to fit with the ${fieldLabel()} neural field.`);
  });
  $('speech-hidden-dim')?.addEventListener('change', () => {
    stopPlayback(); resetView(); updateArchitecture(); busy(false);
    $('speech-fit').textContent = 'Fit audio →';
    status('Ready to fit with the selected network width.');
  });
  $('speech-mfn-scale')?.addEventListener('change', () => {
    stopPlayback(); resetView(); updateArchitecture(); busy(false);
    $('speech-fit').textContent = 'Fit audio →';
    status(`Ready to fit MFN with input scale ${mfnInputScale()}.`);
  });

  $('speech-cancel').addEventListener('click', () => cancel());
  $('speech-example').addEventListener('click', () => { cancelRecording(); setRecording(null); recordingStatus('Record up to 10 seconds, then listen before fitting.'); });
  $('speech-file').addEventListener('change', event => { const next = event.target.files?.[0]; event.target.value = ''; if (next) { cancelRecording(); setRecording(next); recordingStatus('Record up to 10 seconds, then listen before fitting.'); } });
  $('speech-record')?.addEventListener('click', () => {
    if (fittingBusy || captureBusy() || !recorder?.supported) return;
    captureGeneration++; stopPlayback(); onStart?.();
    for (const media of document.querySelectorAll('audio, video')) media.pause();
    $('speech-record-time').textContent = `0.0 / ${MAX_SECONDS} s`;
    recorder.start();
  });
  $('speech-record-stop')?.addEventListener('click', () => recorder.stop());
  $('speech-record-cancel')?.addEventListener('click', () => cancelRecording());
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
  const playbackSources = [original, reconstructed, recordingPreview].filter(Boolean);
  for (const media of playbackSources) {
    media.addEventListener('loadedmetadata', () => {
      if (pendingSeeks.has(media)) {
        media.currentTime = Math.min(pendingSeeks.get(media), media.duration);
        pendingSeeks.delete(media);
      }
    });
    media.addEventListener('play', () => {
      for (const other of playbackSources) if (other !== media) other.pause();
      if (captureBusy()) { media.pause(); return; }
      // The fitted tube belongs to the reconstruction; the original is audio only.
      playing = media === reconstructed ? reconstructed : null;
      cancelAnimationFrame(animation); animation = 0;
      if (playing) tick();
    });
  }
  reconstructed.addEventListener('pause', () => { if (playing === reconstructed) { renderTime(reconstructed.currentTime); playing = null; cancelAnimationFrame(animation); animation = 0; } });
  reconstructed.addEventListener('seeked', () => renderTime(reconstructed.currentTime));
  reconstructed.addEventListener('ended', () => { renderTime(duration); playing = null; cancelAnimationFrame(animation); animation = 0; });
  document.addEventListener('visibilitychange', () => { if (document.hidden) { stopPlayback(); cancelRecording('Recording discarded because the page went into the background.'); cancel('Fitting stopped while the page was in the background. Start again to finish the reconstruction.'); } });
  window.addEventListener('pagehide', () => cancelRecording());
  resetView(); updateArchitecture(); syncInputControls();
  if (recorder && !recorder.supported) recordingStatus('Microphone recording is unavailable here. Open this page on localhost or HTTPS in a browser with recording support, or choose an audio file.');
  return {cancel: message => { cancelRecording(); cancel(message); }};
}
