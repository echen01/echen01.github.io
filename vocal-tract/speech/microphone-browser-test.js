import {setupSpeechExperiment} from '../speech-ui.js?v=playground-simple-2';
import {createMicrophoneRecorder} from './microphone-recorder.js?v=recording-1';

const $ = id => document.getElementById(id);
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));
const tracks = [], oscillators = [], states = [];
let audioContext, experiment, recorder, running = false;
let report = {};

function diagnostic(event, details = {}) {
  (report.diagnostics ??= []).push({event, wallTime: performance.now(), visibility: document.visibilityState,
    audioContext: audioContext ? {state: audioContext.state, currentTime: audioContext.currentTime,
      sampleRate: audioContext.sampleRate, baseLatency: audioContext.baseLatency, outputLatency: audioContext.outputLatency} : null,
    ...details});
  $('test-results').textContent = JSON.stringify(report, null, 2);
}

function instrumentedMediaRecorder() {
  if (!globalThis.MediaRecorder) return undefined;
  return class DiagnosticMediaRecorder extends globalThis.MediaRecorder {
    constructor(stream, options) {
      super(stream, options);
      this.testStream = stream;
      diagnostic('recorder-created', {mimeType: this.mimeType, streamActive: stream.active,
        tracks: stream.getTracks().map(track => ({kind: track.kind, enabled: track.enabled, muted: track.muted,
          readyState: track.readyState, settings: track.getSettings()}))});
      for (const event of ['start', 'pause', 'resume', 'stop', 'error', 'dataavailable']) {
        this.addEventListener(event, value => diagnostic(`recorder-${event}`, {state: this.state,
          streamActive: stream.active, trackStates: stream.getTracks().map(track => track.readyState),
          ...(event === 'dataavailable' ? {bytes: value.data.size, type: value.data.type, timecode: value.timecode} : {}),
          ...(event === 'error' ? {error: value.error?.message} : {})}));
      }
    }
    start(timeslice) { diagnostic('recorder-start-called', {timeslice}); return super.start(timeslice); }
    stop() { diagnostic('recorder-stop-called', {state: this.state, streamActive: this.testStream.active}); return super.stop(); }
  };
}

function show(message) {
  $('test-status').textContent = message;
  $('test-results').textContent = JSON.stringify(report, null, 2);
}
function assert(value, message) {
  if (!value) throw new Error(message);
}
async function waitFor(predicate, description, timeout = 15000, progress) {
  const start = performance.now();
  while (performance.now() - start < timeout) {
    if (predicate()) return;
    progress?.();
    await sleep(60);
  }
  throw new Error(`Timed out waiting for ${description}. Recording: ${$('speech-record-status').textContent}. Fitting: ${$('speech-status').textContent}`);
}
function change(id, value) {
  $(id).value = value;
  $(id).dispatchEvent(new Event('change', {bubbles: true}));
}
async function waitForRecording() {
  await waitFor(() => !$('speech-record-stop').hidden, 'recording to start');
  for (const id of ['speech-record-stop', 'speech-record-cancel']) {
    assert($(id).getClientRects().length > 0 && $(id).textContent.trim(), `${id} must be visible and labeled during capture.`);
  }
  assert($('speech-fit').disabled, 'Fit must be unavailable during capture.');
  assert($('speech-file').disabled, 'File selection must be unavailable during capture.');
}
async function waitForPreview(previousUrl) {
  const preview = $('speech-recording-preview');
  await waitFor(() => !preview.hidden && preview.src !== previousUrl && !$('speech-fit').disabled, 'the recorded sample to become ready');
  await waitFor(() => Number.isFinite(preview.duration) && preview.duration > 0, 'recorded WAV metadata');
  assert(preview.getClientRects().length > 0 && preview.controls, 'Recorded audio must have visible playback controls.');
  assert(preview.duration >= .2 && preview.duration <= 10, 'Recorded preview duration must stay inside the fitting limits.');
  assert($('speech-original').src === preview.src, 'The fitting input and recording preview must use the same selected audio.');
  assert($('speech-caption').textContent.startsWith('Microphone recording'), 'The recorded sample must be selected for fitting.');
  assert(tracks.every(track => track.readyState === 'ended'), 'Recording must release all stream tracks before fitting.');
  return {duration: preview.duration, caption: $('speech-caption').textContent, allTracksEnded: true};
}

async function initialize() {
  const response = await fetch(new URL('../index.html', import.meta.url));
  if (!response.ok) throw new Error(`Main page could not load (${response.status}).`);
  const source = new DOMParser().parseFromString(await response.text(), 'text/html');
  for (const selector of ['#speech-lab']) {
    const section = source.querySelector(selector);
    assert(section, `Missing main page section: ${selector}`);
    $('test-page').append(document.importNode(section, true));
  }
  $('speech-original').src = new URL('../assets/speech-original.wav', import.meta.url).href;
  experiment = setupSpeechExperiment({
    parameterization: 'faithful-rff', iterations: 2,
    createRecorder(options) {
      recorder = createMicrophoneRecorder({
        ...options, maxSeconds: 1, pageTarget: null, MediaRecorder: instrumentedMediaRecorder(),
        onState(state) { states.push(state); options.onState(state); },
        async getUserMedia() {
          assert(audioContext?.state === 'running', 'The synthetic audio source must be running.');
          const destination = audioContext.createMediaStreamDestination();
          const oscillator = audioContext.createOscillator();
          const gain = audioContext.createGain();
          oscillator.type = 'sawtooth'; oscillator.frequency.value = 180;
          gain.gain.value = .2;
          oscillator.connect(gain).connect(destination);
          oscillator.start(); oscillators.push(oscillator);
          tracks.push(...destination.stream.getTracks());
          diagnostic('synthetic-source-started', {oscillator: oscillator.type, frequency: oscillator.frequency.value,
            gain: gain.gain.value, streamActive: destination.stream.active});
          return destination.stream;
        },
      });
      return recorder;
    },
  });
  $('test-run').disabled = false;
  show('Ready. The check uses a generated tone and no microphone permission.');
}

$('test-run').addEventListener('click', async () => {
  if (running) return;
  running = true; $('test-run').disabled = true;
  report = {passed: false, source: 'Native MediaRecorder capturing a synthetic 180 Hz oscillator; no real microphone', checks: []};
  states.length = 0; tracks.length = 0;
  const started = performance.now();
  try {
    assert(!document.hidden, 'Keep the test tab visible; the application intentionally discards background recordings.');
    const Context = window.AudioContext || window.webkitAudioContext;
    assert(Context, 'Web Audio is unavailable.');
    audioContext = new Context();
    await audioContext.resume();
    diagnostic('audio-context-resumed');
    assert(recorder.supported, 'Native MediaRecorder is unavailable.');
    assert($('speech-record').getClientRects().length > 0 && !$('speech-record').disabled, 'Record audio must be visible and enabled on the main page.');
    assert($('speech-field-type').value === 'mfn', 'MFN must be selected by default.');
    assert($('speech-area-scale').textContent === '1', 'MFN area scale must remain 1.');
    $('speech-iterations').value = '2';
    assert($('speech-backend').value === 'fullgpu', 'Recording integration must use WebGPU.');
    report.settings = {field: 'mfn', inputScale: Number($('speech-mfn-scale').value), areaScale: 1, iterations: 2, backend: 'fullgpu'};

    show('Checking manual stop and preview…');
    const initialPreviewUrl = $('speech-recording-preview').src;
    $('speech-record').click();
    await waitForRecording();
    await sleep(800);
    diagnostic('manual-stop-click');
    $('speech-record-stop').click();
    report.manualStop = await waitForPreview(initialPreviewUrl);
    report.checks.push('Manual stop produces a playable selected WAV and releases capture tracks.');

    show('Checking that discard retains the previous sample…');
    const retainedUrl = $('speech-recording-preview').src;
    const retainedCaption = $('speech-caption').textContent;
    $('speech-record').click();
    await waitForRecording();
    await sleep(150);
    $('speech-record-cancel').click();
    await waitFor(() => !$('speech-fit').disabled && !$('speech-recording-preview').hidden, 'discard to restore the prior sample');
    assert($('speech-recording-preview').src === retainedUrl && $('speech-caption').textContent === retainedCaption, 'Discard must retain the previously selected recording.');
    assert(tracks.every(track => track.readyState === 'ended'), 'Discard must release every capture track.');
    report.checks.push('Discard restores the previous audio and releases capture tracks.');

    show('Checking automatic stop and loading the full recorded sample…');
    $('speech-record').click();
    await waitForRecording();
    report.automaticStop = await waitForPreview(retainedUrl);
    assert(report.automaticStop.duration >= .8 && report.automaticStop.duration < 1.3, 'The one-second test capture should stop automatically around one second.');
    report.checks.push('Automatic stop prepares the captured audio for fitting.');

    assert(navigator.gpu, 'WebGPU is unavailable; capture checks passed but the requested GPU integration check cannot run.');
    show('Fitting the recorded sample through SwiftF0 and the full GPU simulator…');
    $('speech-fit').click();
    await waitFor(() => {
      const engine = $('speech-engine').textContent;
      if (engine === 'Stopped') throw new Error($('speech-status').textContent);
      return engine === 'Fit complete';
    }, 'two-step recorded-audio fit', 240000, () => show(`Fitting: ${$('speech-status').textContent}`));
    assert(!$('speech-reconstructed').hidden && $('speech-reconstructed').src.startsWith('blob:'), 'Fitting must produce reconstructed audio.');
    assert($('speech-step').textContent === '2 / 2', 'The fit must complete both optimizer updates.');
    const loss = Number($('speech-loss').textContent);
    assert(Number.isFinite(loss), 'The fitted loss must be finite.');
    assert($('speech-backend-note').textContent === 'WebGPU', 'The integration fit must actually use the WebGPU backend.');
    report.fit = {loss, steps: $('speech-step').textContent, backend: $('speech-backend-note').textContent,
      status: $('speech-status').textContent, spectrogram: $('speech-spectrogram-status').textContent};
    report.checks.push('The real page fits captured audio with SwiftF0 and the full GPU MFN, and produces reconstruction audio.');
    report.passed = true;
  } catch (error) {
    report.error = error.stack || String(error);
    diagnostic('failure', {recordingStatus: $('speech-record-status').textContent,
      speechStatus: $('speech-status').textContent});
    experiment?.cancel();
  } finally {
    for (const oscillator of oscillators.splice(0)) { try { oscillator.stop(); } catch {} }
    for (const track of tracks) track.stop();
    diagnostic('cleanup-before-context-close');
    await audioContext?.close(); audioContext = null;
    report.states = states.slice();
    report.captureTracksEnded = tracks.every(track => track.readyState === 'ended');
    report.elapsedSeconds = (performance.now() - started) / 1000;
    show(report.passed ? 'Passed: record, stop, discard, automatic stop, preview, and GPU fitting.' : 'Failed. See the result below.');
    running = false; $('test-run').disabled = false;
  }
});

initialize().catch(error => { report.error = error.stack || String(error); show('Initialization failed.'); });
