import {createFaithfulFitter} from './faithful-engine.js?v=mfn-area-scale-1';
import {createFaithfulGpuFitter} from './faithful-gpu-engine.js?v=playground-simple-1';
import {createFaithfulSource} from './faithful-source.js';
import {computeSpectrogram} from './spectrogram.js';

/** One fit, with at most one fresh-device restart after an actual GPU loss.
 * Constructors own cleanup when initialization fails before returning a fitter.
 * Once returned, the fitter is always disposed before retry or a terminal post.
 */
export async function runFaithfulFit(data, {
  signal, onMessage = () => {}, createGpuFitter = createFaithfulGpuFitter,
  createCpuFitter = createFaithfulFitter, now = () => performance.now(),
  yieldControl = () => new Promise(resolve => setTimeout(resolve, 0)),
  snapshotIntervalMs = 250,
  spectrogramIntervalMs = 1000,
} = {}) {
  const started = now();
  let fitter;
  const check = () => {
    if (signal?.aborted) throw new DOMException('Fitting canceled.', 'AbortError');
  };
  const emit = (message, transfer = []) => {
    const result = {...message, elapsedMs: now() - started};
    onMessage(result, transfer);
    return result;
  };
  const send = message => { check(); return emit(message); };
  const dispose = async () => {
    const current = fitter;
    fitter = null;
    await current?.dispose();
  };
  try {
    check();
    const sampleRate = data.sampleRate;
    if (![16000, 32000].includes(sampleRate)) throw new Error('The faithful pipeline requires 16 or 32 kHz audio.');
    const iterations = Math.floor(data.iterations ?? 500);
    if (!Number.isInteger(iterations) || iterations < 1 || iterations > 500) throw new Error('Choose 1–500 optimizer updates.');
    const spectrogramPreviews = data.spectrogramPreviews === true;
    // At most 19 intermediate images plus the initial and final states for
    // the standard schedules. Slow browsers still do at least ten updates
    // between readbacks; fast browsers do at most one preview per second.
    const spectrogramStepInterval = Math.max(10, Math.ceil(iterations / 20));
    const sendSpectrogram = (samples, iteration, attempt) => {
      check();
      // Use the raw fixed-key waveform, before playback normalization. The
      // display STFT has no connection to the training loss or its gradients.
      const spectrum = computeSpectrogram(samples, {sampleRate});
      check();
      emit({type: 'spectrogram', spectrum, iteration, iterations, gpuRestarts: attempt}, [spectrum.power.buffer]);
    };
    send({type: 'stage', message: 'Preparing the LF source and original waveform objective…'});
    const uin = data.uin ?? createFaithfulSource(data.samples, data.pitch, {sampleRate}).uin;
    const input = {target: data.samples, uin, sampleRate, metadata: data.pitch};
    const fieldType = data.fieldType ?? 'rff';
    const options = {iterations, fieldType, hiddenDim: data.hiddenDim ?? (fieldType === 'mfn' ? 256 : 512), mfnInputScale: data.mfnInputScale ?? 256,
      mfnAreaScale: data.mfnAreaScale ?? .35, signal,
      onStage: message => send({type: 'stage', message})};
    const gpu = data.backend !== 'cpu';
    const createFitter = gpu ? createGpuFitter : createCpuFitter;
    for (let attempt = 0; attempt < 2; attempt++) {
      try {
        check();
        // Recreate from the original constructor initialization, including Adam
        // state and RNG step. No partially updated weights survive a restart.
        fitter = await createFitter(input, options);
        check();
        send({type: 'progress', ...await fitter.snapshot(), iterations, gpuRestarts: attempt});
        if (spectrogramPreviews) sendSpectrogram(await fitter.synthesize(), 0, attempt);
        const optimizationStarted = now();
        let lastSnapshotAt = optimizationStarted, result;
        let lastSpectrogramAt = optimizationStarted, lastSpectrogramIteration = 0;
        for (let n = 0; n < iterations; n++) {
          check();
          await fitter.step();
          check();
          const optimizationMs = now() - optimizationStarted;
          // Snapshotting runs an additional field forward pass and reads back
          // all areas. Keep scalar optimizer history every step, but refresh
          // the complete visualization at most about four times per second.
          if (n + 1 === iterations || now() - lastSnapshotAt >= snapshotIntervalMs) {
            result = await fitter.snapshot();
            send({type: 'progress', ...result, iterations, gpuRestarts: attempt, optimizationMs,
              iterationsPerSecond: (n + 1) * 1000 / Math.max(1, optimizationMs)});
            lastSnapshotAt = now();
          }
          if (spectrogramPreviews && n + 1 < iterations &&
              n + 1 - lastSpectrogramIteration >= spectrogramStepInterval &&
              now() - lastSpectrogramAt >= spectrogramIntervalMs) {
            // synthesize() evaluates the current post-update parameters with
            // key(0); it neither advances Adam nor consumes the training RNG.
            sendSpectrogram(await fitter.synthesize(), n + 1, attempt);
            lastSpectrogramAt = now();
            lastSpectrogramIteration = n + 1;
          }
          await yieldControl();
        }
        check();
        const optimizationMs = now() - optimizationStarted;
        send({type: 'stage', message: 'Synthesizing the fitted waveform with the original key(0) noise…'});
        const raw = await fitter.synthesize();
        check();
        let peak = 0;
        for (const value of raw) peak = Math.max(peak, Math.abs(value));
        // fit_sample.py normalizes only fitted.wav for playback, after optimization.
        const gain = 1 / (peak + 1e-8), samples = Float32Array.from(raw, value => value * gain);
        await dispose();
        check();
        if (spectrogramPreviews) sendSpectrogram(raw, iterations, attempt);
        return emit({type: 'done', ...result, iterations, gpuRestarts: attempt, samples, sampleRate, normalizationGain: gain,
          optimizationMs, iterationsPerSecond: iterations * 1000 / Math.max(1, optimizationMs)}, [samples.buffer]);
      } catch (error) {
        await dispose();
        check();
        if (gpu && error.code === 'WEBGPU_DEVICE_LOST') {
          if (attempt === 0) {
            send({type: 'stage', restart: true,
              message: 'The WebGPU context was lost. Restarting the same fit from step 0 with a fresh GPU device…'});
            continue;
          }
          error = Object.assign(new Error(`${error.message || 'The WebGPU context was lost.'} The GPU context was lost again. Reload the page to try again, or use another browser with WebGPU support.`),
            {code: error.code});
        }
        throw error;
      }
    }
  } catch (error) {
    await dispose();
    if (signal?.aborted) return emit({type: 'stopped', message: 'Fitting canceled.'});
    return emit({type: 'error', message: error.message || String(error), code: error.code});
  }
}

// Importable in Node for lifecycle tests. A worker accepts one active run; a
// repeated start cannot overlap GPU ownership or produce stale completion.
if (typeof self !== 'undefined' && typeof self.postMessage === 'function' && typeof document === 'undefined') {
  let active;
  self.onmessage = ({data}) => {
    if (data.type === 'stop') { active?.abort(); return; }
    if (data.type !== 'start' || active) return;
    const controller = new AbortController();
    active = controller;
    void runFaithfulFit(data, {signal: controller.signal, onMessage: (message, transfer) => self.postMessage(message, transfer)})
      .finally(() => { if (active === controller) active = null; });
  };
}
