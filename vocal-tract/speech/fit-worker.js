import {createSimulator} from '../simulator/index.js';
import {prepareSpeech, createSpeechFitter, DEFAULT_ITERATIONS, MODEL_DESCRIPTION} from './engine.js';
import {synthesizeTrajectory} from './dsp.js';
import {createRffSpeechFitter, RFF_DEFAULT_ITERATIONS, RFF_MODEL_DESCRIPTION} from './rff-engine.js';
import {createRffField} from './rff-field.js';

let simulator, active = null;
self.onmessage = async ({data}) => {
  if (data.type === 'stop') { active?.abort(); active = null; return; }
  if (data.type !== 'start') return;
  active?.abort();
  const controller = new AbortController(); active = controller;
  const current = () => active === controller && !controller.signal.aborted;
  const started = performance.now();
  const send = message => { if (current()) self.postMessage({...message, elapsedMs: performance.now() - started}); };
  let fitter;
  try {
    const sim = await (simulator ??= createSimulator());
    if (!current()) return;
    const rff = data.parameterization === 'rff';
    const model = rff ? RFF_MODEL_DESCRIPTION : MODEL_DESCRIPTION;
    const requestedIterations = data.iterations ?? (rff ? RFF_DEFAULT_ITERATIONS : DEFAULT_ITERATIONS);
    if (!Number.isFinite(requestedIterations)) throw new Error('The iteration count must be finite.');
    const iterations = Math.max(1, Math.min(rff ? 500 : 120, Math.floor(requestedIterations)));
    const backend = data.backend ?? 'auto';
    if (!['auto', 'cpu', 'webgpu', 'fullgpu'].includes(backend)) throw new Error('Choose a supported fitting backend.');
    const fullGpu = rff && backend === 'fullgpu';
    let fieldFactory = createRffField;
    if (rff && !fullGpu && backend !== 'cpu') {
      const {createWebGpuRffField} = await import('./rff-webgpu.js');
      fieldFactory = config => createWebGpuRffField(createRffField(config), {allowFallback: backend === 'auto'});
    }
    send({type: 'stage', stage: 'analyzing', message: 'Measuring voiced harmonics and unvoiced spectral envelopes…'});
    const prepared = await prepareSpeech(data.samples, data.pitch, {sampleRate: data.sampleRate ?? 16000, signal: controller.signal,
      onProgress: fraction => send({type: 'stage', stage: 'analyzing', fraction})});
    if (!current()) return;
    send({type: 'stage', stage: 'fitting', message: `Fitting ${prepared.activeFrames} active frames with acoustic gradients…`});
    if (fullGpu) {
      const {createFullGpuSpeechFitter} = await import('./full-gpu-engine.js');
      fitter = await createFullGpuSpeechFitter(prepared, {length: data.length ?? 17.5, iterations, signal: controller.signal});
    } else {
      fitter = rff
        ? await createRffSpeechFitter(sim, prepared, {length: data.length ?? 17.5, iterations, signal: controller.signal, fieldFactory})
        : await createSpeechFitter(sim, prepared, {length: data.length ?? 17.5, signal: controller.signal});
    }
    let result = await fitter.snapshot();
    send({type: 'progress', ...result, iterations, model});
    const optimizationStarted = performance.now();
    let lastProgress = optimizationStarted;
    let optimizationMs = 0;
    for (let i = 0; i < iterations; i++) {
      if (!current()) return;
      const update = await fitter.step();
      if (!fullGpu) result = update;
      // Drawing the full heatmap on every fast GPU step can cost more than the
      // fit. Update at most 12.5 times/second, and always send the final state.
      if (performance.now() - lastProgress >= 80 || i === iterations - 1) {
        if (fullGpu) result = await fitter.readSnapshot();
        optimizationMs = performance.now() - optimizationStarted;
        send({type: 'progress', ...result, iterations, model, optimizationMs,
          iterationsPerSecond: result.iteration * 1000 / optimizationMs});
        lastProgress = performance.now();
      }
      await new Promise(resolve => setTimeout(resolve, 0));
    }
    if (!current()) return;
    send({type: 'stage', stage: 'synthesizing', message: 'Synthesizing the moving tract from the estimated source…'});
    const areas = Array.from({length: result.frames}, (_, f) => Array.from(result.areas.subarray(f * 32, (f + 1) * 32)));
    const samples = await synthesizeTrajectory(sim, {samples: data.samples, pitch: data.pitch, areas,
      frameTimes: result.frameTimes, length: result.length, sampleRate: prepared.sampleRate, fftSize: prepared.fftSize, hopLength: prepared.hopLength},
    {signal: controller.signal, onProgress: progress => send({type: 'stage', stage: 'synthesizing', fraction: progress.progress})});
    if (!current()) return;
    self.postMessage({type: 'done', ...result, iterations, samples, sampleRate: prepared.sampleRate,
      normalizationGain: samples.normalizationGain ?? 1, model, optimizationMs,
      iterationsPerSecond: result.iteration * 1000 / optimizationMs,
      elapsedMs: performance.now() - started}, [samples.buffer]);
    active = null;
  } catch (error) {
    if (current()) { self.postMessage({type: 'error', message: error.message || String(error)}); active = null; }
  } finally {
    fitter?.dispose?.();
  }
};
