import {createSimulator} from '../simulator/index.js';
import {prepareSpeech, createSpeechFitter, DEFAULT_ITERATIONS, MODEL_DESCRIPTION} from './engine.js';
import {synthesizeTrajectory} from './dsp.js';

let simulator, active = null;
self.onmessage = async ({data}) => {
  if (data.type === 'stop') { active?.abort(); active = null; return; }
  if (data.type !== 'start') return;
  active?.abort();
  const controller = new AbortController(); active = controller;
  const current = () => active === controller && !controller.signal.aborted;
  const started = performance.now();
  const send = message => { if (current()) self.postMessage({...message, elapsedMs: performance.now() - started}); };
  try {
    const sim = await (simulator ??= createSimulator());
    if (!current()) return;
    const requestedIterations = data.iterations ?? DEFAULT_ITERATIONS;
    if (!Number.isFinite(requestedIterations)) throw new Error('The iteration count must be finite.');
    const iterations = Math.max(1, Math.min(120, Math.floor(requestedIterations)));
    send({type: 'stage', stage: 'analyzing', message: 'Measuring voiced harmonics and unvoiced spectral envelopes…'});
    const prepared = await prepareSpeech(data.samples, data.pitch, {sampleRate: data.sampleRate ?? 16000, signal: controller.signal,
      onProgress: fraction => send({type: 'stage', stage: 'analyzing', fraction})});
    if (!current()) return;
    send({type: 'stage', stage: 'fitting', message: `Fitting ${prepared.activeFrames} active frames with acoustic gradients…`});
    const fitter = await createSpeechFitter(sim, prepared, {length: data.length ?? 17.5, signal: controller.signal});
    let result = fitter.snapshot();
    send({type: 'progress', ...result, iterations, model: MODEL_DESCRIPTION});
    for (let i = 0; i < iterations; i++) {
      if (!current()) return;
      result = await fitter.step();
      send({type: 'progress', ...result, iterations, model: MODEL_DESCRIPTION});
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
      normalizationGain: samples.normalizationGain ?? 1, model: MODEL_DESCRIPTION, elapsedMs: performance.now() - started}, [samples.buffer]);
    active = null;
  } catch (error) {
    if (current()) { self.postMessage({type: 'error', message: error.message || String(error)}); active = null; }
  }
};
