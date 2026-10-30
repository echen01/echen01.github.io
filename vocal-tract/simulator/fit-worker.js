import {createWebGpuSimulator} from './webgpu.js?v=vowel-mfn-e-1';
import {createSimulator} from './index.js?v=vowel-presets-e-1';
import {createVowelRffFitter} from './rff-field.js?v=vowel-mfn-1';
import {createVowelMfnFitter} from './mfn-field.js?v=vowel-mfn-e-1';

// Show each genuine optimization step at a readable pace: 240 steps take
// about six seconds. Waiting leaves the GPU idle and accepts cancellation.
const FRAME_INTERVAL_MS = 25;
let activeRun = 0;
self.onmessage = async ({data}) => {
  if (data.type === 'stop') { activeRun++; return; }
  if (data.type !== 'start') return;
  const run = ++activeRun;
  const backend = data.backend ?? 'cpu';
  const fieldType = data.fieldType ?? 'mfn';
  let simulator, fitter, completion;
  try {
    if (!['cpu', 'webgpu'].includes(backend)) throw new Error('Choose CPU or WebGPU.');
    if (!['rff', 'mfn'].includes(fieldType)) throw new Error('Choose RFF or MFN.');
    simulator = await (backend === 'webgpu' ? createWebGpuSimulator() : createSimulator());
    if (run !== activeRun) return;
    const requestedIterations = data.iterations ?? 240;
    if (!Number.isFinite(requestedIterations)) throw new Error('The iteration count must be finite.');
    const iterations = Math.max(1, Math.min(1000, Math.floor(requestedIterations)));
    fitter = backend === 'webgpu'
      ? await simulator[fieldType === 'mfn' ? 'createMfnFitter' : 'createRffFitter']({...data, iterations})
      : await (fieldType === 'mfn' ? createVowelMfnFitter : createVowelRffFitter)(simulator, {...data, iterations});
    if (run !== activeRun) return;
    const started = performance.now();
    let result = await fitter.snapshot(), lastUpdate = performance.now();
    self.postMessage({type: 'progress', ...result, backend, elapsedMs: 0});
    for (let step = 0; step < iterations; step++) {
      await new Promise(resolve => setTimeout(resolve, Math.max(0, lastUpdate + FRAME_INTERVAL_MS - performance.now())));
      if (run !== activeRun) return;
      await fitter.step();
      if (run !== activeRun) return;
      result = await fitter.snapshot();
      if (run !== activeRun) return;
      lastUpdate = performance.now();
      self.postMessage({type: 'progress', ...result, backend, elapsedMs: lastUpdate - started});
    }
    completion = {type: 'done', ...result, backend, elapsedMs: performance.now() - started};
  } catch (error) {
    completion = {type: 'error', backend, message: error.message || String(error)};
  } finally {
    fitter?.dispose?.(); simulator?.dispose?.();
    if (run === activeRun && completion) self.postMessage(completion);
    if (run !== activeRun) self.postMessage({type: 'stopped', backend});
    self.close();
  }
};
