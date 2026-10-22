import {createSimulator, createFitter} from './index.js';

let simulator;
let activeRun = 0;

self.onmessage = async ({data}) => {
  if (data.type === 'stop') { activeRun++; return; }
  if (data.type !== 'start') return;
  const run = ++activeRun;
  try {
    const sim = await (simulator ??= createSimulator());
    if (run !== activeRun) return;
    const requestedIterations = data.iterations ?? 240;
    if (!Number.isFinite(requestedIterations)) throw new Error('The iteration count must be finite.');
    const iterations = Math.max(1, Math.min(1000, Math.floor(requestedIterations)));
    const fitter = createFitter(sim, data);
    const started = performance.now();
    let result = fitter.snapshot();
    self.postMessage({type: 'progress', ...result, elapsedMs: 0});
    for (let step = 0; step < iterations; step++) {
      if (run !== activeRun) return;
      result = fitter.step();
      self.postMessage({type: 'progress', ...result, elapsedMs: performance.now() - started});
      // Yield to receive cancellation; no timer determines the loss or geometry.
      await new Promise(resolve => setTimeout(resolve, 0));
    }
    if (run === activeRun) self.postMessage({type: 'done', ...result, elapsedMs: performance.now() - started});
  } catch (error) {
    if (run === activeRun) self.postMessage({type: 'error', message: error.message || String(error)});
  }
};
