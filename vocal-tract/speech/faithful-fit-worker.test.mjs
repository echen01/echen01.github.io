import assert from 'node:assert/strict';
import {runFaithfulFit} from './faithful-fit-worker.js';
import {computeSpectrogram} from './spectrogram.js';

const deviceLoss = () => Object.assign(new Error('Mapping WebGPU buffer failed: Context lost'), {code: 'WEBGPU_DEVICE_LOST'});
const data = {type: 'start', samples: new Float32Array([.5, -1]), uin: new Float32Array([1, 0]),
  sampleRate: 16000, hiddenDim: 256, iterations: 10, backend: 'gpu', pitch: {pitchHz: [100]}};
const tick = () => new Promise(resolve => setImmediate(resolve));

function rig({phase, failures = [1], error = deviceLoss, abortPhase, stepMs = 100, hook,
  waveform = () => new Float32Array([.5, -1])} = {}) {
  const events = [], messages = [], calls = [], transfers = [], controller = new AbortController();
  let time = 0, attempts = 0, cpuCalls = 0;
  async function point(name, attempt, iteration) {
    events.push(`${name}:${attempt}:${iteration}`);
    await hook?.({name, attempt, iteration, controller, events});
    if (name === abortPhase) controller.abort();
    if (name === phase && failures.includes(attempt)) throw error();
  }
  const create = async (input, options) => {
    const attempt = ++attempts;
    calls.push({input, options});
    try { await point('create', attempt, 0); }
    catch (failure) {
      // A failing constructor has not returned a disposable fitter. The real
      // engine similarly owns cleanup of its partially initialized device.
      events.push(`constructor-cleanup:${attempt}`);
      throw failure;
    }
    let iteration = 0;
    const history = [];
    return {
      async step() {
        await point('step', attempt, iteration + 1);
        iteration++;
        time += stepMs;
        history.push(1 / iteration);
        return {iteration, loss: history.at(-1)};
      },
      async snapshot() {
        await point('snapshot', attempt, iteration);
        return {iteration, history: [...history], loss: history.at(-1), areas: new Float32Array([attempt, iteration])};
      },
      async synthesize() {
        await point('synthesize', attempt, iteration);
        return waveform(iteration);
      },
      async dispose() { await point('dispose', attempt, iteration); },
    };
  };
  return {
    events, messages, calls, transfers, controller,
    get attempts() { return attempts; }, get cpuCalls() { return cpuCalls; },
    run(overrides = {}) {
      return runFaithfulFit({...data, ...overrides}, {
        signal: controller.signal, now: () => time, yieldControl: async () => {},
        createGpuFitter: create,
        createCpuFitter: (...args) => { cpuCalls++; return create(...args); },
        onMessage(message, transfer) {
          events.push(`post:${message.type}:${message.iteration ?? ''}`);
          messages.push(message); transfers.push(transfer);
        },
      });
    },
  };
}

// The loss history still contains every update; expensive visualization
// snapshots occur initially, every >=250 ms, and at the final update only.
const normal = rig();
const done = await normal.run();
assert.equal(done.type, 'done'); assert.equal(done.gpuRestarts, 0);
assert.deepEqual(normal.messages.filter(m => m.type === 'progress').map(m => m.iteration), [0, 3, 6, 9, 10]);
assert.deepEqual(done.history, Array.from({length: 10}, (_, i) => 1 / (i + 1)));
assert.equal(normal.events.filter(e => e.startsWith('snapshot:')).length, 5);
assert.equal(normal.events.filter(e => e.startsWith('step:')).length, 10);
assert.equal(normal.events.filter(e => e.startsWith('synthesize:')).length, 1);
assert(normal.events.indexOf('dispose:1:10') < normal.events.indexOf('post:done:10'));
assert.equal(done.normalizationGain, 1 / (1 + 1e-8));
assert.deepEqual(done.samples, Float32Array.from([.5, -1], x => x / (1 + 1e-8)));
assert.deepEqual(normal.transfers.at(-1), [done.samples.buffer]);
assert(!normal.messages.some(m => m.type === 'spectrogram'), 'Diagnostics without the opt-in flag do no preview work.');

// MFN selection must reach both backends and survive a fresh-device retry.
for (const backend of ['cpu', 'fullgpu']) {
  const selected = rig(backend === 'cpu' ? {} : {phase: 'step'});
  assert.equal((await selected.run({backend, fieldType: 'mfn', hiddenDim: undefined, mfnInputScale: 64, mfnAreaScale: .7})).type, 'done');
  assert(selected.calls.every(({options}) => options.fieldType === 'mfn' && options.hiddenDim === 256 && options.mfnInputScale === 64 && options.mfnAreaScale === .7));
  assert.equal(selected.attempts, backend === 'cpu' ? 1 : 2);
}

// Sparse preview synthesis samples post-update parameters with a fixed noise
// key, includes initialization, and reuses the final raw playback synthesis.
const waveform = iteration => new Float32Array([iteration + .5, -1]);
const previews = rig({waveform});
const previewDone = await previews.run({iterations: 200, spectrogramPreviews: true});
const checkpoints = previews.messages.filter(m => m.type === 'spectrogram');
assert.deepEqual(checkpoints.map(m => m.iteration), Array.from({length: 21}, (_, i) => i * 10));
assert.equal(previews.events.filter(e => e.startsWith('synthesize:')).length, 21, 'Do not synthesize the final waveform twice.');
assert.equal(previews.events.filter(e => e.startsWith('step:')).length, 200, 'Previews do not add optimizer updates.');
for (const message of checkpoints) {
  assert.equal(message.iterations, 200); assert.equal(message.gpuRestarts, 0);
  assert.deepEqual(message.spectrum, computeSpectrogram(waveform(message.iteration), {sampleRate: data.sampleRate}),
    'Checkpoint uses the current raw waveform, without playback gain.');
  const transfer = previews.transfers[previews.messages.indexOf(message)];
  assert.deepEqual(transfer, [message.spectrum.power.buffer], 'Only one compact STFT power buffer is transferred per checkpoint.');
}
assert(previews.events.indexOf('dispose:1:200') < previews.events.indexOf('post:spectrogram:200'),
  'Final preview waits for cleanup and the final cancellation check.');
assert.deepEqual(previewDone.history, Array.from({length: 200}, (_, i) => 1 / (i + 1)));

for (const [iterations, stepMs, expected] of [
  [200, 1, [0, 200]],
  [200, 50, Array.from({length: 11}, (_, i) => i * 20)],
  [500, 100, Array.from({length: 21}, (_, i) => i * 25)],
  [1, 100, [0, 1]],
]) {
  const timing = rig({stepMs});
  await timing.run({iterations, spectrogramPreviews: true});
  assert.deepEqual(timing.messages.filter(m => m.type === 'spectrogram').map(m => m.iteration), expected,
    'Both elapsed-time and iteration spacing limits apply, while endpoints are always present.');
}

// A reset in a preview follows the same single-restart lifecycle as training,
// and starts a new checkpoint sequence explicitly identified by gpuRestarts.
const previewRetry = rig({hook({name, attempt, iteration}) {
  if (name === 'synthesize' && attempt === 1 && iteration === 10) throw deviceLoss();
}});
assert.equal((await previewRetry.run({iterations: 20, spectrogramPreviews: true})).type, 'done');
assert.deepEqual(previewRetry.messages.filter(m => m.type === 'spectrogram').map(m => [m.gpuRestarts, m.iteration]),
  [[0, 0], [1, 0], [1, 10], [1, 20]]);
assert.equal(previewRetry.messages.filter(m => m.restart).length, 1);
assert(previewRetry.events.indexOf('dispose:1:10') < previewRetry.events.indexOf('create:2:0'));

for (const iterationToCancel of [0, 10, 20]) {
  const previewCanceled = rig({hook({name, iteration, controller}) {
    if (name === 'synthesize' && iteration === iterationToCancel) controller.abort();
  }});
  const stopped = await previewCanceled.run({iterations: 20, spectrogramPreviews: true});
  assert.equal(stopped.type, 'stopped');
  assert(!previewCanceled.messages.some(m => m.type === 'spectrogram' && m.iteration >= iterationToCancel));
  assert(!previewCanceled.messages.some(m => ['done', 'error'].includes(m.type) || m.restart));
}

// Even a very short fit has an initial and final full snapshot.
const fast = rig({stepMs: 1}); await fast.run({iterations: 2});
assert.deepEqual(fast.messages.filter(m => m.type === 'progress').map(m => m.iteration), [0, 2]);

for (const phase of ['create', 'snapshot', 'step', 'synthesize']) {
  const retry = rig({phase});
  const result = await retry.run();
  assert.equal(result.type, 'done', `${phase} loss should recover`);
  assert.equal(result.gpuRestarts, 1); assert.equal(retry.attempts, 2); assert.equal(retry.cpuCalls, 0);
  assert.equal(retry.calls[0].input, retry.calls[1].input, 'Retry reuses the exact source, target and pitch input.');
  assert.equal(retry.calls[0].options, retry.calls[1].options, 'Retry preserves width, iterations and signal.');
  const cleanup = phase === 'create' ? 'constructor-cleanup:1' : retry.events.find(e => e.startsWith('dispose:1:'));
  assert(retry.events.indexOf(cleanup) < retry.events.indexOf('create:2:0'), 'Cleanup must finish before retry.');
  const restart = retry.messages.filter(m => m.restart);
  assert.equal(restart.length, 1); assert.match(restart[0].message, /step 0/);
  assert(retry.events.includes('snapshot:2:0'), 'The new model begins at iteration zero.');
  assert.deepEqual(result.history, done.history, 'Failed attempts must not leak loss history or skip updates.');
  assert.equal(result.areas[0], 2);
  assert.equal(retry.messages.filter(m => m.type === 'done').length, 1);
  assert(retry.events.indexOf('dispose:2:10') < retry.events.indexOf('post:done:10'));
}

const failedTwice = rig({phase: 'step', failures: [1, 2]});
const exhausted = await failedTwice.run();
assert.equal(exhausted.type, 'error'); assert.equal(exhausted.code, 'WEBGPU_DEVICE_LOST');
assert.match(exhausted.message, /lost again/); assert.match(exhausted.message, /WebGPU support/); assert.match(exhausted.message, /Reload/); assert.doesNotMatch(exhausted.message, /CPU reference/);
assert.equal(failedTwice.attempts, 2); assert.equal(failedTwice.cpuCalls, 0);
assert(failedTwice.events.indexOf('dispose:2:0') < failedTwice.events.indexOf('post:error:'));

for (const error of [() => new Error('Invalid tensor shape'), () => new Error('The GPU waveform objective is nonfinite.'),
  () => new DOMException('Mapping aborted without user cancellation', 'AbortError'),
  () => Object.assign(new Error('Validation failed'), {code: 'WEBGPU_VALIDATION_ERROR'})]) {
  const invalid = rig({phase: 'step', error});
  const result = await invalid.run();
  assert.equal(result.type, 'error'); assert.equal(invalid.attempts, 1); assert.equal(invalid.cpuCalls, 0);
  assert(invalid.events.indexOf('dispose:1:0') < invalid.events.indexOf('post:error:'));
  assert(!invalid.messages.some(m => m.restart));
}
const invalidInput = rig();
assert.equal((await invalidInput.run({iterations: NaN})).type, 'error');
assert.equal(invalidInput.attempts, 0);
const cpuLoss = rig({phase: 'step'});
assert.equal((await cpuLoss.run({backend: 'cpu'})).type, 'error');
assert.equal(cpuLoss.attempts, 1); assert.equal(cpuLoss.cpuCalls, 1);

// Cancellation can arrive while any awaited phase is pending, including
// disposal after synthesis. It must prevent retries and all stale completion.
for (const abortPhase of ['create', 'snapshot', 'step', 'synthesize', 'dispose']) {
  const canceled = rig({abortPhase});
  const stopped = await canceled.run();
  assert.equal(stopped.type, 'stopped', abortPhase);
  assert.equal(canceled.attempts, 1);
  assert(!canceled.messages.some(m => ['done', 'error'].includes(m.type) || m.restart));
  const cleanup = canceled.events.find(e => e.startsWith('dispose:1:'));
  assert(cleanup, `${abortPhase} must dispose the acquired fitter`);
  assert(canceled.events.indexOf(cleanup) < canceled.events.indexOf('post:stopped:'));
}
const abortWithLoss = rig({phase: 'step', abortPhase: 'step'});
assert.equal((await abortWithLoss.run()).type, 'stopped'); assert.equal(abortWithLoss.attempts, 1);
const beforeStart = rig(); beforeStart.controller.abort();
assert.equal((await beforeStart.run()).type, 'stopped'); assert.equal(beforeStart.attempts, 0);

// Terminal posts must wait for asynchronous cleanup, not merely call dispose.
let releaseDispose, enteredDispose;
const disposing = new Promise(resolve => { enteredDispose = resolve; });
const deferred = new Promise(resolve => { releaseDispose = resolve; });
const pendingCleanup = rig({hook: async ({name}) => {
  if (name === 'dispose') { enteredDispose(); await deferred; }
}});
const pendingRun = pendingCleanup.run({iterations: 1});
await disposing;
assert(!pendingCleanup.messages.some(m => ['done', 'error', 'stopped'].includes(m.type)));
pendingCleanup.controller.abort(); releaseDispose();
assert.equal((await pendingRun).type, 'stopped');

// Exercise the worker handler itself without GPU access: repeated starts are
// ignored until terminal cleanup; a later start is accepted. Importing in a
// browser main-thread-like scope must not overwrite its message handler.
const previousSelf = globalThis.self, previousDocument = globalThis.document;
try {
  const posted = [];
  globalThis.self = {postMessage: message => posted.push(message)};
  await import(`./faithful-fit-worker.js?worker-lifecycle-test`);
  const invalid = {...data, sampleRate: 0};
  self.onmessage({data: invalid}); self.onmessage({data: invalid});
  await tick();
  assert.equal(posted.length, 1); assert.equal(posted[0].type, 'error');
  self.onmessage({data: invalid}); await tick();
  assert.equal(posted.length, 2);
  self.onmessage({data: invalid}); self.onmessage({data: {type: 'stop'}}); await tick();
  assert.equal(posted.at(-1).type, 'stopped');
  const untouched = () => {};
  self.onmessage = untouched; globalThis.document = {};
  await import(`./faithful-fit-worker.js?main-thread-test`);
  assert.equal(self.onmessage, untouched);
} finally {
  if (previousSelf === undefined) delete globalThis.self; else globalThis.self = previousSelf;
  if (previousDocument === undefined) delete globalThis.document; else globalThis.document = previousDocument;
}

console.log(JSON.stringify({passed: true, retryPhases: ['initialization', 'snapshot', 'step', 'synthesis'],
  retryLimit: 1, noCpuFallback: true, cleanupBeforeTerminalAndRetry: true, cancellationSuppressesStaleResults: true,
  snapshotIterationsAt100msSteps: [0, 3, 6, 9, 10], retainedHistorySteps: done.history.length,
  spectrogramCheckpointsAt100msSteps: checkpoints.map(m => m.iteration),
  spectrogramRawSignalAndTransferVerified: true, spectrogramCancellationAndRetryVerified: true,
  workerRejectsOverlappingStarts: true}));
