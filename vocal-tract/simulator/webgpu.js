// Stationary vowel acoustics and Adam on WebGPU, with direct, RFF and MFN fitting.
// Both use the same spectral objective and no additional penalty. Only
// requested plots/audio/diagnostics read results to the host.
import {createWebGpuAcousticKernels} from '../speech/acoustic-webgpu.js?v=vowel-gpu-1';
import {createWebGpuRffKernels} from '../speech/rff-webgpu.js?v=vowel-rff-1';
import {createVowelRffField, vowelLearningRate} from './rff-field.js?v=vowel-mfn-1';
import {createVowelMfnField, loadVowelMfnInitialization} from './mfn-field.js?v=vowel-mfn-e-1';
import {createFaithfulGpuMfnField} from '../speech/faithful-mfn-webgpu.js?v=vowel-mfn-area01-1';
import {FREQUENCIES} from './index.js';
import {synthesizeLfVowel, vowelFrequencies} from './vowel-source.js';

function checkAreas(areas) {
  if (areas?.length !== 32 || !Array.from(areas).every(a => Number.isFinite(a) && a > 0 && a <= 100)) throw new Error('The model requires 32 positive, finite cross-sectional areas.');
}
function checkLength(length) {
  if (!Number.isFinite(length) || length < 5 || length > 40) throw new Error('Tract length must be between 5 and 40 cm.');
}
function prepared(frequencies, targetDb) {
  return {frames: [{active: true, frequencies, targetDb}]};
}

export async function createWebGpuSimulator({gpu = globalThis.navigator?.gpu, onDeviceLost} = {}) {
  if (!gpu) throw new Error('WebGPU is unavailable. Open this page on localhost or HTTPS in a WebGPU-enabled browser.');
  const adapter = await gpu.requestAdapter();
  if (!adapter) throw new Error('No WebGPU adapter is available.');
  const device = await adapter.requestDevice();
  let destroyed = false, failure = null, pending = Promise.resolve();
  const cleanups = new Set(), responses = new Map();
  const fail = message => { if (!destroyed && !failure) { failure = message; onDeviceLost?.(message); } };
  device.lost.then(info => fail(`The browser reset the GPU: ${info.message || info.reason}. Reload the page to try again.`));
  device.addEventListener('uncapturederror', event => { event.preventDefault(); fail(event.error.message); });
  const check = () => { if (destroyed || failure) throw new Error(failure || 'The WebGPU simulator has been disposed.'); };
  const enqueue = operation => {
    const result = pending.then(() => { check(); return operation(); });
    pending = result.catch(() => {});
    return result;
  };
  const storage = (size, data) => {
    const b = device.createBuffer({size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC});
    if (data) device.queue.writeBuffer(b, 0, data);
    return b;
  };
  const readBuffers = async entries => {
    check();
    const size = entries.reduce((sum, [, bytes]) => sum + bytes, 0);
    const staging = device.createBuffer({size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST});
    try {
      const encoder = device.createCommandEncoder(); let offset = 0;
      for (const [buffer, bytes] of entries) { encoder.copyBufferToBuffer(buffer, 0, staging, offset, bytes); offset += bytes; }
      device.queue.submit([encoder.finish()]);
      await staging.mapAsync(GPUMapMode.READ); check();
      const result = staging.getMappedRange().slice(0); staging.unmap();
      return result;
    } finally { staging.destroy(); }
  };
  const shader = async (label, code, resources) => {
    const module = device.createShaderModule({label, code});
    const info = await module.getCompilationInfo(), errors = info.messages.filter(item => item.type === 'error');
    if (errors.length) throw new Error(`${label}: ${errors.map(item => item.message).join('; ')}`);
    const pipeline = await device.createComputePipelineAsync({label, layout: 'auto', compute: {module, entryPoint: 'main'}});
    const bindGroup = device.createBindGroup({layout: pipeline.getBindGroupLayout(0), entries: resources.map((buffer, binding) => ({binding, resource: {buffer}}))});
    return {pipeline, bindGroup};
  };
  const run = (pass, kernel, groups = 1) => { pass.setPipeline(kernel.pipeline); pass.setBindGroup(0, kernel.bindGroup); pass.dispatchWorkgroups(groups); };

  async function evaluateResponse(areas, length, frequencies) {
    checkAreas(areas); checkLength(length);
    if (!frequencies?.length || frequencies.length > 4097 || !frequencies.every(f => Number.isFinite(f) && f >= 0)) throw new Error('Supply between 1 and 4097 nonnegative frequencies.');
    const key = `${length}:${frequencies.join(',')}`;
    let entry = responses.get(key);
    if (!entry) {
      const logs = storage(128); let acoustic;
      const dispose = () => { acoustic?.destroy(); logs.destroy(); cleanups.delete(dispose); };
      cleanups.add(dispose);
      try {
        acoustic = await createWebGpuAcousticKernels(prepared(frequencies, frequencies.map(() => 0)), {
          device, logAreas: logs, length, gainInvariant: false, includeTransfer: true,
        });
        check();
        entry = {logs, acoustic, dispose}; responses.set(key, entry);
        // Bound live GPU resources when pitch or tract length changes.
        if (responses.size > 3) { const oldest = responses.keys().next().value; responses.get(oldest).dispose(); responses.delete(oldest); }
      } catch (error) { dispose(); throw error; }
    }
    device.queue.writeBuffer(entry.logs, 0, Float32Array.from(areas, Math.log));
    const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
    entry.acoustic.encode(pass); pass.end(); device.queue.submit([encoder.finish()]);
    const values = new Float32Array(await readBuffers([[entry.acoustic.transfer, frequencies.length * 8]]));
    if (!values.every(Number.isFinite)) throw new Error('The WebGPU acoustic response is not finite. Reset the tube and try again.');
    const real = [], imag = [], magnitudeDb = [];
    for (let i = 0; i < frequencies.length; i++) {
      real.push(values[i * 2]); imag.push(values[i * 2 + 1]);
      magnitudeDb.push(20 * Math.log10(Math.max(1e-30, Math.hypot(values[i * 2], values[i * 2 + 1]))));
    }
    return {frequencies, real, imag, magnitudeDb};
  }

  async function makeFitter({areas, targetAreas, length = 17.5, learningRate = .035, iterations = 240}) {
    checkAreas(areas); checkAreas(targetAreas); checkLength(length);
    if (!Number.isInteger(iterations) || iterations < 1 || iterations > 1000) throw new Error('Choose between 1 and 1000 updates.');
    if (!Number.isFinite(learningRate) || learningRate <= 0 || learningRate > 1) throw new Error('Invalid learning rate.');
    const targetDb = (await evaluateResponse(targetAreas, length, FREQUENCIES)).magnitudeDb;
    const buffers = []; let acoustic, disposed = false, iteration = 0;
    const alloc = (size, data) => { const b = storage(size, data); buffers.push(b); return b; };
    const dispose = () => {
      if (disposed) return;
      disposed = true; acoustic?.destroy(); for (const b of buffers) b.destroy(); cleanups.delete(dispose);
    };
    const checkFit = () => { check(); if (disposed) throw new Error('The WebGPU vowel fit has been disposed.'); };
    cleanups.add(dispose);
    try {
      const logs = alloc(128, Float32Array.from(areas, Math.log)), moments = alloc(256), state = alloc(8);
      const history = alloc((iterations + 1) * 4);
      const bias = new Float32Array((iterations + 1) * 2);
      for (let i = 1; i <= iterations; i++) { bias[i * 2] = 1 - .9 ** i; bias[i * 2 + 1] = 1 - .999 ** i; }
      const schedule = alloc(bias.byteLength, bias);
      acoustic = await createWebGpuAcousticKernels(prepared(FREQUENCIES, targetDb), {device, logAreas: logs, length, gainInvariant: false});
      const adam = await shader('Vowel Adam and gradient clipping', `
        @group(0) @binding(0) var<storage, read> result: array<f32>;
        @group(0) @binding(1) var<storage, read_write> logs: array<f32>;
        @group(0) @binding(2) var<storage, read_write> moments: array<vec2<f32>>;
        @group(0) @binding(3) var<storage, read_write> state: array<u32>;
        @group(0) @binding(4) var<storage, read> bias: array<vec2<f32>>;
        @compute @workgroup_size(1) fn main() {
          if (state[1] != 0u) { return; }
          var squared = 0.0;
          for (var i = 0u; i < 32u; i++) { let g = result[i + 2u]; squared += g*g; }
          if (!(squared <= 3.402823e38)) { state[1] = 1u; return; }
          let clip = min(1.0, 5.0 / max(1e-12, sqrt(squared)));
          state[0] += 1u; let correction = bias[state[0]];
          for (var i = 0u; i < 32u; i++) {
            let g = result[i + 2u] * clip;
            let m = 0.9 * moments[i].x + 0.1 * g;
            let v = 0.999 * moments[i].y + 0.001 * g*g;
            moments[i] = vec2<f32>(m,v);
            logs[i] = clamp(logs[i] - ${learningRate} * (m/correction.x) / (sqrt(v/correction.y) + 1e-8), ${Math.log(.12)}, ${Math.log(16)});
          }
        }`, [acoustic.result, logs, moments, state, schedule]);
      const record = await shader('Vowel loss history', `
        @group(0) @binding(0) var<storage, read> result: array<f32>;
        @group(0) @binding(1) var<storage, read_write> history: array<f32>;
        @group(0) @binding(2) var<storage, read_write> state: array<u32>;
        @compute @workgroup_size(1) fn main() {
          history[state[0]] = result[0];
          if (!(abs(result[0]) <= 3.402823e38)) { state[1] = 1u; }
        }`, [acoustic.result, history, state]);
      const evaluate = pass => { acoustic.encode(pass); run(pass, record); };
      checkFit();
      const initial = device.createCommandEncoder(), pass = initial.beginComputePass();
      evaluate(pass); pass.end(); device.queue.submit([initial.finish()]);
      await device.queue.onSubmittedWorkDone(); checkFit();
      return {
        backend: 'webgpu', dispose,
        step(count = 1) {
          return enqueue(async () => {
            checkFit();
            if (!Number.isInteger(count) || count < 1 || count > 8 || iteration + count > iterations) throw new Error('Invalid vowel step count.');
            const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
            for (let i = 0; i < count; i++) { run(pass, adam); evaluate(pass); }
            pass.end(); device.queue.submit([encoder.finish()]);
            await device.queue.onSubmittedWorkDone(); checkFit(); iteration += count;
            return {iteration};
          });
        },
        snapshot() {
          return enqueue(async () => {
            checkFit();
            const raw = await readBuffers([[state, 8], [logs, 128], [acoustic.result, 136], [history, (iteration + 1) * 4], [acoustic.response, FREQUENCIES.length * 33 * 4]]);
            const control = new Uint32Array(raw, 0, 2), values = new Float32Array(raw, 8);
            if (control[1] || control[0] !== iteration || !values.every(Number.isFinite)) throw new Error('The WebGPU vowel fit produced invalid values. Reset the tube and try again.');
            const historyValues = Array.from(values.subarray(66, 66 + iteration + 1));
            const residual = values.subarray(66 + iteration + 1), gradient = Array.from(values.subarray(34, 66));
            return {backend: 'webgpu', iteration, areas: Array.from(values.subarray(0, 32), Math.exp),
              loss: values[32], gradient, gradientNorm: Math.hypot(...gradient), history: historyValues, targetDb: [...targetDb],
              magnitudeDb: FREQUENCIES.map((_, i) => targetDb[i] + residual[i * 33] * 20 / Math.LN10)};
          });
        },
      };
    } catch (error) { dispose(); throw error; }
  }
  async function makeFieldFitter({areas, targetAreas, length = 17.5, learningRate = .005, iterations = 240, fieldConfig = {}, initialization}, family = 'rff') {
    checkAreas(areas); checkAreas(targetAreas); checkLength(length);
    if (!Number.isInteger(iterations) || iterations < 1 || iterations > 1000) throw new Error('Choose between 1 and 1000 updates.');
    if (!Number.isFinite(learningRate) || learningRate <= 0 || learningRate > .1) throw new Error('Invalid field learning rate.');
    const field = family === 'mfn'
      ? createVowelMfnField(areas, initialization ?? await loadVowelMfnInitialization(), fieldConfig)
      : createVowelRffField(areas, fieldConfig);
    const P = field.parameterCount;
    const targetDb = (await evaluateResponse(targetAreas, length, FREQUENCIES)).magnitudeDb;
    const buffers = []; let network, acoustic, disposed = false, iteration = 0;
    const alloc = (size, data) => { const b = storage(size, data); buffers.push(b); return b; };
    const dispose = () => {
      if (disposed) return;
      disposed = true; network?.destroy(); acoustic?.destroy();
      for (const b of buffers) b.destroy(); cleanups.delete(dispose);
    };
    const checkFit = () => { check(); if (disposed) throw new Error('The neural-field vowel fit has been disposed.'); };
    cleanups.add(dispose);
    try {
      if (family === 'mfn') {
        const mfn = await createFaithfulGpuMfnField(device, field, {iterations, computePass: true, optimizer: false});
        network = {...mfn, destroy: mfn.dispose};
      } else network = await createWebGpuRffKernels(field, {device});
      acoustic = await createWebGpuAcousticKernels(prepared(FREQUENCIES, targetDb), {
        device, logAreas: network.logAreaBuffer, length, gainInvariant: false,
      });
      const state = alloc(8), clipValue = alloc(4), moments = alloc(P * 8), history = alloc((iterations + 1) * 4);
      const settings = new Float32Array((iterations + 1) * 4);
      for (let n = 1; n <= iterations; n++) {
        settings[n * 4] = vowelLearningRate(n - 1, iterations, learningRate);
        settings[n * 4 + 1] = 1 - .9 ** n; settings[n * 4 + 2] = 1 - .999 ** n;
      }
      const schedule = alloc(settings.byteLength, settings);
      const cotangent = await shader(`Vowel acoustic to ${family} cotangent`, `
        @group(0) @binding(0) var<storage, read> result: array<f32>;
        @group(0) @binding(1) var<storage, read_write> g: array<f32>;
        @compute @workgroup_size(32) fn main(@builtin(local_invocation_index) i: u32) { g[i] = result[i + 2u]; }
      `, [acoustic.result, network.cotangentBuffer]);
      const clip = await shader(`Vowel ${family} global gradient clipping`, `
        @group(0) @binding(0) var<storage, read> g: array<f32>;
        @group(0) @binding(1) var<storage, read_write> state: array<u32>;
        @group(0) @binding(2) var<storage, read_write> clip: array<f32>;
        var<workgroup> sums: array<f32, 64>;
        @compute @workgroup_size(64) fn main(@builtin(local_invocation_index) lane: u32) {
          var total = 0.0;
          for (var p = lane; p < ${P}u; p += 64u) { total += g[p] * g[p]; }
          sums[lane] = total; workgroupBarrier();
          for (var stride = 32u; stride > 0u; stride /= 2u) {
            if (lane < stride) { sums[lane] += sums[lane + stride]; } workgroupBarrier();
          }
          if (lane == 0u) {
            state[0] += 1u;
            if (!(sums[0] <= 3.402823e38)) { state[1] = 1u; }
            clip[0] = min(1.0, 5.0 / max(1e-12, sqrt(sums[0])));
          }
        }`, [network.gradientBuffer, state, clipValue]);
      const adam = await shader(`Vowel ${family} Adam`, `
        @group(0) @binding(0) var<storage, read> g: array<f32>;
        @group(0) @binding(1) var<storage, read_write> p: array<f32>;
        @group(0) @binding(2) var<storage, read_write> moments: array<vec2<f32>>;
        @group(0) @binding(3) var<storage, read> state: array<u32>;
        @group(0) @binding(4) var<storage, read> schedule: array<vec4<f32>>;
        @group(0) @binding(5) var<storage, read> clip: array<f32>;
        @compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
          let i = id.x; if (i >= ${P}u || state[1] != 0u) { return; }
          let gradient = g[i] * clip[0]; let settings = schedule[state[0]];
          let m = .9 * moments[i].x + .1 * gradient;
          let v = .999 * moments[i].y + .001 * gradient * gradient;
          moments[i] = vec2<f32>(m, v);
          p[i] -= settings.x * (m / settings.y) / (sqrt(v / settings.z) + 1e-8);
        }`, [network.gradientBuffer, network.parameterBuffer, moments, state, schedule, clipValue]);
      const record = await shader(`Vowel ${family} loss history`, `
        @group(0) @binding(0) var<storage, read> result: array<f32>;
        @group(0) @binding(1) var<storage, read_write> history: array<f32>;
        @group(0) @binding(2) var<storage, read_write> state: array<u32>;
        @compute @workgroup_size(1) fn main() {
          history[state[0]] = result[0];
          if (!(abs(result[0]) <= 3.402823e38)) { state[1] = 1u; }
        }`, [acoustic.result, history, state]);
      const evaluate = pass => {
        network.encodeForward(pass); acoustic.encode(pass); run(pass, cotangent);
        network.encodeBackward(pass); run(pass, record);
      };
      checkFit();
      const initial = device.createCommandEncoder(), pass = initial.beginComputePass();
      evaluate(pass); pass.end(); device.queue.submit([initial.finish()]);
      await device.queue.onSubmittedWorkDone(); checkFit();
      return {backend: 'webgpu', dispose,
        step(count = 1) {
          return enqueue(async () => {
            checkFit();
            if (!Number.isInteger(count) || count < 1 || count > 8 || iteration + count > iterations) throw new Error('Invalid field step count.');
            const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
            for (let i = 0; i < count; i++) { run(pass, clip); run(pass, adam, Math.ceil(P / 64)); evaluate(pass); }
            pass.end(); device.queue.submit([encoder.finish()]);
            await device.queue.onSubmittedWorkDone(); checkFit(); iteration += count;
            return {iteration};
          });
        },
        snapshot({diagnostic = false} = {}) {
          return enqueue(async () => {
            checkFit();
            const raw = await readBuffers([[state, 8], [network.logAreaBuffer, 128], [acoustic.result, 136],
              [history, (iteration + 1) * 4], [acoustic.response, FREQUENCIES.length * 33 * 4],
              ...(diagnostic ? [[network.parameterBuffer, P * 4], [network.gradientBuffer, P * 4]] : [])]);
            const control = new Uint32Array(raw, 0, 2), values = new Float32Array(raw, 8);
            if (control[1] || control[0] !== iteration || !values.every(Number.isFinite)) throw new Error('The neural-field vowel fit produced invalid values. Reset and try again.');
            const residualOffset = 66 + iteration + 1, diagnosticOffset = residualOffset + FREQUENCIES.length * 33;
            const gradient = Array.from(values.subarray(34, 66));
            return {backend: 'webgpu', parameterization: family, field: field.metadata, iteration,
              areas: Array.from(values.subarray(0, 32), log => Math.max(.12, Math.min(16, Math.exp(log)))), loss: values[32], gradient, gradientNorm: Math.hypot(...gradient),
              history: Array.from(values.subarray(66, residualOffset)), targetDb: [...targetDb],
              magnitudeDb: FREQUENCIES.map((_, i) => targetDb[i] + values[residualOffset + i * 33] * 20 / Math.LN10),
              ...(diagnostic ? {parameters: values.slice(diagnosticOffset, diagnosticOffset + P), parameterGradient: values.slice(diagnosticOffset + P)} : {})};
          });
        },
      };
    } catch (error) { dispose(); throw error; }
  }
  return {
    backend: 'webgpu',
    response(areas, length = 17.5, frequencies = FREQUENCIES) {
      const shape = Array.from(areas), bins = Array.from(frequencies);
      return enqueue(() => evaluateResponse(shape, length, bins));
    },
    async synthesize(areas, length = 17.5, f0 = 120, duration = .8) {
      if (!Number.isFinite(duration) || duration <= 0 || duration > 5) throw new Error('Sound duration must be between 0 and 5 seconds.');
      const response = await this.response(areas, length, vowelFrequencies(f0));
      return synthesizeLfVowel(response, f0, duration);
    },
    createFitter(options) {
      const input = {...options, areas: Array.from(options.areas), targetAreas: Array.from(options.targetAreas)};
      return enqueue(() => makeFitter(input));
    },
    createRffFitter(options) {
      const input = {...options, areas: Array.from(options.areas), targetAreas: Array.from(options.targetAreas)};
      return enqueue(() => makeFieldFitter(input, 'rff'));
    },
    createMfnFitter(options) {
      const input = {...options, areas: Array.from(options.areas), targetAreas: Array.from(options.targetAreas)};
      return enqueue(() => makeFieldFitter(input, 'mfn'));
    },
    dispose() {
      if (destroyed) return;
      destroyed = true; for (const cleanup of [...cleanups]) cleanup(); responses.clear(); device.destroy();
    },
  };
}
