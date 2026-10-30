// Experimental GPU-resident optimizer for the same browser RFF inverse problem.
// The field, acoustic adjoint, regularizers, clipping and Adam run in float32.
// Targets/features/schedule are uploaded at initialization. A step synchronizes
// GPU completion, but reads no arrays; only requested UI/diagnostic snapshots do.
import {createRffField} from './rff-field.js';
import {createWebGpuRffKernels} from './rff-webgpu.js';
import {createWebGpuAcousticKernels} from './acoustic-webgpu.js';

const WG = 64, N = 32;
const STATE = `struct State {
  iteration: u32, learningRate: f32, clip: f32, invalid: u32,
  mScale: f32, vScale: f32, padding0: f32, padding1: f32,
}`;

export async function createFullGpuSpeechFitter(prepared, {
  length = 17.5, iterations = 120, learningRate = .01,
  spatialWeight = .006, temporalWeight = .08, fieldConfig = {},
  initialParameters, signal, gpu = globalThis.navigator?.gpu,
} = {}) {
  if (!gpu) throw new Error('WebGPU is unavailable in this browser.');
  if (!Number.isFinite(length) || length < 12 || length > 22) throw new Error('Choose a tract length between 12 and 22 cm.');
  if (!Number.isInteger(iterations) || iterations < 1 || iterations > 500) throw new Error('Choose between 1 and 500 optimization steps.');
  if (!Number.isFinite(learningRate) || learningRate <= 0 || learningRate > .05) throw new Error('Invalid RFF learning rate.');
  if (![spatialWeight, temporalWeight].every(x => Number.isFinite(x) && x >= 0)) throw new Error('Regularization weights must be finite and nonnegative.');
  const frames = prepared.frames.length, count = frames * N;
  if (!frames || prepared.frameTimes.length !== frames) throw new Error('A complete prepared speech trajectory is required.');
  if (prepared.activeFrames !== prepared.frames.filter(frame => frame.active).length) throw new Error('Active frame count is inconsistent.');
  let device, fieldKernels, acoustic, destroyed = false, busy = false, failure = null, iteration = 0;
  const buffers = [];
  const check = () => {
    if (signal?.aborted) throw new DOMException('RFF fitting canceled.', 'AbortError');
    if (destroyed || failure) throw new Error(failure || 'The GPU fitter has been disposed.');
  };
  const dispose = () => {
    if (destroyed) return;
    destroyed = true; fieldKernels?.destroy(); acoustic?.destroy();
    for (const buffer of buffers) buffer.destroy();
    device?.destroy();
  };
  try {
    check();
    const adapter = await gpu.requestAdapter({powerPreference: 'high-performance'});
    if (!adapter) throw new Error('No WebGPU adapter is available.');
    device = await adapter.requestDevice();
    device.lost.then(info => { if (!destroyed) failure = info.message || 'WebGPU device was lost.'; });
    device.addEventListener('uncapturederror', event => { failure = event.error.message; event.preventDefault(); });
    check();
    const field = createRffField({...fieldConfig, frameTimes: prepared.frameTimes, duration: prepared.duration, numAreas: N});
    if (initialParameters !== undefined) {
      if (initialParameters.length !== field.parameterCount || !Array.from(initialParameters).every(Number.isFinite)) throw new Error('Initial parameters must be finite and match the field.');
      field.parameters.set(initialParameters);
    }
    fieldKernels = await createWebGpuRffKernels(field, {device});
    acoustic = await createWebGpuAcousticKernels(prepared, {device, logAreas: fieldKernels.logAreaBuffer, length});
    check();
    const P = field.parameterCount;
    const storage = (size, data) => {
      const buffer = device.createBuffer({size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC});
      buffers.push(buffer); if (data) device.queue.writeBuffer(buffer, 0, data); return buffer;
    };
    const penalties = storage(count * 8), historyBuffer = storage((iterations + 1) * 16);
    const state = storage(32), moments = storage(P * 8);
    // Static schedule constants match the float64 reference before rounding.
    // No per-iteration host uploads or host optimizer arithmetic are needed.
    const scheduleValues = new Float32Array((iterations + 1) * 4);
    const warmup = Math.min(10, Math.max(1, Math.floor(iterations / 10)));
    for (let i = 1; i <= iterations; i++) {
      const progress = Math.max(0, (i - warmup) / Math.max(1, iterations - warmup));
      scheduleValues[i * 4] = learningRate * (i <= warmup ? i / warmup : .1 + .9 * .5 * (1 + Math.cos(Math.PI * progress)));
      scheduleValues[i * 4 + 1] = 1 - .9 ** i; scheduleValues[i * 4 + 2] = 1 - .999 ** i;
    }
    const schedule = storage(scheduleValues.byteLength, scheduleValues);
    const shader = async (label, code, resources) => {
      const module = device.createShaderModule({label, code}), info = await module.getCompilationInfo();
      const errors = info.messages.filter(item => item.type === 'error');
      if (errors.length) throw new Error(`${label}: ${errors.map(item => `${item.lineNum}: ${item.message}`).join('; ')}`);
      const pipeline = await device.createComputePipelineAsync({label, layout: 'auto', compute: {module, entryPoint: 'main'}});
      const bindGroup = device.createBindGroup({layout: pipeline.getBindGroupLayout(0), entries: resources.map((buffer, binding) => ({binding, resource: {buffer}}))});
      return {pipeline, bindGroup};
    };
    const spatialScale = spatialWeight / (frames * (N - 1));
    const temporalScale = frames > 1 ? temporalWeight / ((frames - 1) * N) : 0;
    const regularize = await shader('Full GPU acoustic cotangent and regularization', `
      @group(0) @binding(0) var<storage, read> logs: array<f32>;
      @group(0) @binding(1) var<storage, read> acoustic: array<f32>;
      @group(0) @binding(2) var<storage, read_write> cotangent: array<f32>;
      @group(0) @binding(3) var<storage, read_write> penalties: array<vec2<f32>>;
      @compute @workgroup_size(${WG}) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
        let k = id.x; if (k >= ${count}u) { return; }
        let f = k / 32u; let section = k % 32u; let value = logs[k];
        var g = acoustic[f * 34u + 2u + section] / f32(${acoustic.activeFrames});
        var s = 0.0; var t = 0.0;
        if (section > 0u) { let d = value - logs[k - 1u]; s = f32(${spatialScale}) * d * d; g += 2.0 * f32(${spatialScale}) * d; }
        if (section < 31u) { g += 2.0 * f32(${spatialScale}) * (value - logs[k + 1u]); }
        if (f > 0u) { let d = value - logs[k - 32u]; t = f32(${temporalScale}) * d * d; g += 2.0 * f32(${temporalScale}) * d; }
        if (f + 1u < ${frames}u) { g += 2.0 * f32(${temporalScale}) * (value - logs[k + 32u]); }
        penalties[k] = vec2<f32>(s, t); cotangent[k] = g;
      }`, [fieldKernels.logAreaBuffer, acoustic.result, fieldKernels.cotangentBuffer, penalties]);
    const objective = await shader('Full GPU objective and history', `${STATE}
      @group(0) @binding(0) var<storage, read> penalties: array<vec2<f32>>;
      @group(0) @binding(1) var<storage, read> acoustic: array<f32>;
      @group(0) @binding(2) var<storage, read_write> history: array<vec4<f32>>;
      @group(0) @binding(3) var<storage, read_write> state: State;
      var<workgroup> sums: array<vec3<f32>, ${WG}>;
      @compute @workgroup_size(${WG}) fn main(@builtin(local_invocation_index) lane: u32) {
        var sum = vec3<f32>(0.0);
        for (var k = lane; k < ${count}u; k += ${WG}u) { sum.y += penalties[k].x; sum.z += penalties[k].y; }
        for (var f = lane; f < ${frames}u; f += ${WG}u) { sum.x += acoustic[f * 34u] / f32(${acoustic.activeFrames}); }
        sums[lane] = sum; workgroupBarrier();
        for (var stride = ${WG / 2}u; stride > 0u; stride /= 2u) {
          if (lane < stride) { sums[lane] += sums[lane + stride]; } workgroupBarrier();
        }
        if (lane == 0u) {
          let loss = sums[0].x + sums[0].y + sums[0].z;
          history[state.iteration] = vec4<f32>(loss, sums[0]);
          if (!(abs(loss) <= 3.402823e38)) { state.invalid = 1u; }
        }
      }`, [penalties, acoustic.result, historyBuffer, state]);
    const clip = await shader('Full GPU global gradient clipping', `${STATE}
      @group(0) @binding(0) var<storage, read> gradient: array<f32>;
      @group(0) @binding(1) var<storage, read_write> state: State;
      @group(0) @binding(2) var<storage, read> schedule: array<vec4<f32>>;
      var<workgroup> sums: array<f32, ${WG}>;
      var<workgroup> bad: array<u32, ${WG}>;
      @compute @workgroup_size(${WG}) fn main(@builtin(local_invocation_index) lane: u32) {
        var sum = 0.0; var invalid = 0u;
        for (var p = lane; p < ${P}u; p += ${WG}u) {
          let g = gradient[p]; sum += g * g;
          if (!(abs(g) <= 3.402823e38)) { invalid = 1u; }
        }
        sums[lane] = sum; bad[lane] = invalid; workgroupBarrier();
        for (var stride = ${WG / 2}u; stride > 0u; stride /= 2u) {
          if (lane < stride) { sums[lane] += sums[lane + stride]; bad[lane] |= bad[lane + stride]; } workgroupBarrier();
        }
        if (lane == 0u) {
          state.iteration += 1u;
          let settings = schedule[state.iteration];
          state.learningRate = settings.x; state.mScale = settings.y; state.vScale = settings.z;
          state.clip = min(1.0, 5.0 / max(1e-12, sqrt(sums[0])));
          state.invalid |= bad[0];
          if (!(abs(sums[0]) <= 3.402823e38)) { state.invalid = 1u; }
        }
      }`, [fieldKernels.gradientBuffer, state, schedule]);
    const adam = await shader('Full GPU Adam update', `${STATE}
      @group(0) @binding(0) var<storage, read> gradient: array<f32>;
      @group(0) @binding(1) var<storage, read_write> weights: array<f32>;
      @group(0) @binding(2) var<storage, read_write> moments: array<vec2<f32>>;
      @group(0) @binding(3) var<storage, read> state: State;
      @compute @workgroup_size(${WG}) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
        let p = id.x; if (p >= ${P}u || state.invalid != 0u) { return; }
        let g = gradient[p] * state.clip;
        let m = 0.9 * moments[p].x + 0.1 * g;
        let v = 0.999 * moments[p].y + 0.001 * g * g;
        moments[p] = vec2<f32>(m, v);
        weights[p] -= state.learningRate * (m / state.mScale) / (sqrt(v / state.vScale) + 1e-8);
      }`, [fieldKernels.gradientBuffer, fieldKernels.parameterBuffer, moments, state]);
    const run = (pass, kernel, groups = 1) => { pass.setPipeline(kernel.pipeline); pass.setBindGroup(0, kernel.bindGroup); pass.dispatchWorkgroups(groups); };
    const encodeEvaluation = pass => {
      fieldKernels.encodeForward(pass); acoustic.encode(pass);
      run(pass, regularize, Math.ceil(count / WG)); run(pass, objective); fieldKernels.encodeBackward(pass);
    };
    const initialEncoder = device.createCommandEncoder(), initialPass = initialEncoder.beginComputePass();
    encodeEvaluation(initialPass); initialPass.end(); device.queue.submit([initialEncoder.finish()]);
    await device.queue.onSubmittedWorkDone(); check();

    const read = async diagnostic => {
      check(); if (busy) throw new Error('Await the previous GPU fitter operation first.'); busy = true;
      let staging;
      try {
        const entries = [
          ['history', historyBuffer, (iteration + 1) * 16], ['state', state, 32],
          ['logs', fieldKernels.logAreaBuffer, count * 4], ['parameters', fieldKernels.parameterBuffer, P * 4],
          ['acoustic', acoustic.result, frames * 34 * 4],
        ];
        if (diagnostic) entries.push(['gradient', fieldKernels.gradientBuffer, P * 4], ['areaGradient', fieldKernels.cotangentBuffer, count * 4], ['moments', moments, P * 8]);
        const size = entries.reduce((sum, entry) => sum + entry[2], 0);
        staging = device.createBuffer({size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST});
        const encoder = device.createCommandEncoder(); let offset = 0;
        for (const [, buffer, bytes] of entries) { encoder.copyBufferToBuffer(buffer, 0, staging, offset, bytes); offset += bytes; }
        device.queue.submit([encoder.finish()]); await staging.mapAsync(GPUMapMode.READ);
        const mapped = staging.getMappedRange(), values = {}; offset = 0;
        for (const [name, , bytes] of entries) {
          values[name] = name === 'state' ? new Uint32Array(mapped.slice(offset, offset + bytes)) : Float64Array.from(new Float32Array(mapped, offset, bytes / 4));
          offset += bytes;
        }
        staging.unmap(); check();
        if (values.state[0] !== iteration) throw new Error('GPU iteration counter is inconsistent.');
        if (values.state[3] || !['history', 'logs', 'parameters', ...(diagnostic ? ['gradient', 'areaGradient', 'moments'] : [])].every(key => values[key].every(Number.isFinite))) throw new Error('The full GPU fit produced non-finite values.');
        const losses = values.history, base = iteration * 4;
        field.parameters.set(values.parameters);
        const snapshot = {iteration, iterations, loss: losses[base], totalLoss: losses[base],
          dataLoss: losses[base + 1], spatialLoss: losses[base + 2], temporalLoss: losses[base + 3],
          initialLoss: losses[0], history: Array.from({length: iteration + 1}, (_, i) => losses[i * 4]),
          areas: Float64Array.from(values.logs, Math.exp), frames, frameTimes: [...prepared.frameTimes],
          framePitchHz: [...prepared.framePitchHz], frameVoiced: [...prepared.frameVoiced], frameActive: [...prepared.frameActive],
          activeFrames: acoustic.activeFrames, gainDb: Array.from({length: frames}, (_, f) => values.acoustic[f * 34 + 1]),
          length, framePeriod: prepared.hopLength / prepared.sampleRate, duration: prepared.duration,
          parameterization: 'rff', field: {...field.serialize(), computeBackend: 'webgpu-full-f32'},
          fieldMetadata: field.metadata, parameterCount: P, learningRate: scheduleValues[iteration * 4],
          backend: 'webgpu-full-f32', fallbackReason: null};
        if (!diagnostic) return snapshot;
        return {...snapshot, logs: values.logs, parameters: values.parameters, gradient: values.gradient,
          areaGradient: values.areaGradient, regularizationLoss: snapshot.spatialLoss + snapshot.temporalLoss,
          m: Float64Array.from({length: P}, (_, p) => values.moments[p * 2]),
          v: Float64Array.from({length: P}, (_, p) => values.moments[p * 2 + 1])};
      } finally { staging?.destroy(); busy = false; }
    };
    return {
      backend: 'webgpu-full-f32', adapterInfo: {vendor: adapter.info?.vendor, architecture: adapter.info?.architecture, device: adapter.info?.device, description: adapter.info?.description},
      readSnapshot: () => read(false), snapshot: () => read(false), diagnostics: () => read(true), dispose,
      async step() {
        check(); if (busy) throw new Error('Await the previous GPU fitter operation first.');
        if (iteration >= iterations) throw new Error('The requested RFF optimization steps are complete.');
        busy = true;
        try {
          const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
          run(pass, clip); run(pass, adam, Math.ceil(P / WG)); encodeEvaluation(pass);
          pass.end(); device.queue.submit([encoder.finish()]);
          await device.queue.onSubmittedWorkDone(); iteration++; check();
          return {iteration};
        } finally { busy = false; }
      },
    };
  } catch (error) { dispose(); throw error; }
}
