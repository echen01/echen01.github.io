/** Experimental WebGPU float32 implementation of the existing RFF MLP.
 * Fourier features are calculated once using the same JS float64 encoding and
 * rounded on upload. Forward includes parameter upload + log-area readback;
 * backward includes cotangent upload + parameter-gradient readback. CPU Adam
 * and the float64 acoustic model can therefore use this without modification.
 * No claim of float64 parity: compare the returned values against rff-field.js.
 */
const WG = 64, REDUCTION_ROWS = 128;

function featuresFor(field) {
  const {frameTimes, duration, numAreas, frequencies} = field.config;
  const features = new Float32Array(frameTimes.length * numAreas * frequencies * 2);
  let row = 0;
  for (const time of frameTimes) for (let section = 0; section < numAreas; section++, row++) {
    for (let k = 0; k < frequencies; k++) {
      const phase = 2 * Math.PI * (time / duration * field.encoding[k] + section / (numAreas - 1) * field.encoding[frequencies + k]);
      features[row * 2 * frequencies + k] = Math.cos(phase);
      features[row * 2 * frequencies + frequencies + k] = Math.sin(phase);
    }
  }
  return features;
}

/** GPU-resident kernels for a caller-owned device. Parameters are uploaded once;
 * the caller can write the exposed parameter and cotangent buffers directly and
 * compose forward/backward with other compute stages in the same command pass.
 * destroy() releases this field's buffers only, never the shared device.
 */
export async function createWebGpuRffKernels(field, {device} = {}) {
  if (!device) throw new Error('A WebGPU device is required for resident RFF kernels.');
  const buffers = [];
  let destroyed = false, hasForward = false;
  try {
    const count = field.config.frameTimes.length * field.config.numAreas, params = field.parameters;
    const groups = Math.ceil(count / REDUCTION_ROWS);
    const storage = (length, data) => {
      const buffer = device.createBuffer({size: Math.max(4, length * 4), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC});
      buffers.push(buffer);
      if (data) device.queue.writeBuffer(buffer, 0, data);
      return buffer;
    };
    const parameterBuffer = storage(params.length, Float32Array.from(params));
    const activations = [storage(count * field.config.frequencies * 2, featuresFor(field)), ...field.layout.map(layer => storage(count * layer.outputSize))];
    const deltas = field.layout.map(layer => storage(count * layer.outputSize));
    const logs = storage(count), derivative = storage(count), cotangent = storage(count), gradient = storage(params.length);
    const partials = field.layout.map(layer => storage((layer.weightCount + layer.biasCount) * groups));
    const shader = async (label, code, boundBuffers) => {
      const module = device.createShaderModule({label, code});
      const compilation = await module.getCompilationInfo();
      const errors = compilation.messages.filter(item => item.type === 'error');
      if (errors.length) throw new Error(`${label}: ${errors.map(item => `${item.lineNum}: ${item.message}`).join('; ')}`);
      const pipeline = await device.createComputePipelineAsync({label, layout: 'auto', compute: {module, entryPoint: 'main'}});
      const bindGroup = device.createBindGroup({layout: pipeline.getBindGroupLayout(0), entries: boundBuffers.map((buffer, binding) => ({binding, resource: {buffer}}))});
      return {pipeline, bindGroup};
    };
    const forward = [], reverse = [];
    for (let i = 0; i < field.layout.length; i++) {
      const l = field.layout[i];
      forward.push(await shader(`RFF layer ${i}`, `
        @group(0) @binding(0) var<storage, read> weights: array<f32>;
        @group(0) @binding(1) var<storage, read> input: array<f32>;
        @group(0) @binding(2) var<storage, read_write> output: array<f32>;
        @compute @workgroup_size(${WG}) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
          let index = id.x; if (index >= ${count * l.outputSize}u) { return; }
          let row = index / ${l.outputSize}u; let out = index % ${l.outputSize}u;
          var value = weights[${l.biasOffset}u + out];
          for (var k = 0u; k < ${l.inputSize}u; k++) { value += weights[${l.weightOffset}u + out * ${l.inputSize}u + k] * input[row * ${l.inputSize}u + k]; }
          output[index] = ${l.activation === 'tanh' ? 'tanh(value)' : 'value'};
        }`, [parameterBuffer, activations[i], activations[i + 1]]));
      const partial = await shader(`RFF layer ${i} gradient partial`, `
        @group(0) @binding(0) var<storage, read> input: array<f32>;
        @group(0) @binding(1) var<storage, read> delta: array<f32>;
        @group(0) @binding(2) var<storage, read_write> partial: array<f32>;
        @compute @workgroup_size(${WG}) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
          let p = id.x; let block = id.y; if (p >= ${l.weightCount + l.biasCount}u) { return; }
          let bias = p >= ${l.weightCount}u;
          let out = select(p / ${l.inputSize}u, p - ${l.weightCount}u, bias);
          let col = p % ${l.inputSize}u;
          var sum = 0.0;
          for (var row = block * ${REDUCTION_ROWS}u; row < min((block + 1u) * ${REDUCTION_ROWS}u, ${count}u); row++) {
            var x = 1.0; if (!bias) { x = input[row * ${l.inputSize}u + col]; }
            sum += delta[row * ${l.outputSize}u + out] * x;
          }
          partial[block * ${l.weightCount + l.biasCount}u + p] = sum;
        }`, [activations[i], deltas[i], partials[i]]);
      const reduce = await shader(`RFF layer ${i} gradient reduce`, `
        @group(0) @binding(0) var<storage, read> partial: array<f32>;
        @group(0) @binding(1) var<storage, read_write> gradient: array<f32>;
        @compute @workgroup_size(${WG}) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
          let p = id.x; if (p >= ${l.weightCount + l.biasCount}u) { return; }
          var sum = 0.0;
          for (var block = 0u; block < ${groups}u; block++) { sum += partial[block * ${l.weightCount + l.biasCount}u + p]; }
          gradient[${l.weightOffset}u + p] = sum;
        }`, [partials[i], gradient]);
      let delta = null;
      if (i > 0) delta = await shader(`RFF layer ${i} reverse`, `
        @group(0) @binding(0) var<storage, read> weights: array<f32>;
        @group(0) @binding(1) var<storage, read> input: array<f32>;
        @group(0) @binding(2) var<storage, read> delta: array<f32>;
        @group(0) @binding(3) var<storage, read_write> previous: array<f32>;
        @compute @workgroup_size(${WG}) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
          let index = id.x; if (index >= ${count * l.inputSize}u) { return; }
          let row = index / ${l.inputSize}u; let col = index % ${l.inputSize}u;
          var sum = 0.0;
          for (var out = 0u; out < ${l.outputSize}u; out++) { sum += delta[row * ${l.outputSize}u + out] * weights[${l.weightOffset}u + out * ${l.inputSize}u + col]; }
          let value = input[index]; previous[index] = sum * (1.0 - value * value);
        }`, [parameterBuffer, activations[i], deltas[i], deltas[i - 1]]);
      reverse.push({partial, reduce, delta});
    }
    const residual = field.metadata.areaTransform?.type === 'residual-log-area';
    if (residual && (field.baseLogAreas?.length !== count || !field.baseLogAreas.every(Number.isFinite))) throw new Error('The vowel field requires one finite starting log area per section.');
    const baseline = residual ? storage(count, Float32Array.from(field.baseLogAreas)) : null;
    const area = await shader('RFF area transform', residual ? `
      @group(0) @binding(0) var<storage, read> input: array<f32>;
      @group(0) @binding(1) var<storage, read_write> logs: array<f32>;
      @group(0) @binding(2) var<storage, read_write> derivative: array<f32>;
      @group(0) @binding(3) var<storage, read> baseline: array<f32>;
      @compute @workgroup_size(${WG}) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
        let row = id.x; if (row >= ${count}u) { return; }
        let value = baseline[row] + input[row];
        logs[row] = clamp(value, ${Math.log(.12)}, ${Math.log(16)});
        derivative[row] = select(0.0, 1.0, value > ${Math.log(.12)} && value < ${Math.log(16)});
      }` : `
      @group(0) @binding(0) var<storage, read> input: array<f32>;
      @group(0) @binding(1) var<storage, read_write> logs: array<f32>;
      @group(0) @binding(2) var<storage, read_write> derivative: array<f32>;
      @compute @workgroup_size(${WG}) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
        let row = id.x; if (row >= ${count}u) { return; }
        if (row % ${field.config.numAreas}u == 0u) { logs[row] = ${Math.log(field.metadata.fixedGlottisArea)}; derivative[row] = 0.0; return; }
        let z = 0.35 * input[row] + 0.1;
        let tail = exp(-abs(z));
        var log1p = log(1.0 + tail);
        if (tail < 0.001) { log1p = tail * (1.0 - tail * 0.5 + tail * tail / 3.0); }
        let raw = max(z, 0.0) + log1p + 0.0001;
        let area = min(16.0, raw); logs[row] = log(area);
        var sigmoid = 1.0 / (1.0 + exp(-z));
        if (z < 0.0) { let ez = exp(z); sigmoid = ez / (1.0 + ez); }
        derivative[row] = select(0.35 * sigmoid / area, 0.0, raw >= 16.0);
      }`, [activations.at(-1), logs, derivative, ...(residual ? [baseline] : [])]);
    const firstDelta = await shader('RFF area cotangent', `
      @group(0) @binding(0) var<storage, read> cotangent: array<f32>;
      @group(0) @binding(1) var<storage, read> derivative: array<f32>;
      @group(0) @binding(2) var<storage, read_write> delta: array<f32>;
      @compute @workgroup_size(${WG}) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
        if (id.x < ${count}u) { delta[id.x] = cotangent[id.x] * derivative[id.x]; }
      }`, [cotangent, derivative, deltas.at(-1)]);
    const run = (pass, kernel, x, y = 1) => { pass.setPipeline(kernel.pipeline); pass.setBindGroup(0, kernel.bindGroup); pass.dispatchWorkgroups(Math.ceil(x / WG), y); };
    const check = () => {
      if (destroyed) throw new Error('The WebGPU field kernels have been destroyed.');
    };
    return {
      count, parameterCount: params.length,
      parameterBuffer, logAreaBuffer: logs, logsBuffer: logs,
      cotangentBuffer: cotangent, gradientBuffer: gradient,
      uploadParameters(values = params) {
        check();
        if (values.length !== params.length) throw new Error('Parameters must match the field layout.');
        device.queue.writeBuffer(parameterBuffer, 0, Float32Array.from(values));
        hasForward = false;
      },
      refreshEncoding() {
        check(); field.refreshEncoding();
        device.queue.writeBuffer(activations[0], 0, featuresFor(field));
        hasForward = false;
      },
      encodeForward(pass) {
        check();
        forward.forEach((kernel, i) => run(pass, kernel, count * field.layout[i].outputSize));
        run(pass, area, count); hasForward = true;
      },
      encodeBackward(pass) {
        check();
        if (!hasForward) throw new Error('Encode forward() before backward().');
        run(pass, firstDelta, count);
        for (let i = field.layout.length - 1; i >= 0; i--) {
          const l = field.layout[i], kernels = reverse[i];
          run(pass, kernels.partial, l.weightCount + l.biasCount, groups);
          run(pass, kernels.reduce, l.weightCount + l.biasCount);
          if (kernels.delta) run(pass, kernels.delta, count * l.inputSize);
        }
      },
      destroy() {
        if (destroyed) return;
        destroyed = true;
        for (const buffer of buffers) buffer.destroy();
      },
    };
  } catch (error) {
    for (const buffer of buffers) buffer.destroy();
    throw error;
  }
}

/** Readback/upload wrapper for CPU acoustic models and CPU Adam. The fallback
 * preserves the original field's API. Always await both forward and backward:
 * CPU results are synchronous, GPU results asynchronous.
 */
export async function createWebGpuRffField(field, {allowFallback = true, gpu = globalThis.navigator?.gpu} = {}) {
  const buffers = [];
  let device, kernels;
  const fallback = error => {
    for (const buffer of buffers) buffer.destroy();
    kernels?.destroy();
    device?.destroy();
    if (!allowFallback) throw error;
    return {...field, backend: 'javascript-f64', fallbackReason: error.message, destroy() {}};
  };
  try {
    if (!gpu) throw new Error('WebGPU is unavailable in this browser.');
    const adapter = await gpu.requestAdapter({powerPreference: 'high-performance'});
    if (!adapter) throw new Error('No WebGPU adapter is available.');
    device = await adapter.requestDevice();
    let lost = null, destroyed = false, busy = false, hasForward = false;
    device.lost.then(info => { lost = info.message || 'WebGPU device was lost.'; });
    kernels = await createWebGpuRffKernels(field, {device});
    const {count, parameterCount, logAreaBuffer, cotangentBuffer, gradientBuffer} = kernels;
    const staging = length => {
      const buffer = device.createBuffer({size: length * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST});
      buffers.push(buffer); return buffer;
    };
    const logStaging = staging(count), gradientStaging = staging(parameterCount);
    const read = async buffer => {
      await buffer.mapAsync(GPUMapMode.READ);
      const values = Float64Array.from(new Float32Array(buffer.getMappedRange()));
      buffer.unmap(); return values;
    };
    const check = () => {
      if (destroyed || lost) throw new Error(lost || 'The WebGPU field has been destroyed.');
      if (busy) throw new Error('Await the previous WebGPU field operation first.');
    };
    return {
      ...field, backend: 'webgpu-f32', adapterInfo: {vendor: adapter.info?.vendor, architecture: adapter.info?.architecture, device: adapter.info?.device, description: adapter.info?.description},
      serialize() { return {...field.serialize(), computeBackend: 'webgpu-f32'}; },
      forwardAt() { throw new Error('The experimental GPU backend supports the configured training grid only. Use sampleAreas() for continuous queries.'); },
      refreshEncoding() { check(); kernels.refreshEncoding(); hasForward = false; },
      async forward() {
        check(); busy = true;
        try {
          kernels.uploadParameters();
          const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
          kernels.encodeForward(pass); pass.end();
          encoder.copyBufferToBuffer(logAreaBuffer, 0, logStaging, 0, count * 4);
          device.queue.submit([encoder.finish()]);
          const result = await read(logStaging); hasForward = true; return result;
        } finally { busy = false; }
      },
      async backward(values) {
        check();
        if (!hasForward) throw new Error('Call forward() before backward().');
        if (values.length !== count) throw new Error('The cotangent must match the training batch.');
        busy = true;
        try {
          device.queue.writeBuffer(cotangentBuffer, 0, Float32Array.from(values));
          const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
          kernels.encodeBackward(pass); pass.end();
          encoder.copyBufferToBuffer(gradientBuffer, 0, gradientStaging, 0, parameterCount * 4);
          device.queue.submit([encoder.finish()]); return await read(gradientStaging);
        } finally { busy = false; }
      },
      destroy() {
        if (destroyed) return;
        destroyed = true;
        for (const buffer of buffers) buffer.destroy();
        kernels.destroy(); device.destroy();
      },
    };
  } catch (error) { return fallback(error); }
}
