/** Batched complex FFT used by the faithful waveform simulator and STFT loss.
 * The float32 path uses one workgroup per transform. Both arithmetic modes
 * stay within WebGPU's baseline 16 KiB of workgroup memory; compensated 2048
 * transforms use two 1024 transforms followed by a merge.
 * Buffers contain [real, imaginary] float32 pairs in batch-major order.
 * With inputFormat:'complex-pairs', double-single input instead contains
 * [realHi, realLo, imagHi, imagLo]; output always uses float32 complex pairs.
 */
export async function createGpuFft(device, {size, batchCount, inverse = false, twiddleMode = 'shader', arithmetic = 'float32', inputFormat = 'complex32'}) {
  if (![256, 512, 1024, 2048].includes(size)) throw new RangeError('FFT size must be 256, 512, 1024, or 2048.');
  if (!Number.isInteger(batchCount) || batchCount < 1) throw new RangeError('FFT batch count must be a positive integer.');
  if (!['shader', 'table'].includes(twiddleMode)) throw new RangeError('Unknown FFT twiddle mode.');
  if (!['float32', 'double-single'].includes(arithmetic)) throw new RangeError('Unknown FFT arithmetic.');
  if (!['complex32', 'complex-pairs'].includes(inputFormat)) throw new RangeError('Unknown FFT input format.');
  if (inputFormat === 'complex-pairs' && arithmetic !== 'double-single') throw new RangeError('Complex-pairs input requires double-single FFT arithmetic.');
  if (arithmetic === 'double-single') return createDoubleSingleFft(device, {size, batchCount, inverse, inputFormat});
  const maximum = device.limits.maxComputeWorkgroupsPerDimension;
  const workgroupsX = Math.min(batchCount, maximum);
  const workgroupsY = Math.ceil(batchCount / workgroupsX);
  if (workgroupsY > maximum) throw new RangeError('FFT batch count exceeds WebGPU dispatch limits.');
  if (size * 8 > device.limits.maxComputeWorkgroupStorageSize) throw new RangeError('FFT exceeds available workgroup storage.');
  if (256 > device.limits.maxComputeInvocationsPerWorkgroup || 256 > device.limits.maxComputeWorkgroupSizeX) {
    throw new RangeError('FFT requires 256 compute invocations per workgroup.');
  }
  const label = `faithful ${inverse ? 'inverse ' : ''}FFT ${size} × ${batchCount}`;
  const code = `
@group(0) @binding(0) var<storage, read> source: array<vec2f>;
@group(0) @binding(1) var<storage, read_write> destination: array<vec2f>;
${twiddleMode === 'table' ? '@group(0) @binding(2) var<storage, read> twiddles: array<vec2f>;' : ''}
var<workgroup> values: array<vec2f, ${size}>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) group: vec3u, @builtin(local_invocation_id) lane: vec3u) {
  let batch = group.y * ${workgroupsX}u + group.x;
  if (batch >= ${batchCount}u) { return; }
  let offset = batch * ${size}u;
  for (var i = lane.x; i < ${size}u; i += 256u) {
    let reverse = reverseBits(i) >> ${32 - Math.log2(size)}u;
    values[i] = source[offset + reverse];
  }
  workgroupBarrier();
  for (var span = 2u; span <= ${size}u; span *= 2u) {
    let half = span / 2u;
    for (var butterfly = lane.x; butterfly < ${size / 2}u; butterfly += 256u) {
      let position = butterfly % half;
      let first = (butterfly / half) * span + position;
      let second = first + half;
      ${twiddleMode === 'table' ? `let tableValue = twiddles[position * (${size}u / span)];
      let phase = vec2f(tableValue.x, ${inverse ? '-' : ''}tableValue.y);` : `let angle = ${inverse ? '' : '-'}6.283185307179586 * f32(position) / f32(span);
      let phase = vec2f(cos(angle), sin(angle));`}
      let even = values[first];
      let odd = values[second];
      let rotated = vec2f(phase.x * odd.x - phase.y * odd.y, phase.x * odd.y + phase.y * odd.x);
      values[first] = even + rotated;
      values[second] = even - rotated;
    }
    workgroupBarrier();
  }
  for (var i = lane.x; i < ${size}u; i += 256u) {
    destination[offset + i] = values[i]${inverse ? ` / ${size}.0` : ''};
  }
}`;
  const module = device.createShaderModule({label, code});
  const compilation = await module.getCompilationInfo();
  const errors = compilation.messages.filter(message => message.type === 'error');
  if (errors.length) throw new Error(`${label}: ${errors.map(message => `${message.lineNum}:${message.linePos} ${message.message}`).join('\n')}`);
  const pipeline = await device.createComputePipelineAsync({label, layout: 'auto', compute: {module, entryPoint: 'main'}});
  let twiddleBuffer;
  if (twiddleMode === 'table') {
    // Constants are evaluated once in float64 and rounded once to float32.
    // Exact quarter turns avoid the spurious real residual from cos(pi/2).
    const values = new Float32Array(size);
    for (let k = 0; k < size / 2; k++) {
      values[2 * k] = k === size / 4 ? 0 : Math.cos(-2 * Math.PI * k / size);
      values[2 * k + 1] = k === 0 ? 0 : Math.sin(-2 * Math.PI * k / size);
    }
    twiddleBuffer = device.createBuffer({size: values.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, label: `${label} twiddles`});
    device.queue.writeBuffer(twiddleBuffer, 0, values);
  }
  const cache = new WeakMap();
  const requiredBytes = batchCount * size * 8;
  function bindGroup(input, output) {
    if (input === output) throw new Error('FFT input and output must be distinct buffers.');
    if (input.size < requiredBytes || output.size < requiredBytes) throw new RangeError('FFT input/output buffer is too small.');
    let byOutput = cache.get(input);
    if (!byOutput) cache.set(input, byOutput = new WeakMap());
    let binding = byOutput.get(output);
    if (!binding) {
      binding = device.createBindGroup({label, layout: pipeline.getBindGroupLayout(0), entries: [
        {binding: 0, resource: {buffer: input, offset: 0, size: requiredBytes}},
        {binding: 1, resource: {buffer: output, offset: 0, size: requiredBytes}},
        ...(twiddleBuffer ? [{binding: 2, resource: {buffer: twiddleBuffer}}] : []),
      ]});
      byOutput.set(output, binding);
    }
    return binding;
  }
  return {
    size, batchCount, inverse, twiddleMode, arithmetic, inputFormat, scratchBytes: 0,
    encode(encoder, inputComplexBuffer, outputComplexBuffer) {
      const binding = bindGroup(inputComplexBuffer, outputComplexBuffer);
      const pass = encoder.beginComputePass({label});
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, binding);
      pass.dispatchWorkgroups(workgroupsX, workgroupsY);
      pass.end();
    },
    destroy() { twiddleBuffer?.destroy(); },
    dispose() { twiddleBuffer?.destroy(); },
  };
}

// A float32 pair carries each real value. Products retain their rounding
// residual with fma, and additions retain both terms through two-sum. The
// interfaces remain float32; compensation applies to the FFT butterflies.
const DOUBLE_SINGLE_WGSL = `
@group(0) @binding(3) var<uniform> roundingMask: vec4u;
// The uploaded mask is zero. The opaque bit operation preserves a float32
// rounding boundary that fast-math otherwise reassociates out of two-sum.
fn round32(value: f32) -> f32 {
  return bitcast<f32>(bitcast<u32>(value) ^ roundingMask.x);
}
fn dsAdd(a: vec2f, b: vec2f) -> vec2f {
  let sum = round32(a.x + b.x);
  let virtualB = round32(sum - a.x);
  let error = round32(round32(a.x - round32(sum - virtualB)) + round32(b.x - virtualB));
  let residual = round32(error + round32(a.y + b.y));
  let high = round32(sum + residual);
  return vec2f(high, round32(residual - round32(high - sum)));
}
fn dsMul(a: vec2f, b: vec2f) -> vec2f {
  let product = round32(a.x * b.x);
  let error = fma(a.x, b.x, -product);
  let cross = round32(round32(a.x * b.y) + round32(a.y * b.x));
  let residual = round32(round32(error + cross) + round32(a.y * b.y));
  let high = round32(product + residual);
  return vec2f(high, round32(residual - round32(high - product)));
}
fn complexAdd(a: vec4f, b: vec4f) -> vec4f {
  return vec4f(dsAdd(a.xy, b.xy), dsAdd(a.zw, b.zw));
}
fn complexMul(a: vec4f, b: vec4f) -> vec4f {
  return vec4f(dsAdd(dsMul(a.xy, b.xy), -dsMul(a.zw, b.zw)),
    dsAdd(dsMul(a.xy, b.zw), dsMul(a.zw, b.xy)));
}
`;

const precisionChecks = new WeakMap();

/** Verify the residual arithmetic once on each actual device/compiler pair. */
export function checkGpuFloatPairPrecision(device) {
  if (precisionChecks.has(device)) return precisionChecks.get(device);
  const check = (async () => {
    const buffers = [];
    const makeBuffer = (size, usage, label) => {
      const buffer = device.createBuffer({size, usage, label}); buffers.push(buffer); return buffer;
    };
    try {
      const source = makeBuffer(32, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, 'float-pair probe input');
      const destination = makeBuffer(32, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC, 'float-pair probe output');
      const roundingMask = makeBuffer(16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, 'float-pair probe rounding mask');
      const staging = makeBuffer(32, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST, 'float-pair probe readback');
      device.queue.writeBuffer(source, 0, Float32Array.from([1 + 2 ** -23, 1 + 2 ** -23, 1, 2 ** -24, 1, 2 ** -24, -1, 0]));
      device.queue.writeBuffer(roundingMask, 0, new Uint32Array(4));
      const module = device.createShaderModule({label: 'float-pair precision self-check', code: `
@group(0) @binding(0) var<storage, read> source: array<vec2f>;
@group(0) @binding(1) var<storage, read_write> destination: array<vec2f>;
${DOUBLE_SINGLE_WGSL}
@compute @workgroup_size(1)
fn main() {
  let a = source[0].x; let b = source[0].y;
  destination[0] = dsMul(vec2f(a, 0.0), vec2f(b, 0.0));
  destination[1] = dsAdd(vec2f(source[1].x, 0.0), vec2f(source[1].y, 0.0));
  destination[2] = dsAdd(source[2], source[3]);
  let product = round32(a * b);
  destination[3] = vec2f(product, fma(a, b, -product));
}`});
      const compilation = await module.getCompilationInfo();
      const errors = compilation.messages.filter(message => message.type === 'error');
      if (errors.length) throw new Error(errors.map(message => message.message).join('; '));
      const pipeline = await device.createComputePipelineAsync({label: 'float-pair precision self-check', layout: 'auto', compute: {module, entryPoint: 'main'}});
      const binding = device.createBindGroup({layout: pipeline.getBindGroupLayout(0), entries: [
        {binding: 0, resource: {buffer: source}}, {binding: 1, resource: {buffer: destination}},
        {binding: 3, resource: {buffer: roundingMask}},
      ]});
      const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
      pass.setPipeline(pipeline); pass.setBindGroup(0, binding); pass.dispatchWorkgroups(1); pass.end();
      encoder.copyBufferToBuffer(destination, 0, staging, 0, 32);
      device.queue.submit([encoder.finish()]); await staging.mapAsync(GPUMapMode.READ);
      const actual = Array.from(new Float32Array(staging.getMappedRange())); staging.unmap();
      const product = 1 + 2 ** -22;
      const expected = [product, 2 ** -46, 1, 2 ** -24, 2 ** -24, 0, product, 2 ** -46];
      const labels = ['double-single product', 'two-sum', 'cancellation', 'fma residual'];
      const cases = labels.map((name, i) => ({name, expected: expected.slice(2 * i, 2 * i + 2), actual: actual.slice(2 * i, 2 * i + 2),
        passed: actual[2 * i] === expected[2 * i] && actual[2 * i + 1] === expected[2 * i + 1]}));
      const failures = cases.filter(item => !item.passed);
      if (failures.length) throw new Error(`Required float32 residuals were not preserved (${failures.map(item => item.name).join(', ')}).`);
      return Object.freeze({passed: true, cases});
    } catch (error) {
      throw new Error(`This WebGPU device/compiler failed the compensated arithmetic self-check. Use the CPU reference backend. ${error.message}`, {cause: error});
    } finally { for (const buffer of buffers) buffer.destroy(); }
  })();
  precisionChecks.set(device, check);
  return check;
}

async function createDoubleSingleFft(device, {size, batchCount, inverse, inputFormat}) {
  // 2048 complex pairs need 32 KiB; split into even/odd 1024 transforms and
  // merge in a second dispatch, preserving WebGPU's 16 KiB baseline limit.
  const split = size === 2048, transformSize = split ? 1024 : size;
  const pairedInput = inputFormat === 'complex-pairs';
  const transformCount = batchCount * (split ? 2 : 1);
  const maximum = device.limits.maxComputeWorkgroupsPerDimension;
  const groupsX = Math.min(transformCount, maximum), groupsY = Math.ceil(transformCount / groupsX);
  const mergeGroupCount = Math.ceil(batchCount * transformSize / 256);
  const mergeGroupsX = Math.min(mergeGroupCount, maximum), mergeGroupsY = Math.ceil(mergeGroupCount / mergeGroupsX);
  if (groupsY > maximum) throw new RangeError('FFT batch count exceeds WebGPU dispatch limits.');
  if (split && mergeGroupsY > maximum) throw new RangeError('FFT merge exceeds WebGPU dispatch limits.');
  if (transformSize * 16 > device.limits.maxComputeWorkgroupStorageSize) throw new RangeError('Double-single FFT exceeds available workgroup storage.');
  if (256 > device.limits.maxComputeInvocationsPerWorkgroup || 256 > device.limits.maxComputeWorkgroupSizeX) {
    throw new RangeError('FFT requires 256 compute invocations per workgroup.');
  }
  const precisionCheck = await checkGpuFloatPairPrecision(device);
  const label = `faithful double-single ${inverse ? 'inverse ' : ''}FFT ${size} × ${batchCount}`;
  const values = new Float32Array(size * 2);
  for (let k = 0; k < size / 2; k++) {
    const real = k === size / 4 ? 0 : Math.cos(-2 * Math.PI * k / size);
    const imaginary = k === 0 ? 0 : Math.sin(-2 * Math.PI * k / size);
    values[4 * k] = real; values[4 * k + 1] = real - values[4 * k];
    values[4 * k + 2] = imaginary; values[4 * k + 3] = imaginary - values[4 * k + 2];
  }
  const twiddles = device.createBuffer({label: `${label} split twiddles`, size: values.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST});
  device.queue.writeBuffer(twiddles, 0, values);
  const roundingMask = device.createBuffer({label: `${label} rounding mask`, size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST});
  device.queue.writeBuffer(roundingMask, 0, new Uint32Array(4));
  const scratch = split ? device.createBuffer({label: `${label} intermediate pairs`, size: batchCount * size * 16,
    usage: GPUBufferUsage.STORAGE}) : undefined;
  async function compile(code, suffix) {
    const module = device.createShaderModule({label: `${label} ${suffix}`, code});
    const compilation = await module.getCompilationInfo();
    const errors = compilation.messages.filter(message => message.type === 'error');
    if (errors.length) throw new Error(`${label}: ${errors.map(message => `${message.lineNum}:${message.linePos} ${message.message}`).join('\n')}`);
    return device.createComputePipelineAsync({label: `${label} ${suffix}`, layout: 'auto', compute: {module, entryPoint: 'main'}});
  }
  let transformPipeline, mergePipeline;
  try {
    transformPipeline = await compile(`
@group(0) @binding(0) var<storage, read> source: array<${pairedInput ? 'vec4f' : 'vec2f'}>;
@group(0) @binding(1) var<storage, read_write> destination: array<${split ? 'vec4f' : 'vec2f'}>;
@group(0) @binding(2) var<storage, read> twiddles: array<vec4f>;
var<workgroup> values: array<vec4f, ${transformSize}>;
${DOUBLE_SINGLE_WGSL}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) group: vec3u, @builtin(local_invocation_id) lane: vec3u) {
  let transform = group.y * ${groupsX}u + group.x;
  if (transform >= ${transformCount}u) { return; }
  ${split ? `let batch = transform / 2u;
  let parity = transform % 2u;` : 'let batch = transform;'}
  for (var i = lane.x; i < ${transformSize}u; i += 256u) {
    let reverse = reverseBits(i) >> ${32 - Math.log2(transformSize)}u;
    let inputValue = source[batch * ${size}u + ${split ? '2u * reverse + parity' : 'reverse'}];
    values[i] = ${pairedInput ? 'inputValue' : 'vec4f(inputValue.x, 0.0, inputValue.y, 0.0)'};
  }
  workgroupBarrier();
  for (var span = 2u; span <= ${transformSize}u; span *= 2u) {
    let half = span / 2u;
    for (var butterfly = lane.x; butterfly < ${transformSize / 2}u; butterfly += 256u) {
      let position = butterfly % half;
      let first = (butterfly / half) * span + position;
      let second = first + half;
      let tableValue = twiddles[position * (${size}u / span)];
      let phase = vec4f(tableValue.xy, ${inverse ? '-' : ''}tableValue.zw);
      let even = values[first];
      let rotated = complexMul(phase, values[second]);
      values[first] = complexAdd(even, rotated);
      values[second] = complexAdd(even, -rotated);
    }
    workgroupBarrier();
  }
  for (var i = lane.x; i < ${transformSize}u; i += 256u) {
    ${split ? `destination[transform * ${transformSize}u + i] = values[i];` : `let value = values[i];
    destination[batch * ${size}u + i] = vec2f(value.x + value.y, value.z + value.w)${inverse ? ` / ${size}.0` : ''};`}
  }
}`, 'butterflies');
    if (split) mergePipeline = await compile(`
@group(0) @binding(0) var<storage, read> source: array<vec4f>;
@group(0) @binding(1) var<storage, read_write> destination: array<vec2f>;
@group(0) @binding(2) var<storage, read> twiddles: array<vec4f>;
${DOUBLE_SINGLE_WGSL}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) group: vec3u, @builtin(local_invocation_id) lane: vec3u) {
  let index = (group.y * ${mergeGroupsX}u + group.x) * 256u + lane.x;
  if (index >= ${batchCount * transformSize}u) { return; }
  let batch = index / ${transformSize}u;
  let bin = index % ${transformSize}u;
  let tableValue = twiddles[bin];
  let phase = vec4f(tableValue.xy, ${inverse ? '-' : ''}tableValue.zw);
  let even = source[batch * ${size}u + bin];
  let rotated = complexMul(phase, source[batch * ${size}u + ${transformSize}u + bin]);
  let first = complexAdd(even, rotated);
  let second = complexAdd(even, -rotated);
  destination[batch * ${size}u + bin] = vec2f(first.x + first.y, first.z + first.w)${inverse ? ` / ${size}.0` : ''};
  destination[batch * ${size}u + ${transformSize}u + bin] = vec2f(second.x + second.y, second.z + second.w)${inverse ? ` / ${size}.0` : ''};
}`, 'merge');
  } catch (error) { twiddles.destroy(); roundingMask.destroy(); scratch?.destroy(); throw error; }
  const requiredBytes = batchCount * size * 8, requiredInputBytes = requiredBytes * (pairedInput ? 2 : 1), cache = new WeakMap();
  function bind(input, output) {
    if (input === output) throw new Error('FFT input and output must be distinct buffers.');
    if (input.size < requiredInputBytes || output.size < requiredBytes) throw new RangeError('FFT input/output buffer is too small.');
    let byOutput = cache.get(input);
    if (!byOutput) cache.set(input, byOutput = new WeakMap());
    let groups = byOutput.get(output);
    if (!groups) {
      const group = (pipeline, source, destination, sourceBytes, destinationBytes) => device.createBindGroup({label,
        layout: pipeline.getBindGroupLayout(0), entries: [
          {binding: 0, resource: {buffer: source, size: sourceBytes}},
          {binding: 1, resource: {buffer: destination, size: destinationBytes}},
          {binding: 2, resource: {buffer: twiddles}},
          {binding: 3, resource: {buffer: roundingMask}},
        ]});
      groups = [group(transformPipeline, input, scratch ?? output, requiredInputBytes, split ? requiredBytes * 2 : requiredBytes)];
      if (split) groups.push(group(mergePipeline, scratch, output, requiredBytes * 2, requiredBytes));
      byOutput.set(output, groups);
    }
    return groups;
  }
  const destroy = () => { twiddles.destroy(); roundingMask.destroy(); scratch?.destroy(); };
  return {size, batchCount, inverse, twiddleMode: 'table', arithmetic: 'double-single', inputFormat,
    scratchBytes: scratch?.size ?? 0, precisionCheck,
    encode(encoder, input, output) {
      const groups = bind(input, output), pass = encoder.beginComputePass({label});
      pass.setPipeline(transformPipeline); pass.setBindGroup(0, groups[0]); pass.dispatchWorkgroups(groupsX, groupsY);
      if (split) {
        pass.setPipeline(mergePipeline); pass.setBindGroup(0, groups[1]);
        pass.dispatchWorkgroups(mergeGroupsX, mergeGroupsY);
      }
      pass.end();
    }, destroy, dispose: destroy};
}
