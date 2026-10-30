/** Experimental batched float32 WebGPU version of Geometry::log_transfer_gradient.
 * Each frequency runs the same complex transmission-line recurrence and analytic
 * reverse pass as the Rust model. A second GPU pass eliminates per-frame gain.
 * The kernel factory can share its input/result buffers with a GPU training loop;
 * the convenience factory uploads log areas and reads the result back to the CPU.
 */
const N = 32, WG = 64;

function prepareTargets(prepared, length) {
  if (!Number.isFinite(length) || length < 5 || length > 40) throw new Error('Tract length must be between 5 and 40 cm.');
  if (!Array.isArray(prepared?.frames) || !prepared.frames.length) throw new Error('Acoustic target frames are required.');
  const frames = prepared.frames.length, offsets = new Uint32Array(frames + 1), spectra = [], frameIds = [];
  for (let f = 0; f < frames; f++) {
    const frame = prepared.frames[f];
    offsets[f] = frameIds.length;
    if (!frame.active) continue;
    if (!frame.frequencies?.length || frame.targetDb?.length !== frame.frequencies.length) throw new Error('Invalid acoustic target frame.');
    for (let k = 0; k < frame.frequencies.length; k++) {
      const frequency = frame.frequencies[k], target = frame.targetDb[k];
      if (!Number.isFinite(frequency) || frequency < 0 || !Number.isFinite(target)) throw new Error('Acoustic targets must be finite, with nonnegative frequencies.');
      spectra.push(frequency, target * Math.LN10 / 20); frameIds.push(f);
    }
  }
  offsets[frames] = frameIds.length;
  const activeFrames = prepared.frames.filter(frame => frame.active).length, frequencyCount = frameIds.length;
  if (!activeFrames) throw new Error('At least one active frame is required.');
  return {frames, activeFrames, frequencyCount, offsets, spectra, frameIds};
}

/** Encode acoustics on an existing device without any per-step host transfers.
 * `logAreas` is a frame-major float32 storage buffer with 32 log areas per frame.
 * `result` is frame-major float32, stride 34: [loss, gainDb, ...32 log-area grads].
 * Losses/gradients are per-frame and NOT averaged over activeFrames. Inactive
 * frames produce zeros. The caller owns the device and logAreas; destroy() only
 * releases buffers allocated by this factory. Call encode after writing logAreas.
 */
export async function createWebGpuAcousticKernels(prepared, {device, logAreas, length = 17.5, gainInvariant = true, includeTransfer = false} = {}) {
  const {frames, activeFrames, frequencyCount, offsets, spectra, frameIds} = prepareTargets(prepared, length);
  if (!device?.createBuffer || !logAreas || logAreas.size < frames * N * 4) throw new Error('Supply a WebGPU device and a log-area storage buffer large enough for all frames.');
  const buffers = [];
  let destroyed = false, lost = null;
  device.lost.then(info => { lost = info.message || 'WebGPU device was lost.'; });
  const storage = (size, data) => {
    const buffer = device.createBuffer({size: Math.max(4, size), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC});
    buffers.push(buffer); if (data) device.queue.writeBuffer(buffer, 0, data); return buffer;
  };
  const destroy = () => { if (destroyed) return; destroyed = true; for (const buffer of buffers) buffer.destroy(); };
  try {
    const geometry = storage(frames * N * 16);
    const targets = storage(spectra.length * 4, Float32Array.from(spectra));
    const frequencyFrames = storage(frameIds.length * 4, Uint32Array.from(frameIds));
    const frameOffsets = storage(offsets.byteLength, offsets);
    const response = storage(frequencyCount * (N + 1) * 4);
    const transfer = includeTransfer ? storage(frequencyCount * 2 * 4) : null;
    const result = storage(frames * (N + 2) * 4);
    const shader = async (label, code, resources) => {
      const module = device.createShaderModule({label, code}), info = await module.getCompilationInfo();
      const errors = info.messages.filter(message => message.type === 'error');
      if (errors.length) throw new Error(`${label}: ${errors.map(error => `${error.lineNum}: ${error.message}`).join('; ')}`);
      const pipeline = await device.createComputePipelineAsync({label, layout: 'auto', compute: {module, entryPoint: 'main'}});
      const bindGroup = device.createBindGroup({layout: pipeline.getBindGroupLayout(0), entries: resources.map((buffer, binding) => ({binding, resource: {buffer}}))});
      return {pipeline, bindGroup};
    };
    const geometryKernel = await shader('Acoustic geometry', `
      @group(0) @binding(0) var<storage, read> logs: array<f32>;
      @group(0) @binding(1) var<storage, read_write> geometry: array<vec4<f32>>;
      @compute @workgroup_size(${WG}) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
        let index = id.x; if (index >= ${frames * N}u) { return; }
        let a = exp(logs[index]); let pi = 3.141592653589793; let rho = 0.00114; let c = 34300.0;
        if (index % 32u == 31u) {
          geometry[index] = vec4<f32>(128.0*rho*c/(9.0*pi*pi*a), 8.0*rho/(3.0*pi*sqrt(pi*a)), 0.0, 0.0);
        } else {
          let li = ${length / N}; let sa = 2.0*sqrt(pi*a)*li;
          geometry[index] = vec4<f32>(sa/(2.0*a*a)*sqrt(rho*0.000186*0.5), rho*li/(2.0*a), sa, a*li/(rho*c*c));
        }
      }`, [logAreas, geometry]);
    const acousticKernel = await shader('Acoustic analytic reverse', `
      @group(0) @binding(0) var<storage, read> geometry: array<vec4<f32>>;
      @group(0) @binding(1) var<storage, read> targets: array<vec2<f32>>;
      @group(0) @binding(2) var<storage, read> frames: array<u32>;
      @group(0) @binding(3) var<storage, read_write> response: array<f32>;
      ${includeTransfer ? '@group(0) @binding(4) var<storage, read_write> transfer: array<vec2<f32>>;' : ''}
      fn mul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> { return vec2<f32>(a.x*b.x-a.y*b.y, a.x*b.y+a.y*b.x); }
      fn div(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> { return vec2<f32>(a.x*b.x+a.y*b.y, a.y*b.x-a.x*b.y)/dot(b,b); }
      struct Step { a: vec2<f32>, b: vec2<f32>, c: vec2<f32>, da: vec2<f32>, db: vec2<f32>, dc: vec2<f32>, k0: vec2<f32>, k1: vec2<f32> }
      @compute @workgroup_size(${WG}) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
        let index = id.x; if (index >= ${frequencyCount}u) { return; }
        let frame = frames[index]; let omega = 6.283185307179586*max(1e-8, targets[index].x);
        let rootOmega = sqrt(omega); let one = vec2<f32>(1.0,0.0);
        var k0 = one; var k1 = vec2<f32>(0.0); var steps: array<Step,31>;
        for (var i=0u; i<31u; i++) {
          let g = geometry[frame*32u+i];
          let za = vec2<f32>(g.x*rootOmega, omega*g.y); let dza = vec2<f32>(-1.5*za.x,-za.y);
          let yw = div(vec2<f32>(g.z,0.0),vec2<f32>(800.0,omega*2.1-84500.0/omega));
          let ya = vec2<f32>(0.0,omega*g.w); let y = yw+ya; let dy = 0.5*yw+ya;
          let a = one+mul(za,y); let b = -2.0*za-mul(mul(za,za),y); let c = -y;
          let da = mul(dza,y)+mul(za,dy);
          let db = -2.0*dza-2.0*mul(mul(za,dza),y)-mul(mul(za,za),dy); let dc = -dy;
          steps[i] = Step(a,b,c,da,db,dc,k0,k1);
          let next0 = mul(a,k0)+mul(b,k1); k1 = mul(c,k0)+mul(a,k1); k0 = next0;
        }
        let lip = geometry[frame*32u+31u]; let q = omega*lip.y; let radiationDen = vec2<f32>(lip.x,q);
        let zr = div(vec2<f32>(0.0,lip.x*q),radiationDen);
        let dzr = div(vec2<f32>(0.0,-1.5*lip.x*q)-mul(zr,vec2<f32>(-lip.x,-0.5*q)),radiationDen);
        let den = k0-mul(k1,zr); let h = div(zr,den);
        ${includeTransfer ? 'transfer[index] = h;' : ''}
        response[index*33u] = 0.5*log(dot(h,h))-targets[index].y;
        response[index*33u+32u] = mul(div(one,zr)+div(k1,den),dzr).x;
        var g0 = -div(one,den); var g1 = div(zr,den);
        for (var j=31u; j>0u; j--) {
          let i = j-1u; let s = steps[i];
          response[index*33u+i+1u] = (mul(g0,mul(s.da,s.k0)+mul(s.db,s.k1))+mul(g1,mul(s.dc,s.k0)+mul(s.da,s.k1))).x;
          let next0 = mul(g0,s.a)+mul(g1,s.c); g1 = mul(g0,s.b)+mul(g1,s.a); g0 = next0;
        }
      }`, [geometry, targets, frequencyFrames, response, ...(transfer ? [transfer] : [])]);
    const reductionKernel = await shader('Acoustic objective reduction', `
      const GAIN_INVARIANT: bool = ${gainInvariant ? 'true' : 'false'};
      @group(0) @binding(0) var<storage, read> response: array<f32>;
      @group(0) @binding(1) var<storage, read> offsets: array<u32>;
      @group(0) @binding(2) var<storage, read_write> result: array<f32>;
      @compute @workgroup_size(${WG}) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
        let index=id.x; if(index>=${frames * 33}u){return;}
        let frame=index/33u; let parameter=index%33u; let start=offsets[frame]; let end=offsets[frame+1u];
        var mean=0.0; var jacobianMean=0.0; var covariance=0.0;
        for(var k=start;k<end;k++) {
          let residual=response[k*33u];
          if (GAIN_INVARIANT) {
            let delta=residual-mean; let weight=1.0/f32(k-start+1u); mean+=delta*weight;
            if(parameter==32u) { covariance+=delta*(residual-mean); }
            else { let jacobian=response[k*33u+parameter+1u]; jacobianMean+=(jacobian-jacobianMean)*weight; covariance+=delta*(jacobian-jacobianMean); }
          } else {
            if(parameter==32u) { covariance+=residual*residual; }
            else { covariance+=residual*response[k*33u+parameter+1u]; }
          }
        }
        let scale=1.0/max(1.0,f32(end-start));
        if(parameter==32u) { result[frame*34u]=covariance*scale; result[frame*34u+1u]=-mean*8.685889638065037; }
        else { result[frame*34u+parameter+2u]=2.0*scale*covariance; }
      }`, [response, frameOffsets, result]);
    const run = (pass, kernel, count) => { pass.setPipeline(kernel.pipeline); pass.setBindGroup(0, kernel.bindGroup); pass.dispatchWorkgroups(Math.ceil(count / WG)); };
    return {
      backend: 'webgpu-acoustic-f32', frames, activeFrames, frequencyCount, result, response, transfer,
      encode(pass) {
        if (destroyed || lost) throw new Error(lost || 'The WebGPU acoustic model has been destroyed.');
        run(pass, geometryKernel, frames * N); run(pass, acousticKernel, frequencyCount); run(pass, reductionKernel, frames * 33);
      },
      destroy,
    };
  } catch (error) { destroy(); throw error; }
}

export async function createWebGpuAcoustic(prepared, {length = 17.5, gpu = globalThis.navigator?.gpu} = {}) {
  if (!gpu) throw new Error('WebGPU is unavailable in this browser.');
  const {frames} = prepareTargets(prepared, length);
  const adapter = await gpu.requestAdapter({powerPreference: 'high-performance'});
  if (!adapter) throw new Error('No WebGPU adapter is available.');
  const device = await adapter.requestDevice(), buffers = [];
  let kernels, destroyed = false, busy = false, lost = null;
  device.lost.then(info => { lost = info.message || 'WebGPU device was lost.'; });
  const destroy = () => {
    if (destroyed) return;
    destroyed = true; kernels?.destroy(); for (const buffer of buffers) buffer.destroy(); device.destroy();
  };
  try {
    const logAreas = device.createBuffer({size: frames * N * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST});
    buffers.push(logAreas);
    kernels = await createWebGpuAcousticKernels(prepared, {device, logAreas, length});
    const staging = device.createBuffer({size: frames * (N + 2) * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST});
    buffers.push(staging);
    const {activeFrames, frequencyCount, result} = kernels;
    return {
      backend: kernels.backend, frames, activeFrames, frequencyCount,
      adapterInfo: {vendor: adapter.info?.vendor, architecture: adapter.info?.architecture, device: adapter.info?.device, description: adapter.info?.description},
      async evaluate(values) {
        if (destroyed || lost) throw new Error(lost || 'The WebGPU acoustic model has been destroyed.');
        if (busy) throw new Error('Await the previous WebGPU acoustic operation first.');
        if (values.length !== frames * N || !values.every(value => Number.isFinite(value) && Math.exp(value) > 0 && Math.exp(value) <= 100)) throw new Error('Supply 32 finite log areas per frame, with areas in (0, 100].');
        busy = true;
        try {
          device.queue.writeBuffer(logAreas, 0, Float32Array.from(values));
          const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
          kernels.encode(pass);
          pass.end(); encoder.copyBufferToBuffer(result, 0, staging, 0, frames * 34 * 4); device.queue.submit([encoder.finish()]);
          await staging.mapAsync(GPUMapMode.READ);
          const packed = new Float32Array(staging.getMappedRange()).slice(); staging.unmap();
          const gradient = new Float64Array(frames * N), gainsDb = new Float64Array(frames), frameLosses = new Float64Array(frames);
          let dataLoss = 0;
          for (let f = 0; f < frames; f++) {
            frameLosses[f] = packed[f * 34]; dataLoss += frameLosses[f] / activeFrames; gainsDb[f] = packed[f * 34 + 1];
            for (let i = 0; i < N; i++) gradient[f * N + i] = packed[f * 34 + i + 2] / activeFrames;
          }
          if (!Number.isFinite(dataLoss) || !gradient.every(Number.isFinite)) throw new Error('Float32 acoustic computation produced a nonfinite result.');
          return {dataLoss, gradient, gainsDb, frameLosses};
        } finally { busy = false; }
      },
      destroy,
    };
  } catch (error) { destroy(); throw error; }
}
