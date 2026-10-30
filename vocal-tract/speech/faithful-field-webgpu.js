/** Tiled float32 RFF network; layer widths come from the loaded NNX layout.
 * All trainable tensors, their gradients, Adam moments and updates stay on GPU.
 */
import {faithfulLearningRate} from './faithful-field.js?v=fixed-grid';

export async function createFaithfulGpuField(device, field, {iterations = 500} = {}) {
  const owned = [], count = field.config.frameTimes.length * 32, parameterCount = field.parameters.length;
  const storage = (length, data) => {
    const buffer = device.createBuffer({size: Math.max(4, length * 4), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST});
    owned.push(buffer); if (data) device.queue.writeBuffer(buffer, 0, data); return buffer;
  };
  const compile = async (label, code, buffers) => {
    const module = device.createShaderModule({label, code});
    const errors = (await module.getCompilationInfo()).messages.filter(m => m.type === 'error');
    if (errors.length) throw new Error(`${label}: ${errors.map(m => `${m.lineNum}: ${m.message}`).join('; ')}`);
    const pipeline = await device.createComputePipelineAsync({label, layout: 'auto', compute: {module, entryPoint: 'main'}});
    const bindGroup = device.createBindGroup({layout: pipeline.getBindGroupLayout(0), entries: buffers.map((buffer, binding) => ({binding, resource: {buffer}}))});
    return {pipeline, bindGroup};
  };
  const run = (encoder, kernel, x, y = 1, z = 1) => {
    const pass = encoder.beginComputePass(); pass.setPipeline(kernel.pipeline); pass.setBindGroup(0, kernel.bindGroup); pass.dispatchWorkgroups(x, y, z); pass.end();
  };
  try {
    const features = new Float32Array(count * field.config.frequencies * 2), f = Math.fround;
    for (let row = 0; row < count; row++) for (let k = 0; k < field.config.frequencies; k++) {
      const frame = Math.floor(row / 32), frames = field.config.frameTimes.length;
      const t = frames > 1 ? (frame === frames - 1 ? 1 : f(frame * f(1 / (frames - 1)))) : 0, x = row % 32 === 31 ? 1 : f((row % 32) * f(1 / 31));
      const phase = f(f(2 * Math.PI) * f(f(t * field.encoding[k]) + f(x * field.encoding[field.config.frequencies + k])));
      features[row * 2 * field.config.frequencies + k] = Math.cos(phase);
      features[row * 2 * field.config.frequencies + field.config.frequencies + k] = Math.sin(phase);
    }
    const parameterBuffer = storage(parameterCount, Float32Array.from(field.parameters));
    const activations = [storage(features.length, features), ...field.layout.map(l => storage(count * l.outputSize))];
    const deltas = field.layout.map(l => storage(count * l.outputSize));
    const logAreaBuffer = storage(count), areaBuffer = storage(count), derivative = storage(count);
    const cotangentBuffer = storage(count), gradientBuffer = storage(parameterCount);
    const groups = Math.min(32, Math.max(1, Math.ceil(count / 128))), rowsPerGroup = Math.ceil(count / groups);
    const forward = [], reverse = [];
    for (let i = 0; i < field.layout.length; i++) {
      const l = field.layout[i], W = l.inputSize, O = l.outputSize;
      const declarations = `@group(0) @binding(0) var<storage,read> p:array<f32>;
        @group(0) @binding(1) var<storage,read> x:array<f32>;
        @group(0) @binding(2) var<storage,read_write> y:array<f32>;`;
      const code = O === 1 ? `${declarations}
        @compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id:vec3<u32>) {
          let row=id.x; if(row>=${count}u){return;} var v=p[${l.biasOffset}u];
          for(var k=0u;k<${W}u;k++){v+=x[row*${W}u+k]*p[${l.weightOffset}u+k];} y[row]=v;
        }` : `${declarations}
        var<workgroup> a:array<f32,256>; var<workgroup> b:array<f32,256>;
        @compute @workgroup_size(16,16) fn main(@builtin(workgroup_id) wg:vec3<u32>,@builtin(local_invocation_id) lid:vec3<u32>){
          let row=wg.y*16u+lid.y; let col=wg.x*16u+lid.x; let local=lid.y*16u+lid.x; var v=0.0;
          for(var tile=0u;tile<${W}u;tile+=16u){
            a[local]=0.0; b[local]=0.0;
            if(row<${count}u && tile+lid.x<${W}u){a[local]=x[row*${W}u+tile+lid.x];}
            if(col<${O}u && tile+lid.y<${W}u){b[local]=p[${l.weightOffset}u+col*${W}u+tile+lid.y];}
            workgroupBarrier(); for(var k=0u;k<16u;k++){v+=a[lid.y*16u+k]*b[k*16u+lid.x];} workgroupBarrier();
          }
          if(row<${count}u && col<${O}u){y[row*${O}u+col]=tanh(v+p[${l.biasOffset}u+col]);}
        }`;
      forward.push({kernel: await compile(`Faithful RFF layer ${i}`, code, [parameterBuffer, activations[i], activations[i + 1]]), x: O === 1 ? Math.ceil(count / 64) : Math.ceil(O / 16), y: O === 1 ? 1 : Math.ceil(count / 16)});
      const partialBuffer = storage(groups * (l.weightCount + O));
      const partial = await compile(`Faithful RFF gradient ${i}`, `
        @group(0) @binding(0) var<storage,read> x:array<f32>;
        @group(0) @binding(1) var<storage,read> d:array<f32>;
        @group(0) @binding(2) var<storage,read_write> partial:array<f32>;
        var<workgroup> a:array<f32,256>; var<workgroup> b:array<f32,256>;
        @compute @workgroup_size(16,16) fn main(@builtin(workgroup_id) wg:vec3<u32>,@builtin(local_invocation_id) lid:vec3<u32>){
          let out=wg.y*16u+lid.y; let col=wg.x*16u+lid.x; let local=lid.y*16u+lid.x; var v=0.0;
          let start=wg.z*${rowsPerGroup}u; let end=min(start+${rowsPerGroup}u,${count}u);
          for(var tile=start;tile<end;tile+=16u){
            a[local]=0.0; b[local]=0.0;
            if(out<${O}u && tile+lid.x<end){a[local]=d[(tile+lid.x)*${O}u+out];}
            if(col<=${W}u && tile+lid.y<end){
              b[local]=1.0; if(col<${W}u){b[local]=x[(tile+lid.y)*${W}u+col];}
            }
            workgroupBarrier(); for(var k=0u;k<16u;k++){v+=a[lid.y*16u+k]*b[k*16u+lid.x];} workgroupBarrier();
          }
          if(out<${O}u && col<=${W}u){
            let p=select(out*${W}u+col,${l.weightCount}u+out,col==${W}u);
            partial[wg.z*${l.weightCount + O}u+p]=v;
          }
        }`, [activations[i], deltas[i], partialBuffer]);
      const reduce = await compile(`Faithful RFF reduce ${i}`, `
        @group(0) @binding(0) var<storage,read> partial:array<f32>;
        @group(0) @binding(1) var<storage,read_write> g:array<f32>;
        @compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id:vec3<u32>){
          let p=id.x;if(p>=${l.weightCount + O}u){return;}var v=0.0;
          for(var block=0u;block<${groups}u;block++){v+=partial[block*${l.weightCount + O}u+p];}g[${l.weightOffset}u+p]=v;
        }`, [partialBuffer, gradientBuffer]);
      let delta;
      if (i > 0) delta = await compile(`Faithful RFF reverse ${i}`, `
        @group(0) @binding(0) var<storage,read> p:array<f32>;
        @group(0) @binding(1) var<storage,read> d:array<f32>;
        @group(0) @binding(2) var<storage,read> x:array<f32>;
        @group(0) @binding(3) var<storage,read_write> previous:array<f32>;
        var<workgroup> a:array<f32,256>; var<workgroup> b:array<f32,256>;
        @compute @workgroup_size(16,16) fn main(@builtin(workgroup_id) wg:vec3<u32>,@builtin(local_invocation_id) lid:vec3<u32>){
          let row=wg.y*16u+lid.y;let col=wg.x*16u+lid.x;let local=lid.y*16u+lid.x;var v=0.0;
          for(var tile=0u;tile<${O}u;tile+=16u){
            a[local]=0.0;b[local]=0.0;
            if(row<${count}u&&tile+lid.x<${O}u){a[local]=d[row*${O}u+tile+lid.x];}
            if(col<${W}u&&tile+lid.y<${O}u){b[local]=p[${l.weightOffset}u+(tile+lid.y)*${W}u+col];}
            workgroupBarrier();for(var k=0u;k<16u;k++){v+=a[lid.y*16u+k]*b[k*16u+lid.x];}workgroupBarrier();
          }
          if(row<${count}u&&col<${W}u){let value=x[row*${W}u+col];previous[row*${W}u+col]=v*(1.0-value*value);}
        }`, [parameterBuffer, deltas[i], activations[i], deltas[i - 1]]);
      reverse.push({partial, reduce, delta});
    }
    const area = await compile('Faithful area transform', `
      @group(0) @binding(0) var<storage,read> x:array<f32>;
      @group(0) @binding(1) var<storage,read_write> area:array<f32>;
      @group(0) @binding(2) var<storage,read_write> logs:array<f32>;
      @group(0) @binding(3) var<storage,read_write> derivative:array<f32>;
      @compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id:vec3<u32>){
        let row=id.x;if(row>=${count}u){return;}
        var value=x[row]+1.0;if(row%32u==0u){value=1.0;}let z=value*0.35+(0.1-0.35);
        let tail=exp(-abs(z));var log1p=log(1.0+tail);
        // WGSL has no log1p. Preserve small softplus tails that log(1+x)
        // would round away; the cubic remainder is below float32 precision.
        if(tail<0.001){log1p=tail*(1.0-tail*0.5+tail*tail/3.0);}
        let raw=max(z,0.0)+log1p+0.0001;let a=min(16.0,raw);area[row]=a;logs[row]=log(a);
        let sigmoid=select(tail/(1.0+tail),1.0/(1.0+tail),z>=0.0);
        var clipDerivative=select(1.0,0.0,raw>16.0);if(raw==16.0){clipDerivative=0.5;}
        derivative[row]=select(0.35*sigmoid*clipDerivative/a,0.0,row%32u==0u);
      }`, [activations.at(-1), areaBuffer, logAreaBuffer, derivative]);
    const firstDelta = await compile('Faithful area VJP', `
      @group(0) @binding(0) var<storage,read> g:array<f32>;
      @group(0) @binding(1) var<storage,read> derivative:array<f32>;
      @group(0) @binding(2) var<storage,read_write> delta:array<f32>;
      @compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id:vec3<u32>){if(id.x<${count}u){delta[id.x]=g[id.x]*derivative[id.x];}}`, [cotangentBuffer, derivative, deltas.at(-1)]);
    const firstMoment = storage(parameterCount), secondMoment = storage(parameterCount), stepBuffer = storage(4);
    const rates = storage(iterations, Float32Array.from({length: iterations}, (_, n) => faithfulLearningRate(n, iterations)));
    const adam = await compile('Faithful Optax Adam', `
      @group(0) @binding(0) var<storage,read_write> p:array<f32>;
      @group(0) @binding(1) var<storage,read> g:array<f32>;
      @group(0) @binding(2) var<storage,read_write> m:array<f32>;
      @group(0) @binding(3) var<storage,read_write> v:array<f32>;
      @group(0) @binding(4) var<storage,read> state:array<u32>;
      @group(0) @binding(5) var<storage,read> lr:array<f32>;
      @compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id:vec3<u32>){
        let i=id.x;if(i>=${parameterCount}u){return;}let n=state[0];let t=f32(n+1u);
        m[i]=0.9*m[i]+0.1*g[i];v[i]=0.999*v[i]+0.001*g[i]*g[i];
        p[i]-=lr[n]*(m[i]/(1.0-pow(0.9,t)))/(sqrt(v[i]/(1.0-pow(0.999,t)))+1e-8);
      }`, [parameterBuffer, gradientBuffer, firstMoment, secondMoment, stepBuffer, rates]);
    return {parameterBuffer, areaBuffer, logAreaBuffer, cotangentBuffer, gradientBuffer, parameterCount, count,
      encodeForward(encoder) {
        for (const k of forward) run(encoder,k.kernel,k.x,k.y);run(encoder,area,Math.ceil(count/64));
      },
      encodeBackward(encoder) {
        run(encoder,firstDelta,Math.ceil(count/64));
        for(let i=reverse.length-1;i>=0;i--){const l=field.layout[i],k=reverse[i];
          run(encoder,k.partial,Math.ceil((l.inputSize+1)/16),Math.ceil(l.outputSize/16),groups);
          run(encoder,k.reduce,Math.ceil((l.weightCount+l.outputSize)/64));
          if(k.delta)run(encoder,k.delta,Math.ceil(l.inputSize/16),Math.ceil(count/16));
        }
      },
      encodeUpdate(encoder,step) {device.queue.writeBuffer(stepBuffer,0,Uint32Array.of(step,0,0,0));run(encoder,adam,Math.ceil(parameterCount/64));},
      dispose(){for(const b of owned)b.destroy();},
    };
  }catch(error){for(const b of owned)b.destroy();throw error;}
}
