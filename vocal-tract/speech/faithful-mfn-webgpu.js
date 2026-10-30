/** Original FourierNet/MFN in float32 WebGPU.
 * Learned sine filters gate three affine hidden layers. Parameters, all
 * parameter gradients and Adam state remain on the GPU between updates.
 * computePass allows composition into a caller's compute pass; optimizer:false
 * omits Adam buffers when that caller supplies its own optimizer. The stationary
 * vowel field uses an optional residual log-area transform; speech is unchanged.
 */
import {faithfulLearningRate} from './faithful-field.js?v=fixed-grid';

export async function createFaithfulGpuMfnField(device, field, {iterations = 500, computePass = false, optimizer = true} = {}) {
  const owned = [], width = field.config.width;
  const areaScale = field.config.areaScale ?? field.metadata?.areaTransform?.scale ?? .35;
  if (!Number.isFinite(areaScale) || areaScale <= 0 || !Number.isFinite(Math.fround(areaScale)) || Math.fround(areaScale) <= 0) throw new Error('MFN areaScale must be positive, finite, and representable in float32.');
  const shaderAreaScale = `${areaScale.toExponential()}f`;
  // Keep the original float32 evaluation order for the JAX-reference scale.
  // Other scales use the direct affine transform; pinning z separately avoids
  // cancellation affecting the fixed glottis as the output scale increases.
  const areaTransform = areaScale === .35
    ? 'var value=x[row]+1.0;if(row%32u==0u){value=1.0;}let z=value*0.35+(0.1-0.35);'
    : `let z=select(x[row]*${shaderAreaScale}+0.1,0.1,row%32u==0u);`;
  const count = field.config.frameTimes.length * 32, parameterCount = field.parameters.length;
  if (field.layout.length !== 4 || field.filterLayout.length !== 4 ||
      field.layout.slice(0, 3).some(layer => layer.inputSize !== width || layer.outputSize !== width) ||
      field.layout[3].inputSize !== width || field.layout[3].outputSize !== 1 ||
      field.filterLayout.some(layer => layer.inputSize !== 2 || layer.outputSize !== width)) {
    throw new Error('The MFN GPU field requires the original three-layer FourierNet layout.');
  }
  const storage = (length, data) => {
    const buffer = device.createBuffer({size: Math.max(4, length * 4), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST});
    owned.push(buffer);
    if (data) device.queue.writeBuffer(buffer, 0, data);
    return buffer;
  };
  const compile = async (label, code, buffers) => {
    const module = device.createShaderModule({label, code});
    const errors = (await module.getCompilationInfo()).messages.filter(message => message.type === 'error');
    if (errors.length) throw new Error(`${label}: ${errors.map(message => `${message.lineNum}: ${message.message}`).join('; ')}`);
    const pipeline = await device.createComputePipelineAsync({label, layout: 'auto', compute: {module, entryPoint: 'main'}});
    const bindGroup = device.createBindGroup({layout: pipeline.getBindGroupLayout(0), entries: buffers.map((buffer, binding) => ({binding, resource: {buffer}}))});
    return {pipeline, bindGroup};
  };
  const run = (encoder, kernel, x, y = 1, z = 1) => {
    const pass = computePass ? encoder : encoder.beginComputePass();
    pass.setPipeline(kernel.pipeline); pass.setBindGroup(0, kernel.bindGroup);
    pass.dispatchWorkgroups(x, y, z); if (!computePass) pass.end();
  };
  // The same phase expression is used for the primal and VJP. The learned
  // filters already contain the input scale: there is no RFF 2*pi factor.
  const phase = filter => `coords[row*2u]*p[${filter.weightOffset}u+col*2u]+coords[row*2u+1u]*p[${filter.weightOffset + 1}u+col*2u]+p[${filter.biasOffset}u+col]`;
  try {
    if (field.coordinates?.length !== count * 2) throw new Error('MFN coordinates do not match the field grid.');
    const coordinateBuffer = storage(count * 2, Float32Array.from(field.coordinates));
    const parameterBuffer = storage(parameterCount, Float32Array.from(field.parameters));
    const activations = Array.from({length: 4}, () => storage(count * width));
    const affineOutputs = Array.from({length: 3}, () => storage(count * width));
    const rawOutput = storage(count), outputDelta = storage(count);
    // Reuse these work buffers as reverse-mode traverses each layer. The
    // backward pass finishes before any parameter update can be encoded.
    const hiddenCotangent = storage(count * width), affineDelta = storage(count * width), filterDelta = storage(count * width);
    const logAreaBuffer = storage(count), areaBuffer = storage(count), derivative = storage(count);
    const cotangentBuffer = storage(count), gradientBuffer = storage(parameterCount);
    const groups = Math.min(32, Math.max(1, Math.ceil(count / 128))), rowsPerGroup = Math.ceil(count / groups);
    const partialBuffer = storage(groups * Math.max(...[...field.layout, ...field.filterLayout].map(layer => layer.weightCount + layer.outputSize)));

    const initial = await compile('Faithful MFN first sine filter', `
      @group(0) @binding(0) var<storage,read> p:array<f32>;
      @group(0) @binding(1) var<storage,read> coords:array<f32>;
      @group(0) @binding(2) var<storage,read_write> h:array<f32>;
      @compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id:vec3<u32>){
        let row=id.y;let col=id.x;if(row>=${count}u||col>=${width}u){return;}let index=row*${width}u+col;
        h[index]=sin(${phase(field.filterLayout[0])});
      }`, [parameterBuffer, coordinateBuffer, activations[0]]);
    const forward = [], gate = [], reverse = [], gradients = [];
    for (let i = 0; i < 3; i++) {
      const layer = field.layout[i], filter = field.filterLayout[i + 1];
      forward.push(await compile(`Faithful MFN affine and filter ${i + 1}`, `
        @group(0) @binding(0) var<storage,read> p:array<f32>;
        @group(0) @binding(1) var<storage,read> x:array<f32>;
        @group(0) @binding(2) var<storage,read> coords:array<f32>;
        @group(0) @binding(3) var<storage,read_write> z:array<f32>;
        @group(0) @binding(4) var<storage,read_write> h:array<f32>;
        var<workgroup> a:array<f32,256>;var<workgroup> b:array<f32,256>;
        @compute @workgroup_size(16,16) fn main(@builtin(workgroup_id) wg:vec3<u32>,@builtin(local_invocation_id) lid:vec3<u32>){
          let row=wg.y*16u+lid.y;let col=wg.x*16u+lid.x;let local=lid.y*16u+lid.x;var value=0.0;
          for(var tile=0u;tile<${width}u;tile+=16u){
            a[local]=0.0;b[local]=0.0;
            if(row<${count}u&&tile+lid.x<${width}u){a[local]=x[row*${width}u+tile+lid.x];}
            if(col<${width}u&&tile+lid.y<${width}u){b[local]=p[${layer.weightOffset}u+col*${width}u+tile+lid.y];}
            workgroupBarrier();for(var k=0u;k<16u;k++){value+=a[lid.y*16u+k]*b[k*16u+lid.x];}workgroupBarrier();
          }
          if(row<${count}u&&col<${width}u){
            value+=p[${layer.biasOffset}u+col];z[row*${width}u+col]=value;
            h[row*${width}u+col]=sin(${phase(filter)})*value;
          }
        }`, [parameterBuffer, activations[i], coordinateBuffer, affineOutputs[i], activations[i + 1]]));
      gate.push(await compile(`Faithful MFN product VJP ${i + 1}`, `
        @group(0) @binding(0) var<storage,read> p:array<f32>;
        @group(0) @binding(1) var<storage,read> coords:array<f32>;
        @group(0) @binding(2) var<storage,read> z:array<f32>;
        @group(0) @binding(3) var<storage,read> dh:array<f32>;
        @group(0) @binding(4) var<storage,read_write> dz:array<f32>;
        @group(0) @binding(5) var<storage,read_write> df:array<f32>;
        @compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id:vec3<u32>){
          let row=id.y;let col=id.x;if(row>=${count}u||col>=${width}u){return;}let index=row*${width}u+col;
          let angle=${phase(filter)};let cotangent=dh[index];
          dz[index]=cotangent*sin(angle);df[index]=cotangent*z[index]*cos(angle);
        }`, [parameterBuffer, coordinateBuffer, affineOutputs[i], hiddenCotangent, affineDelta, filterDelta]));
      reverse.push(await compile(`Faithful MFN reverse affine ${i}`, `
        @group(0) @binding(0) var<storage,read> p:array<f32>;
        @group(0) @binding(1) var<storage,read> d:array<f32>;
        @group(0) @binding(2) var<storage,read_write> previous:array<f32>;
        var<workgroup> a:array<f32,256>;var<workgroup> b:array<f32,256>;
        @compute @workgroup_size(16,16) fn main(@builtin(workgroup_id) wg:vec3<u32>,@builtin(local_invocation_id) lid:vec3<u32>){
          let row=wg.y*16u+lid.y;let col=wg.x*16u+lid.x;let local=lid.y*16u+lid.x;var value=0.0;
          for(var tile=0u;tile<${width}u;tile+=16u){
            a[local]=0.0;b[local]=0.0;
            if(row<${count}u&&tile+lid.x<${width}u){a[local]=d[row*${width}u+tile+lid.x];}
            if(col<${width}u&&tile+lid.y<${width}u){b[local]=p[${layer.weightOffset}u+(tile+lid.y)*${width}u+col];}
            workgroupBarrier();for(var k=0u;k<16u;k++){value+=a[lid.y*16u+k]*b[k*16u+lid.x];}workgroupBarrier();
          }
          if(row<${count}u&&col<${width}u){previous[row*${width}u+col]=value;}
        }`, [parameterBuffer, affineDelta, hiddenCotangent]));
    }
    const outputLayer = field.layout[3];
    const output = await compile('Faithful MFN linear output', `
      @group(0) @binding(0) var<storage,read> p:array<f32>;
      @group(0) @binding(1) var<storage,read> x:array<f32>;
      @group(0) @binding(2) var<storage,read_write> y:array<f32>;
      @compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id:vec3<u32>){
        let row=id.x;if(row>=${count}u){return;}var value=p[${outputLayer.biasOffset}u];
        for(var k=0u;k<${width}u;k++){value+=x[row*${width}u+k]*p[${outputLayer.weightOffset}u+k];}y[row]=value;
      }`, [parameterBuffer, activations[3], rawOutput]);
    const outputReverse = await compile('Faithful MFN output VJP', `
      @group(0) @binding(0) var<storage,read> p:array<f32>;
      @group(0) @binding(1) var<storage,read> d:array<f32>;
      @group(0) @binding(2) var<storage,read_write> previous:array<f32>;
      @compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id:vec3<u32>){
        let row=id.y;let col=id.x;if(row>=${count}u||col>=${width}u){return;}let index=row*${width}u+col;
        previous[index]=d[row]*p[${outputLayer.weightOffset}u+col];
      }`, [parameterBuffer, outputDelta, hiddenCotangent]);
    const initialGate = await compile('Faithful MFN first sine VJP', `
      @group(0) @binding(0) var<storage,read> p:array<f32>;
      @group(0) @binding(1) var<storage,read> coords:array<f32>;
      @group(0) @binding(2) var<storage,read> dh:array<f32>;
      @group(0) @binding(3) var<storage,read_write> df:array<f32>;
      @compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id:vec3<u32>){
        let row=id.y;let col=id.x;if(row>=${count}u||col>=${width}u){return;}let index=row*${width}u+col;
        df[index]=dh[index]*cos(${phase(field.filterLayout[0])});
      }`, [parameterBuffer, coordinateBuffer, hiddenCotangent, filterDelta]);
    // Matrix parameter VJPs: dW = delta^T * input; db = sum(delta).
    for (let i = 0; i < field.layout.length; i++) {
      const layer = field.layout[i], W = layer.inputSize, O = layer.outputSize, size = layer.weightCount + O;
      const partial = await compile(`Faithful MFN affine parameter gradient ${i}`, `
        @group(0) @binding(0) var<storage,read> x:array<f32>;
        @group(0) @binding(1) var<storage,read> d:array<f32>;
        @group(0) @binding(2) var<storage,read_write> partial:array<f32>;
        var<workgroup> a:array<f32,256>;var<workgroup> b:array<f32,256>;
        @compute @workgroup_size(16,16) fn main(@builtin(workgroup_id) wg:vec3<u32>,@builtin(local_invocation_id) lid:vec3<u32>){
          let out=wg.y*16u+lid.y;let col=wg.x*16u+lid.x;let local=lid.y*16u+lid.x;var value=0.0;
          let start=wg.z*${rowsPerGroup}u;let end=min(start+${rowsPerGroup}u,${count}u);
          for(var tile=start;tile<end;tile+=16u){
            a[local]=0.0;b[local]=0.0;
            if(out<${O}u&&tile+lid.x<end){a[local]=d[(tile+lid.x)*${O}u+out];}
            if(col<=${W}u&&tile+lid.y<end){b[local]=1.0;if(col<${W}u){b[local]=x[(tile+lid.y)*${W}u+col];}}
            workgroupBarrier();for(var k=0u;k<16u;k++){value+=a[lid.y*16u+k]*b[k*16u+lid.x];}workgroupBarrier();
          }
          if(out<${O}u&&col<=${W}u){
            let index=select(out*${W}u+col,${layer.weightCount}u+out,col==${W}u);partial[wg.z*${size}u+index]=value;
          }
        }`, [activations[i], i === 3 ? outputDelta : affineDelta, partialBuffer]);
      const reduce = await compile(`Faithful MFN affine gradient reduction ${i}`, `
        @group(0) @binding(0) var<storage,read> partial:array<f32>;
        @group(0) @binding(1) var<storage,read_write> gradient:array<f32>;
        @compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id:vec3<u32>){
          let index=id.x;if(index>=${size}u){return;}var value=0.0;
          for(var block=0u;block<${groups}u;block++){value+=partial[block*${size}u+index];}gradient[${layer.weightOffset}u+index]=value;
        }`, [partialBuffer, gradientBuffer]);
      gradients.push({partial, reduce, W, O, size});
    }
    // The two-coordinate filters use a direct reduction rather than a padded
    // matrix multiplication. The same partial workspace is reused sequentially.
    const filterPartial = await compile('Faithful MFN filter parameter gradient', `
      @group(0) @binding(0) var<storage,read> coords:array<f32>;
      @group(0) @binding(1) var<storage,read> d:array<f32>;
      @group(0) @binding(2) var<storage,read_write> partial:array<f32>;
      @compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id:vec3<u32>){
        let out=id.x;if(out>=${width}u){return;}let block=id.y;
        let start=block*${rowsPerGroup}u;let end=min(start+${rowsPerGroup}u,${count}u);var dt=0.0;var dx=0.0;var db=0.0;
        for(var row=start;row<end;row++){let value=d[row*${width}u+out];dt+=value*coords[row*2u];dx+=value*coords[row*2u+1u];db+=value;}
        partial[block*${width * 3}u+out*2u]=dt;partial[block*${width * 3}u+out*2u+1u]=dx;partial[block*${width * 3}u+${width * 2}u+out]=db;
      }`, [coordinateBuffer, filterDelta, partialBuffer]);
    const filterReduce = [];
    for (let i = 0; i < 4; i++) {
      const filter = field.filterLayout[i];
      filterReduce.push(await compile(`Faithful MFN filter gradient reduction ${i}`, `
        @group(0) @binding(0) var<storage,read> partial:array<f32>;
        @group(0) @binding(1) var<storage,read_write> gradient:array<f32>;
        @compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id:vec3<u32>){
          let index=id.x;if(index>=${width * 3}u){return;}var value=0.0;
          for(var block=0u;block<${groups}u;block++){value+=partial[block*${width * 3}u+index];}gradient[${filter.weightOffset}u+index]=value;
        }`, [partialBuffer, gradientBuffer]));
    }
    const residual = field.metadata.areaTransform?.type === 'residual-log-area';
    if (residual && (field.baseLogAreas?.length !== count || !field.baseLogAreas.every(Number.isFinite))) throw new Error('The vowel MFN requires one finite starting log area per section.');
    const baseline = residual ? storage(count, Float32Array.from(field.baseLogAreas)) : null;
    const area = await compile('Faithful MFN area transform', residual ? `
      @group(0) @binding(0) var<storage,read> x:array<f32>;
      @group(0) @binding(1) var<storage,read_write> area:array<f32>;
      @group(0) @binding(2) var<storage,read_write> logs:array<f32>;
      @group(0) @binding(3) var<storage,read_write> derivative:array<f32>;
      @group(0) @binding(4) var<storage,read> baseline:array<f32>;
      @compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id:vec3<u32>){
        let row=id.x;if(row>=${count}u){return;}
        let value=baseline[row]+${shaderAreaScale}*x[row];logs[row]=clamp(value,${Math.log(.12)},${Math.log(16)});
        area[row]=exp(logs[row]);derivative[row]=select(0.0,${shaderAreaScale},value>${Math.log(.12)}&&value<${Math.log(16)});
      }` : `
      @group(0) @binding(0) var<storage,read> x:array<f32>;
      @group(0) @binding(1) var<storage,read_write> area:array<f32>;
      @group(0) @binding(2) var<storage,read_write> logs:array<f32>;
      @group(0) @binding(3) var<storage,read_write> derivative:array<f32>;
      @compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id:vec3<u32>){
        let row=id.x;if(row>=${count}u){return;}${areaTransform}
        let tail=exp(-abs(z));var log1p=log(1.0+tail);
        if(tail<0.001){log1p=tail*(1.0-tail*0.5+tail*tail/3.0);}
        let raw=max(z,0.0)+log1p+0.0001;let a=min(16.0,raw);area[row]=a;logs[row]=log(a);
        let sigmoid=select(tail/(1.0+tail),1.0/(1.0+tail),z>=0.0);
        var clipDerivative=select(1.0,0.0,raw>16.0);if(raw==16.0){clipDerivative=0.5;}
        derivative[row]=select(${shaderAreaScale}*sigmoid*clipDerivative/a,0.0,row%32u==0u);
      }`, [rawOutput, areaBuffer, logAreaBuffer, derivative, ...(residual ? [baseline] : [])]);
    const firstDelta = await compile('Faithful MFN area VJP', `
      @group(0) @binding(0) var<storage,read> g:array<f32>;
      @group(0) @binding(1) var<storage,read> derivative:array<f32>;
      @group(0) @binding(2) var<storage,read_write> delta:array<f32>;
      @compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id:vec3<u32>){if(id.x<${count}u){delta[id.x]=g[id.x]*derivative[id.x];}}
      `, [cotangentBuffer, derivative, outputDelta]);
    let adam, stepBuffer;
    if (optimizer) {
    const firstMoment = storage(parameterCount), secondMoment = storage(parameterCount); stepBuffer = storage(4);
    const rates = storage(iterations, Float32Array.from({length: iterations}, (_, n) => faithfulLearningRate(n, iterations, 0.005)));
    adam = await compile('Faithful MFN Optax Adam', `
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
    }
    const affineGradient = (encoder, index) => {
      const kernel = gradients[index];
      run(encoder, kernel.partial, Math.ceil((kernel.W + 1) / 16), Math.ceil(kernel.O / 16), groups);
      run(encoder, kernel.reduce, Math.ceil(kernel.size / 64));
    };
    const filterGradient = (encoder, index) => {
      run(encoder, filterPartial, Math.ceil(width / 64), groups);
      run(encoder, filterReduce[index], Math.ceil(width * 3 / 64));
    };
    return {parameterBuffer, areaBuffer, logAreaBuffer, cotangentBuffer, gradientBuffer, parameterCount, count,
      encodeForward(encoder) {
        run(encoder, initial, Math.ceil(width / 64), count);
        for (const kernel of forward) run(encoder, kernel, Math.ceil(width / 16), Math.ceil(count / 16));
        run(encoder, output, Math.ceil(count / 64));run(encoder, area, Math.ceil(count / 64));
      },
      encodeBackward(encoder) {
        run(encoder, firstDelta, Math.ceil(count / 64));affineGradient(encoder, 3);
        run(encoder, outputReverse, Math.ceil(width / 64), count);
        for (let i = 2; i >= 0; i--) {
          run(encoder, gate[i], Math.ceil(width / 64), count);
          affineGradient(encoder, i);filterGradient(encoder, i + 1);
          run(encoder, reverse[i], Math.ceil(width / 16), Math.ceil(count / 16));
        }
        run(encoder, initialGate, Math.ceil(width / 64), count);filterGradient(encoder, 0);
      },
      encodeUpdate(encoder, step) {
        if (!adam) throw new Error('This MFN uses the caller-owned optimizer.');
        device.queue.writeBuffer(stepBuffer, 0, Uint32Array.of(step, 0, 0, 0));
        run(encoder, adam, Math.ceil(parameterCount / 64));
      },
      dispose() { for (const buffer of owned) buffer.destroy(); },
    };
  } catch (error) { for (const buffer of owned) buffer.destroy(); throw error; }
}
