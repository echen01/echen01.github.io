import {rolloutGeometry} from './faithful-rollout.js';
import {createFaithfulGpuAcoustic} from './faithful-acoustic-webgpu.js';
import {getFaithfulRolloutWindows} from './faithful-window-constants.js';
import {acousticFftSize} from './faithful-acoustic.js';

const THREEFRY = `
fn threefry(key:vec2u, counter:vec2u)->vec2u {
  let keys=array<u32,3>(key.x,key.y,key.x^key.y^0x1bd11bdau);
  let rotations=array<u32,8>(13u,15u,26u,6u,17u,29u,16u,24u);
  var x=counter.x+keys[0];var y=counter.y+keys[1];
  for(var round=0u;round<20u;round++){
    let r=rotations[round%8u];x+=y;y=((y<<r)|(y>>(32u-r)))^x;
    if(round%4u==3u){let inject=(round+1u)/4u;x+=keys[inject%3u];y+=keys[(inject+1u)%3u]+inject;}
  }
  return vec2u(x,y);
}
fn randomUniform(key:vec2u,index:u32)->f32{let bits=threefry(key,vec2u(0u,index));return f32((bits.x^bits.y)>>9u)/8388608.0;}
`;

/**
 * Original Perlin draw per iteration, complete-frame synthesis and OLA VJP.
 * compensatedWindowProducts retains both f32 parts of source/window and
 * cotangent/window products until the double-single FFT, where cancellation
 * can expose their low bits. The mathematical equations are unchanged.
 */
export async function createFaithfulGpuRollout(device, uin, {
  sampleRate = 32000, fftSize = acousticFftSize(sampleRate), length = 17.5, batchSize = 8, fftOptions,
  analysisWindow: suppliedAnalysisWindow, synthesisWindow: suppliedSynthesisWindow,
  fixedNoise, diagnostics = false, acousticOptions = {}, compensatedWindowProducts = false,
} = {}) {
  if (compensatedWindowProducts && fftOptions?.arithmetic !== 'double-single') throw new Error('Compensated rollout window products require double-single FFT arithmetic.');
  const geometry = rolloutGeometry(uin.length, sampleRate, fftSize);
  const pinnedWindows = [16000, 32000].includes(sampleRate) ? getFaithfulRolloutWindows({sampleRate}) : geometry;
  for (const [key, supplied] of [['analysisWindow', suppliedAnalysisWindow], ['synthesisWindow', suppliedSynthesisWindow]]) {
    const selected = supplied ?? pinnedWindows[key];
    if (selected.length !== geometry.frameLength) throw new Error(`${key} must contain frameLength entries.`);
    geometry[key] = Float32Array.from(selected);
  }
  if (fixedNoise !== undefined && fixedNoise.length !== uin.length * 31) throw new Error('Fixed noise must be channel-major [31,sampleCount].');
  const {sampleCount, frames, frameLength, hopLength, analysisWindow, synthesisWindow} = geometry;
  const owned = [], acoustics = [];
  const make = (count, data, uniform = false) => {
    const buffer = device.createBuffer({size: Math.max(16, count * 4), usage: (uniform ? GPUBufferUsage.UNIFORM : GPUBufferUsage.STORAGE) | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST});
    owned.push(buffer); if(data)device.queue.writeBuffer(buffer,0,data); return buffer;
  };
  const compile = async (label, code, buffers) => {
    const module = device.createShaderModule({label,code});
    const errors=(await module.getCompilationInfo()).messages.filter(m=>m.type==='error');
    if(errors.length)throw new Error(`${label}: ${errors.map(m=>m.message).join('; ')}`);
    const pipeline=await device.createComputePipelineAsync({label,layout:'auto',compute:{module,entryPoint:'main'}});
    const bindGroup=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:buffers.map((buffer,binding)=>({binding,resource:{buffer}}))});
    return {pipeline,bindGroup};
  };
  const dispatch = (encoder,kernel,count) => {
    const pass=encoder.beginComputePass();pass.setPipeline(kernel.pipeline);pass.setBindGroup(0,kernel.bindGroup);
    const groups=Math.ceil(count/128);pass.dispatchWorkgroups(Math.min(65535,groups),Math.ceil(groups/65535));pass.end();
  };
  const index='let i=id.x+id.y*8388480u;';
  try {
    const analysis32=Float32Array.from(analysisWindow),inputFrames=new Float32Array(frames*fftSize);
    const inputFramesLow=compensatedWindowProducts?new Float32Array(frames*fftSize):undefined;
    for(let frame=0;frame<frames;frame++)for(let i=0;i<frameLength;i++){
      const index=frame*fftSize+i,product=Math.fround(uin[frame*hopLength+i])*analysis32[i];
      inputFrames[index]=product;
      if(compensatedWindowProducts)inputFramesLow[index]=product-inputFrames[index];
    }
    const inputBuffer=make(inputFrames.length,inputFrames), areaBuffer=make(frames*32), gradientBuffer=make(frames*32);
    const inputLowBuffer=compensatedWindowProducts?make(inputFramesLow.length,inputFramesLow):undefined;
    const noiseBuffer=make(sampleCount*31,fixedNoise === undefined ? undefined : Float32Array.from(fixedNoise));
    const frameOutput=make(frames*fftSize), frameCotangent=make(frames*fftSize), waveformBuffer=make(sampleCount);
    const frameCotangentLow=compensatedWindowProducts?make(frames*fftSize):undefined;
    const roundingMask=compensatedWindowProducts?make(4,new Uint32Array(4),true):undefined;
    const cotangentBuffer=make(sampleCount), seedBuffer=make(4,undefined,true);
    const analysis=make(frameLength,analysis32),synthesis=make(frameLength,Float32Array.from(synthesisWindow));
    const resolutions=Array.from({length:5},(_,i)=>Math.floor(sampleCount/64)*2**i), counts=resolutions.map(r=>r+2);
    const gradientCount=counts.reduce((a,b)=>a+b,0), randomCount=gradientCount+5;
    const randomBuffer=make(randomCount*31);
    const random=await compile('Faithful Threefry uniforms',`${THREEFRY}
      @group(0) @binding(0) var<uniform> seed:vec4u;
      @group(0) @binding(1) var<storage,read_write> random:array<f32>;
      @compute @workgroup_size(128) fn main(@builtin(global_invocation_id) id:vec3u){${index}if(i>=${randomCount*31}u){return;}
        let section=i/${randomCount}u;let j=i%${randomCount}u;let key=threefry(vec2u(0u,seed.x),vec2u(0u,section));
        let child=threefry(key,vec2u(0u,select(0u,1u,j>=${gradientCount}u)));
        var value=0.0;if(j<${gradientCount}u){value=randomUniform(child,j)*2.0-1.0;}else{value=randomUniform(child,j-${gradientCount}u);}
        random[i]=value;
      }`,[seedBuffer,randomBuffer]);
    const perlin=await compile('Faithful five octave Perlin',`
      @group(0) @binding(0) var<storage,read> random:array<f32>;
      @group(0) @binding(1) var<storage,read_write> noise:array<f32>;
      @compute @workgroup_size(128) fn main(@builtin(global_invocation_id) id:vec3u){${index}if(i>=${sampleCount*31}u){return;}
        let section=i/${sampleCount}u;let sample=i%${sampleCount}u;let base=section*${randomCount}u;
        let resolutions=array<u32,5>(${resolutions.map(r=>`${r}u`).join(',')});var offset=0u;var value=0.0;var amplitude=1.0;
        for(var octave=0u;octave<5u;octave++){
          let position=f32(sample)*(f32(resolutions[octave])/${sampleCount}.0)+random[base+${gradientCount}u+octave];
          let cell=u32(floor(position));let t=position-f32(cell);
          let d0=random[base+offset+cell]*t;let d1=random[base+offset+cell+1u]*(t-1.0);
          let interpolant=t*t*t*(t*(t*6.0-15.0)+10.0);value+=amplitude*(d0+interpolant*(d1-d0));
          offset+=resolutions[octave]+2u;amplitude*=0.5;
        }noise[i]=value;
      }`,[randomBuffer,noiseBuffer]);
    const ola=await compile('Faithful synthesis overlap add',`
      @group(0) @binding(0) var<storage,read> frames:array<f32>;
      @group(0) @binding(1) var<storage,read> window:array<f32>;
      @group(0) @binding(2) var<storage,read_write> wave:array<f32>;
      @compute @workgroup_size(128) fn main(@builtin(global_invocation_id) id:vec3u){${index}if(i>=${sampleCount}u){return;}var value=0.0;
        let last=min(i/${hopLength}u,${frames-1}u);let first=select(0u,(i-${frameLength}u)/${hopLength}u+1u,i>=${frameLength}u);
        for(var frame=first;frame<=last;frame++){let n=i-frame*${hopLength}u;if(n<${frameLength}u){value+=frames[frame*${fftSize}u+n]*window[n];}}
        wave[i]=value;
      }`,[frameOutput,synthesis,waveformBuffer]);
    const olaVjp=await compile('Faithful overlap add VJP',`
      @group(0) @binding(0) var<storage,read> gradient:array<f32>;
      @group(0) @binding(1) var<storage,read> window:array<f32>;
      @group(0) @binding(2) var<storage,read_write> local:array<f32>;
      ${compensatedWindowProducts ? `@group(0) @binding(3) var<storage,read_write> localLow:array<f32>;
      @group(0) @binding(4) var<uniform> roundingMask:vec4u;
      fn round32(value:f32)->f32{return bitcast<f32>(bitcast<u32>(value)^roundingMask.x);}` : ''}
      @compute @workgroup_size(128) fn main(@builtin(global_invocation_id) id:vec3u){${index}if(i>=${frames*fftSize}u){return;}
        let frame=i/${fftSize}u;let n=i%${fftSize}u;var value=0.0;
        ${compensatedWindowProducts ? `var residual=0.0;
        if(n<${frameLength}u){let g=gradient[frame*${hopLength}u+n];let w=window[n];value=round32(g*w);residual=fma(g,w,-value);}localLow[i]=residual;` : `if(n<${frameLength}u){value=gradient[frame*${hopLength}u+n]*window[n];}`}
        local[i]=value;
      }`,[cotangentBuffer,synthesis,frameCotangent,...(compensatedWindowProducts?[frameCotangentLow,roundingMask]:[])]);
    const full=await createFaithfulGpuAcoustic(device,{...acousticOptions,diagnostics:diagnostics||acousticOptions.diagnostics,compensatedCotangent:compensatedWindowProducts,compensatedInput:compensatedWindowProducts,frames:Math.min(batchSize,frames),fftSize,sampleRate,length,fftOptions});acoustics.push(full);
    const remainder=frames%batchSize;
    let tail=full;
    if(frames>batchSize&&remainder){tail=await createFaithfulGpuAcoustic(device,{...acousticOptions,diagnostics:diagnostics||acousticOptions.diagnostics,compensatedCotangent:compensatedWindowProducts,compensatedInput:compensatedWindowProducts,frames:remainder,fftSize,sampleRate,length,fftOptions});acoustics.push(tail);}
    // Materializing every overlapping noise window would require > 200 MB for
    // a 10-second clip, exceeding the baseline 128 MiB storage binding limit.
    // The underlying Perlin noise is shared; gather only the current acoustic
    // batch directly into its noise input, on both forward and recomputation.
    const batches=[];let windowNoisePipeline;
    const windowNoiseCode=`
      @group(0) @binding(0) var<storage,read> noise:array<f32>;
      @group(0) @binding(1) var<storage,read> window:array<f32>;
      @group(0) @binding(2) var<storage,read_write> frames:array<f32>;
      @group(0) @binding(3) var<uniform> batch:vec4u;
      @compute @workgroup_size(128) fn main(@builtin(global_invocation_id) id:vec3u){${index}if(i>=batch.y*${31*fftSize}u){return;}
        let sample=i%${fftSize}u;let section=(i/${fftSize}u)%31u;let frame=batch.x+i/${31*fftSize}u;var value=0.0;
        if(sample<${frameLength}u){value=noise[section*${sampleCount}u+frame*${hopLength}u+sample]*window[sample];}frames[i]=value;
      }`;
    for(let first=0;first<frames;first+=batchSize){
      const size=Math.min(batchSize,frames-first),kernel=size===batchSize||frames<=batchSize?full:tail;
      const metadata=make(4,Uint32Array.of(first,size,0,0),true);
      const buffers=[noiseBuffer,analysis,kernel.noiseBuffer,metadata];
      let windowKernel;
      if(!windowNoisePipeline){
        windowKernel=await compile('Faithful batched noise Tukey windows',windowNoiseCode,buffers);
        windowNoisePipeline=windowKernel.pipeline;
      }else{
        const bindGroup=device.createBindGroup({layout:windowNoisePipeline.getBindGroupLayout(0),
          entries:buffers.map((buffer,binding)=>({binding,resource:{buffer}}))});
        windowKernel={pipeline:windowNoisePipeline,bindGroup};
      }
      batches.push({first,size,kernel,windowKernel});
    }
    function loadBatch(encoder,{first,size,kernel,windowKernel}){
      encoder.copyBufferToBuffer(areaBuffer,first*32*4,kernel.areaBuffer,0,size*32*4);
      encoder.copyBufferToBuffer(inputBuffer,first*fftSize*4,kernel.uinBuffer,0,size*fftSize*4);
      if(compensatedWindowProducts)encoder.copyBufferToBuffer(inputLowBuffer,first*fftSize*4,kernel.uinLowBuffer,0,size*fftSize*4);
      dispatch(encoder,windowKernel,size*31*fftSize);
    }
    return {...geometry,areaBuffer,gradientBuffer,waveformBuffer,cotangentBuffer,noiseBuffer,
      diagnosticBuffers:diagnostics?{inputFramesBuffer:inputBuffer,inputFramesLowBuffer:inputLowBuffer,frameOutputBuffer:frameOutput,frameCotangentBuffer:frameCotangent,frameCotangentLowBuffer:frameCotangentLow,analysisWindowBuffer:analysis,synthesisWindowBuffer:synthesis,lastAcoustic:batches.at(-1).kernel.diagnosticBuffers,lastBatchFirstFrame:batches.at(-1).first,lastBatchFrames:batches.at(-1).size}:undefined,
      encodeForward(encoder,iteration=0){
        if(fixedNoise===undefined){
          device.queue.writeBuffer(seedBuffer,0,Uint32Array.of(iteration,0,0,0));
          dispatch(encoder,random,randomCount*31);dispatch(encoder,perlin,sampleCount*31);
        }
        for(const batch of batches){const {first,size,kernel}=batch;
          loadBatch(encoder,batch);kernel.encodeForward(encoder);encoder.copyBufferToBuffer(kernel.outputBuffer,0,frameOutput,first*fftSize*4,size*fftSize*4);
        }dispatch(encoder,ola,sampleCount);
      },
      encodeBackward(encoder){
        dispatch(encoder,olaVjp,frames*fftSize);
        for(const batch of batches){const {first,size,kernel}=batch;
          loadBatch(encoder,batch);kernel.encodeForward(encoder);
          encoder.copyBufferToBuffer(frameCotangent,first*fftSize*4,kernel.cotangentBuffer,0,size*fftSize*4);
          if(compensatedWindowProducts)encoder.copyBufferToBuffer(frameCotangentLow,first*fftSize*4,kernel.cotangentLowBuffer,0,size*fftSize*4);
          kernel.encodeBackward(encoder);
          encoder.copyBufferToBuffer(kernel.dlogAreaBuffer,0,gradientBuffer,first*32*4,size*32*4);
        }
      },
      dispose(){for(const a of acoustics)a.dispose();for(const b of owned)b.destroy();},
    };
  }catch(error){for(const a of acoustics)a.dispose();for(const b of owned)b.destroy();throw error;}
}
