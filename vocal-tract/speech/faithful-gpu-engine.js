import {faithfulNetwork} from './faithful-networks.js?v=mfn-area-scale-1';
import {createFaithfulGpuField} from './faithful-field-webgpu.js?v=fixed-grid';
import {createFaithfulGpuMfnField} from './faithful-mfn-webgpu.js?v=mfn-area-scale-1';
import {createFaithfulGpuRollout} from './faithful-rollout-webgpu.js';
import {createFaithfulGpuLoss} from './faithful-loss-webgpu.js';
import {rolloutGeometry} from './faithful-rollout.js';
import {getFaithfulStftConstants} from './faithful-window-constants.js';

/** Entire fit_sample neural-field training computation on GPU, with scalar progress
 * readback. Source/target preparation and static initialization happen once.
 */
export async function createFaithfulGpuFitter({target, uin, sampleRate = 32000, metadata = {}}, {
  iterations = 500, fieldType = 'rff', hiddenDim = fieldType === 'mfn' ? 256 : 512, mfnInputScale = 256, mfnAreaScale = .35, initialization, device: suppliedDevice, signal, onStage,
  fftOptions = {arithmetic:'double-single'}, lossOptions = {}, rolloutOptions = {},
  submissionMode = 'staged',
} = {}) {
  const network = faithfulNetwork(fieldType, hiddenDim);
  if (target.length !== uin.length) throw new Error('Target and glottal source lengths differ.');
  if (!Number.isInteger(iterations) || iterations < 1 || iterations > 500) throw new Error('Use 1–500 optimizer iterations.');
  if (!['staged','combined'].includes(submissionMode)) throw new Error('Unknown GPU submission mode.');
  if (signal?.aborted) throw new DOMException('Fitting canceled.', 'AbortError');
  const geometry=rolloutGeometry(target.length,sampleRate);
  const adapter = suppliedDevice ? null : await globalThis.navigator?.gpu?.requestAdapter({powerPreference:'high-performance'});
  if (!suppliedDevice && !adapter) throw new Error('WebGPU is unavailable. Open this page in a browser with WebGPU support.');
  const device = suppliedDevice ?? await adapter.requestDevice();
  const owned = new Set(); let gpuField, rollout, loss, destroyed = false, failure = null, lostInfo = null;
  const deviceLoss=device.lost.then(info => {if(!destroyed)lostInfo=info;});
  const gpuError = event => {failure=event.error;};device.addEventListener('uncapturederror',gpuError);
  function normalizeError(error) {
    if(signal?.aborted)return new DOMException('Fitting canceled.','AbortError');
    if(error?.code==='WEBGPU_DEVICE_LOST')return error;
    // Some browsers reject mapAsync before delivering device.lost. Recognize
    // that specific failure too, without retrying ordinary validation errors.
    if(lostInfo || /(?:context|device)\s+(?:was\s+|is\s+)?lost/i.test(error?.message ?? '')) {
      const detail=lostInfo?.message || error?.message || 'Device lost';
      return Object.assign(new Error(`The browser reset the GPU: ${detail}`,{cause:error}),
        {code:'WEBGPU_DEVICE_LOST',reason:lostInfo?.reason ?? 'unknown'});
    }
    return error;
  }
  function check() {
    if(signal?.aborted)throw new DOMException('Fitting canceled.','AbortError');
    if(lostInfo || failure)throw normalizeError(failure ?? new Error('WebGPU device lost.'));
    if(destroyed)throw new Error('The GPU fitter has been disposed.');
  }
  async function reportedFailure(error) {
    // A reset can reject a pending operation before its separate loss event.
    let timeout;
    try{await Promise.race([deviceLoss,new Promise(resolve=>{timeout=setTimeout(resolve,50);})]);}
    finally{clearTimeout(timeout);}
    const reported=normalizeError(error);
    // An aborted GPU operation alone is not a user cancellation.
    if(reported===error && error.name==='AbortError')return new Error(`GPU operation was interrupted: ${error.message}`,{cause:error});
    return reported;
  }
  function dispose() {
    if(destroyed)return;
    destroyed=true;
    gpuField?.dispose();rollout?.dispose();loss?.destroy();
    for(const b of owned)b.destroy();owned.clear();
    device.removeEventListener('uncapturederror',gpuError);
    if(!suppliedDevice)device.destroy();
  }
  const staging=count=>{check();const b=device.createBuffer({label:'Fitting readback',size:Math.max(4,count*4),usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});owned.add(b);return b;};
  async function mappedValues(stage,count) {
    let mapped=false;
    try {
      await stage.mapAsync(GPUMapMode.READ);mapped=true;check();
      return new Float32Array(stage.getMappedRange().slice(0,count*4));
    } catch(error) {
      throw await reportedFailure(error);
    } finally {if(mapped)stage.unmap();}
  }
  async function read(source,count,stage){check();const encoder=device.createCommandEncoder();encoder.copyBufferToBuffer(source,0,stage,0,count*4);device.queue.submit([encoder.finish()]);return mappedValues(stage,count);}
  async function readTemporary(source,count) {
    const stage=staging(count);
    try{return await read(source,count,stage);}
    finally{stage.destroy();owned.delete(stage);}
  }
  try {
    check();
    onStage?.(`Loading the original ${network.label} initialization…`);
    const initial=initialization??await network.load(undefined,{hiddenDim});
    if(initial.manifest.hiddenDim!==hiddenDim)throw new Error('Requested hidden width does not match the supplied initialization.');
    check();
    const field=network.create(geometry.frames,target.length/sampleRate,initial,{lazyFeatures:true,inputScale:mfnInputScale,areaScale:mfnAreaScale});
    onStage?.(`Compiling the waveform simulator, loss, and ${network.label} gradient kernels…`);
    gpuField=await (fieldType==='mfn' ? createFaithfulGpuMfnField : createFaithfulGpuField)(device,field,{iterations});check();
    const compensated=fftOptions.arithmetic==='double-single';
    rollout=await createFaithfulGpuRollout(device,uin,{sampleRate,fftOptions,compensatedWindowProducts:compensated,...rolloutOptions});check();
    loss=await createFaithfulGpuLoss(device,target,{sampleRate,fftOptions,...getFaithfulStftConstants({sampleRate}),
      scalingMode:'after-fft',targetMode:'gpu',compensatedWindow:compensated,...lossOptions});check();
    const areaStage=staging(geometry.frames*32),lossStage=staging(1),waveStage=staging(target.length);
    let iteration=0,lastLoss=NaN;const history=[];
    const frameTimes=Float64Array.from({length:geometry.frames},(_,i)=>(i*geometry.hopLength+geometry.frameLength/2)/sampleRate);
    const framePitchHz=Float32Array.from(frameTimes,t=>metadata.pitchHz?.[Math.min(metadata.pitchHz.length-1,Math.floor(t/.016))]??0);
    const frameVoiced=Uint8Array.from(frameTimes,t=>Number((metadata.confidence?.[Math.min(metadata.confidence.length-1,Math.floor(t/.016))]??0)>=.5));
    function objectiveStages(n,update){
      return [
        ['Field forward',encoder=>{
          gpuField.encodeForward(encoder);
          encoder.copyBufferToBuffer(gpuField.areaBuffer,0,rollout.areaBuffer,0,geometry.frames*32*4);
        }],
        ['Acoustic forward',encoder=>rollout.encodeForward(encoder,n)],
        ['Waveform loss',encoder=>{
          loss.encodeForward(encoder,rollout.waveformBuffer);
          encoder.copyBufferToBuffer(loss.gradientBuffer,0,rollout.cotangentBuffer,0,target.length*4);
        }],
        ['Acoustic backward',encoder=>{
          rollout.encodeBackward(encoder);
          encoder.copyBufferToBuffer(rollout.gradientBuffer,0,gpuField.cotangentBuffer,0,geometry.frames*32*4);
        }],
        ['Field backward and update',encoder=>{
          gpuField.encodeBackward(encoder);
          if(update)gpuField.encodeUpdate(encoder,n);
          encoder.copyBufferToBuffer(loss.lossBuffer,0,lossStage,0,4);
        }],
      ];
    }
    async function submitObjective(n,update=false){
      check();const stages=objectiveStages(n,update);
      try {
        if(submissionMode==='combined'){
          const encoder=device.createCommandEncoder();for(const [,encode] of stages)encode(encoder);
          device.queue.submit([encoder.finish()]);
        }else{
          // A whole step contains thousands of dispatches. Finish each physical
          // stage before the next submission to bound driver resources (Firefox
          // on Metal loses the device with the single combined command buffer).
          // Buffer contents, kernels, accumulation order and RNG are unchanged.
          for(let i=0;i<stages.length;i++){
            check();const [label,encode]=stages[i],encoder=device.createCommandEncoder({label});encode(encoder);
            device.queue.submit([encoder.finish()]);
            if(i<stages.length-1){await device.queue.onSubmittedWorkDone();check();}
          }
        }
        return (await mappedValues(lossStage,1))[0];
      }catch(error){throw await reportedFailure(error);}
    }
    return {
      device,gpuField,rollout,objective:loss,field,
      async evaluate(n=0){lastLoss=await submitObjective(n);return lastLoss;},
      async step(){
        check();if(iteration>=iterations)throw new Error('All configured optimizer steps have completed.');
        lastLoss=await submitObjective(iteration,true);
        if(!Number.isFinite(lastLoss))throw new Error('The GPU waveform objective is nonfinite.');history.push(lastLoss);iteration++;
        return {iteration,loss:lastLoss};
      },
      async snapshot(){
        check();const encoder=device.createCommandEncoder();gpuField.encodeForward(encoder);device.queue.submit([encoder.finish()]);
        const areas=await read(gpuField.areaBuffer,geometry.frames*32,areaStage);
        if(!areas.every(Number.isFinite))throw new Error('The GPU area trajectory is nonfinite.');
        return {iteration,loss:lastLoss,initialLoss:history[0],history:[...history],areas,frames:geometry.frames,frameTimes,framePitchHz,frameVoiced,
          frameActive:new Uint8Array(geometry.frames).fill(1),length:17.5,duration:target.length/sampleRate,
          backend:'faithful-full-gpu',fieldMetadata:field.metadata,model:`Pinned diffTLM ${network.label} waveform objective; no auxiliary loss terms.`};
      },
      async synthesize(){
        check();const encoder=device.createCommandEncoder();gpuField.encodeForward(encoder);
        encoder.copyBufferToBuffer(gpuField.areaBuffer,0,rollout.areaBuffer,0,geometry.frames*32*4);rollout.encodeForward(encoder,0);
        device.queue.submit([encoder.finish()]);return read(rollout.waveformBuffer,target.length,waveStage);
      },
      readParameters(){return readTemporary(gpuField.parameterBuffer,field.parameterCount);},
      readGradient(){return readTemporary(gpuField.gradientBuffer,field.parameterCount);},
      dispose,
    };
  }catch(error){const reported=await reportedFailure(error);dispose();throw reported;}
}
