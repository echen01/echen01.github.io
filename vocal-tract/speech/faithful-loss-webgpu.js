// The fit_sample.py waveform objective, evaluated entirely on WebGPU after
// one-time target feature preparation. Fixed-size batches bound FFT scratch
// memory independently of recording length; each batch contributes its exact
// window overlap-add derivative before its scratch buffers are reused.
import {createFaithfulLoss, createSlaneyMelBank, getFaithfulLossConfig} from './faithful-loss.js';
import {createGpuFft} from './faithful-fft-webgpu.js';

const BATCH = 64;
const WORKGROUP = 128;

export async function createFaithfulGpuLoss(device, target, {
  sampleRate = 32000, fftOptions = {}, scalingMode = 'before-fft', targetMode = 'cpu-f64',
  stftWindows, stftScales, targetLogMagnitudes, diagnostics = false, compensatedWindow = false,
} = {}) {
  if (!['before-fft', 'after-fft'].includes(scalingMode)) throw new Error('Unknown STFT scaling mode.');
  if (!['cpu-f64', 'gpu'].includes(targetMode)) throw new Error('Unknown STFT target mode.');
  if (compensatedWindow && fftOptions.arithmetic !== 'double-single') throw new Error('Compensated STFT windows require double-single FFT arithmetic.');
  const config = getFaithfulLossConfig(sampleRate);
  const reference = createFaithfulLoss(target, sampleRate);
  const targetFeatures = reference.features(target);
  const sampleCount = target.length;
  const ownedBuffers = [], ffts = [];
  let destroyed = false;
  function buffer(size, label, extraUsage = 0) {
    const result = device.createBuffer({size: Math.max(4, Math.ceil(size / 4) * 4),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC | extraUsage,
      label});
    ownedBuffers.push(result);
    return result;
  }
  function upload(array, label, extraUsage = 0) {
    const result = buffer(array.byteLength, label, extraUsage);
    device.queue.writeBuffer(result, 0, array);
    return result;
  }
  async function pipeline(code, label) {
    const module = device.createShaderModule({code, label});
    const info = await module.getCompilationInfo();
    const errors = info.messages.filter(message => message.type === 'error');
    if (errors.length) throw new Error(`${label}: ${errors.map(error => error.message).join('; ')}`);
    return await device.createComputePipelineAsync({layout: 'auto', compute: {module, entryPoint: 'main'}, label});
  }
  function group(computePipeline, buffers) {
    return device.createBindGroup({layout: computePipeline.getBindGroupLayout(0),
      entries: buffers.map((resource, binding) => ({binding, resource: resource.buffer ? resource : {buffer: resource}}))});
  }
  function dispatch(encoder, computePipeline, bindings, count) {
    const pass = encoder.beginComputePass();
    pass.setPipeline(computePipeline); pass.setBindGroup(0, bindings); pass.dispatchWorkgroups(count); pass.end();
  }

  const gradientBuffer = buffer(sampleCount * 4, 'faithful waveform loss gradient');
  const lossBuffer = buffer(4, 'faithful waveform loss');
  const chunkCount = targetFeatures.reduce((sum, feature) => sum + Math.ceil(feature.frameCount / BATCH), 0);
  const partialLosses = buffer(chunkCount * 4, 'faithful STFT batch losses');
  const resolutions = [];
  let partialIndex = 0;
  try {
    for (let level = 0; level < config.fftSizes.length; level++) {
      const size = config.fftSizes[level];
      const hop = config.hopSizes[level];
      const win = config.winLengths[level];
      const frames = targetFeatures[level].frameCount;
      const bins = size / 2 + 1;
      const bank = createSlaneyMelBank(sampleRate, size);
      const window = stftWindows?.[level] ?? Float64Array.from({length: win}, (_, i) => .5 - .5 * Math.cos(2 * Math.PI * i / (win - 1)));
      if (window.length !== win || !window.every(Number.isFinite)) throw new Error(`Invalid STFT window ${level}.`);
      const sum = window.reduce((a, b) => a + b, 0);
      const spectrumScale = stftScales?.[level] ?? 1 / sum;
      if (!Number.isFinite(spectrumScale) || spectrumScale <= 0) throw new Error(`Invalid STFT scale ${level}.`);
      const postFftScale = scalingMode === 'after-fft' ? Math.fround(spectrumScale) : 1;
      const windowValues = Float32Array.from(window, value => scalingMode === 'before-fft'
        ? (stftScales?.[level] === undefined ? value / sum : value * spectrumScale) : value);
      const scaledWindow = upload(windowValues, `STFT ${size} window`);
      const weights = upload(bank.weights, `STFT ${size} Slaney filters`);
      const ranges = new Uint32Array(128 * 2);
      const binBands = new Uint32Array(bins * 2), binWeights = new Float32Array(bins * 2);
      for (let band = 0; band < 128; band++) {
        const {indices, values} = bank.sparse[band];
        ranges[2 * band] = indices.length ? indices[0] : 0;
        ranges[2 * band + 1] = indices.length ? indices[indices.length - 1] + 1 : 0;
        for (let k = 0; k < indices.length; k++) {
          const index = indices[k] * 2 + (binWeights[indices[k] * 2] !== 0 ? 1 : 0);
          if (binWeights[index] !== 0) throw new Error('A frequency bin belongs to more than two triangular mel filters.');
          binBands[index] = band; binWeights[index] = values[k];
        }
      }
      const rangeBuffer = upload(ranges, `STFT ${size} mel ranges`);
      const binBandsBuffer = upload(binBands, `STFT ${size} mel transpose indices`);
      const binWeightsBuffer = upload(binWeights, `STFT ${size} mel transpose weights`);
      const targetLogValues = targetLogMagnitudes?.[level] ?? targetFeatures[level].logMagnitudes;
      if (targetLogValues.length !== frames * 128 || !targetLogValues.every(Number.isFinite)) throw new Error(`Invalid target log-mel features ${level}.`);
      const targetLogs = upload(Float32Array.from(targetLogValues), `STFT ${size} target log-mel`);
      const inputComplex = buffer(BATCH * size * 8, `STFT ${size} window / frequency adjoint`);
      const forwardInput = compensatedWindow ? buffer(BATCH * size * 16, `STFT ${size} compensated window`) : inputComplex;
      const spectrum = buffer(BATCH * size * 8, `STFT ${size} spectrum`);
      const timeAdjoint = buffer(BATCH * size * 8, `STFT ${size} time adjoint`);
      const melGradient = buffer(BATCH * 128 * 4, `STFT ${size} mel gradient`);
      const errors = buffer(BATCH * 128 * 4, `STFT ${size} log-mel errors`);
      const forwardFft = await createGpuFft(device, {...fftOptions, inputFormat: compensatedWindow ? 'complex-pairs' : 'complex32', size, batchCount: BATCH, inverse: false});
      ffts.push(forwardFft);
      const backwardFft = await createGpuFft(device, {...fftOptions, inputFormat: 'complex32', size, batchCount: BATCH, inverse: true});
      ffts.push(backwardFft);
      const diagnosticMagnitude = diagnostics ? buffer(frames * 128 * 4, `STFT ${size} diagnostic mel magnitude`) : null;
      const diagnosticLogs = diagnostics ? buffer(frames * 128 * 4, `STFT ${size} diagnostic log-mel`) : null;
      const diagnosticSpectrum = diagnostics ? buffer(frames * bins * 8, `STFT ${size} diagnostic complex spectrum`) : null;
      const params = `struct Params { firstFrame:u32, frameCount:u32, firstSample:u32, lastSample:u32, partialIndex:u32, _pad0:u32, _pad1:u32, _pad2:u32 };`;
      const windowPipeline = await pipeline(`${params}
@group(0) @binding(0) var<storage,read> waveform:array<f32>;
@group(0) @binding(1) var<storage,read> window:array<f32>;
@group(0) @binding(2) var<storage,read_write> output:array<${compensatedWindow ? 'vec4f' : 'vec2f'}>;
@group(0) @binding(3) var<uniform> p:Params;
@compute @workgroup_size(${WORKGROUP}) fn main(@builtin(global_invocation_id) id:vec3u) {
 let index=id.x; if(index>=${BATCH * size}u){return;}
 let frame=index/${size}u; let offset=index%${size}u;
 var value=0.;${compensatedWindow ? 'var residual=0.;' : ''}
 if(frame<p.frameCount && offset<${win}u){
  let sample=i32((p.firstFrame+frame)*${hop}u+offset)-${win / 2};
  if(sample>=0 && sample<${sampleCount}){
   value=waveform[u32(sample)]*window[offset];
   ${compensatedWindow ? `// _pad0 is an uploaded zero: preserve the product rounding before fma.
   value=bitcast<f32>(bitcast<u32>(value)^p._pad0);
   residual=fma(waveform[u32(sample)],window[offset],-value);` : ''}
  }
 }
 output[index]=${compensatedWindow ? 'vec4f(value,residual,0.,0.)' : 'vec2f(value,0.)'};
}`, `STFT ${size} window kernel`);
      const scalingPipeline = scalingMode === 'after-fft' ? await pipeline(`
@group(0) @binding(0) var<storage,read_write> spectrum:array<vec2f>;
@compute @workgroup_size(${WORKGROUP}) fn main(@builtin(global_invocation_id) id:vec3u) {
 if(id.x<${BATCH * size}u){spectrum[id.x]*=${postFftScale};}
}`, `STFT ${size} spectrum scaling`) : null;
      const scalingBindings = scalingPipeline ? group(scalingPipeline, [spectrum]) : null;
      const melPipeline = await pipeline(`${params}
@group(0) @binding(0) var<storage,read> spectrum:array<vec2f>;
@group(0) @binding(1) var<storage,read> weights:array<f32>;
@group(0) @binding(2) var<storage,read> ranges:array<vec2u>;
@group(0) @binding(3) var<storage,read> targetLogs:array<f32>;
@group(0) @binding(4) var<storage,read_write> gradient:array<f32>;
@group(0) @binding(5) var<storage,read_write> errors:array<f32>;
@group(0) @binding(6) var<uniform> p:Params;
${diagnostics ? '@group(0) @binding(7) var<storage,read_write> diagnosticMagnitude:array<f32>;\n@group(0) @binding(8) var<storage,read_write> diagnosticLogs:array<f32>;' : ''}
@compute @workgroup_size(${WORKGROUP}) fn main(@builtin(global_invocation_id) id:vec3u) {
 let index=id.x; if(index>=${BATCH * 128}u){return;}
 let frame=index/128u; let band=index%128u;
 if(frame>=p.frameCount){gradient[index]=0.;errors[index]=0.;return;}
 var magnitude=0.;let range=ranges[band];
 for(var bin=range.x;bin<range.y;bin++){magnitude+=weights[band*${bins}u+bin]*length(spectrum[frame*${size}u+bin]);}
 let denominator=magnitude+1e-8;
 let predictionLog=log(denominator)/2.302585092994046;
 let difference=targetLogs[band*${frames}u+p.firstFrame+frame]-predictionLog;
 ${diagnostics ? `diagnosticMagnitude[band*${frames}u+p.firstFrame+frame]=magnitude;diagnosticLogs[band*${frames}u+p.firstFrame+frame]=predictionLog;` : ''}
 errors[index]=abs(difference)/${3 * 128 * frames}.;
 gradient[index]=select(1.,-1.,difference>=0.)/(${3 * 128 * frames}. * 2.302585092994046 * denominator);
}`, `STFT ${size} mel objective kernel`);
      const targetPipeline = targetMode === 'gpu' && !targetLogMagnitudes ? await pipeline(`${params}
@group(0) @binding(0) var<storage,read> spectrum:array<vec2f>;
@group(0) @binding(1) var<storage,read> weights:array<f32>;
@group(0) @binding(2) var<storage,read> ranges:array<vec2u>;
@group(0) @binding(3) var<storage,read_write> targetLogs:array<f32>;
@group(0) @binding(4) var<uniform> p:Params;
@compute @workgroup_size(${WORKGROUP}) fn main(@builtin(global_invocation_id) id:vec3u) {
 let index=id.x;if(index>=${BATCH * 128}u){return;}let frame=index/128u;let band=index%128u;
 if(frame>=p.frameCount){return;}var magnitude=0.;let range=ranges[band];
 for(var bin=range.x;bin<range.y;bin++){magnitude+=weights[band*${bins}u+bin]*length(spectrum[frame*${size}u+bin]);}
 targetLogs[band*${frames}u+p.firstFrame+frame]=log(magnitude+1e-8)/2.302585092994046;
}`, `STFT ${size} target features`) : null;
      const diagnosticSpectrumPipeline = diagnostics ? await pipeline(`${params}
@group(0) @binding(0) var<storage,read> spectrum:array<vec2f>;
@group(0) @binding(1) var<storage,read_write> output:array<vec2f>;
@group(0) @binding(2) var<uniform> p:Params;
@compute @workgroup_size(${WORKGROUP}) fn main(@builtin(global_invocation_id) id:vec3u) {
 let i=id.x;if(i>=p.frameCount*${bins}u){return;}
 let frame=i/${bins}u;let bin=i%${bins}u;
 output[(p.firstFrame+frame)*${bins}u+bin]=spectrum[frame*${size}u+bin];
}`, `STFT ${size} diagnostic spectrum`) : null;
      const spectrumGradientPipeline = await pipeline(`
@group(0) @binding(0) var<storage,read> spectrum:array<vec2f>;
@group(0) @binding(1) var<storage,read> melGradient:array<f32>;
@group(0) @binding(2) var<storage,read> bands:array<vec2u>;
@group(0) @binding(3) var<storage,read> weights:array<vec2f>;
@group(0) @binding(4) var<storage,read_write> gradient:array<vec2f>;
@compute @workgroup_size(${WORKGROUP}) fn main(@builtin(global_invocation_id) id:vec3u) {
 let index=id.x;if(index>=${BATCH * size}u){return;}
 let frame=index/${size}u;let bin=index%${size}u;var value=vec2f(0.);
 if(bin<${bins}u){
  let z=spectrum[index];let magnitude=length(z);
  if(magnitude>0.){
   let b=bands[bin];let w=weights[bin];
   let g=melGradient[frame*128u+b.x]*w.x+melGradient[frame*128u+b.y]*w.y;
   value=z*(g/magnitude);
  }
 }
 gradient[index]=value*${postFftScale};
}`, `STFT ${size} spectral adjoint kernel`);
      const overlapPipeline = await pipeline(`${params}
@group(0) @binding(0) var<storage,read> adjoint:array<vec2f>;
@group(0) @binding(1) var<storage,read> window:array<f32>;
@group(0) @binding(2) var<storage,read_write> gradient:array<f32>;
@group(0) @binding(3) var<uniform> p:Params;
@compute @workgroup_size(${WORKGROUP}) fn main(@builtin(global_invocation_id) id:vec3u) {
 let sample=p.firstSample+id.x;if(sample>=p.lastSample){return;}
 // Exactly one invocation owns each output sample. Batches dispatch in order,
 // so shared boundary samples accumulate without float atomics or races.
 let first=max(i32(p.firstFrame),(i32(sample)-${win / 2}+${hop})/${hop});
 let last=min(i32(p.firstFrame+p.frameCount)-1,(i32(sample)+${win / 2})/${hop});
 var value=0.;
 for(var frame=first;frame<=last;frame++){
  let offset=i32(sample)-frame*${hop}+${win / 2};
  if(offset>=0 && offset<${win}){value+=adjoint[u32(frame-i32(p.firstFrame))*${size}u+u32(offset)].x * ${size}. * window[u32(offset)];}
 }
 gradient[sample]+=value;
}`, `STFT ${size} waveform adjoint kernel`);
      const reductionPipeline = await pipeline(`${params}
@group(0) @binding(0) var<storage,read> errors:array<f32>;
@group(0) @binding(1) var<storage,read_write> losses:array<f32>;
@group(0) @binding(2) var<uniform> p:Params;
var<workgroup> sums:array<f32,128>;
@compute @workgroup_size(128) fn main(@builtin(local_invocation_id) id:vec3u) {
 var sum=0.;for(var i=id.x;i<${BATCH * 128}u;i+=128u){sum+=errors[i];}
 sums[id.x]=sum;workgroupBarrier();
 for(var stride=64u;stride>0u;stride/=2u){if(id.x<stride){sums[id.x]+=sums[id.x+stride];}workgroupBarrier();}
 if(id.x==0u){losses[p.partialIndex]=sums[0];}
}`, `STFT ${size} loss reduction kernel`);
      const spectralBindings = group(spectrumGradientPipeline, [spectrum, melGradient, binBandsBuffer, binWeightsBuffer, inputComplex]);
      const chunks = [];
      for (let firstFrame = 0; firstFrame < frames; firstFrame += BATCH) {
        const frameCount = Math.min(BATCH, frames - firstFrame);
        const firstSample = Math.max(0, firstFrame * hop - win / 2);
        const lastSample = Math.min(sampleCount, (firstFrame + frameCount - 1) * hop + win / 2);
        const metadata = upload(new Uint32Array([firstFrame, frameCount, firstSample, lastSample, partialIndex++, 0, 0, 0]),
          `STFT ${size} batch ${firstFrame}`, GPUBufferUsage.UNIFORM);
        chunks.push({metadata, firstSample, lastSample,
          melBindings: group(melPipeline, [spectrum, weights, rangeBuffer, targetLogs, melGradient, errors, metadata,
            ...(diagnostics ? [diagnosticMagnitude, diagnosticLogs] : [])]),
          targetBindings: targetPipeline ? group(targetPipeline, [spectrum, weights, rangeBuffer, targetLogs, metadata]) : null,
          diagnosticSpectrumBindings: diagnostics ? group(diagnosticSpectrumPipeline, [spectrum, diagnosticSpectrum, metadata]) : null,
          overlapBindings: group(overlapPipeline, [timeAdjoint, scaledWindow, gradientBuffer, metadata]),
          reductionBindings: group(reductionPipeline, [errors, partialLosses, metadata]),
          waveformBindings: new WeakMap(),
        });
      }
      resolutions.push({size, frames, bins, scaledWindow, windowValues, spectrumScale, targetLogs,
        inputComplex, forwardInput, spectrum, timeAdjoint, forwardFft, backwardFft,
        windowPipeline, scalingPipeline, scalingBindings, melPipeline, targetPipeline,
        diagnosticSpectrumPipeline, diagnosticMagnitude, diagnosticLogs, diagnosticSpectrum,
        spectrumGradientPipeline, overlapPipeline, reductionPipeline, spectralBindings, chunks});
    }
    const finalPipeline = await pipeline(`
@group(0) @binding(0) var<storage,read> partial:array<f32>;
@group(0) @binding(1) var<storage,read_write> output:array<f32>;
@compute @workgroup_size(1) fn main(){var sum=0.;for(var i=0u;i<${chunkCount}u;i++){sum+=partial[i];}output[0]=sum;}
`, 'multiresolution waveform loss sum');
    const finalBindings = group(finalPipeline, [partialLosses, lossBuffer]);

    function encodeSpectrum(encoder, resolution, chunk, waveformBuffer) {
      const {size, windowPipeline, scaledWindow, forwardInput, spectrum, forwardFft, scalingPipeline, scalingBindings} = resolution;
      let windowBindings = chunk.waveformBindings.get(waveformBuffer);
      if (!windowBindings) {
        windowBindings = group(windowPipeline, [waveformBuffer, scaledWindow, forwardInput, chunk.metadata]);
        chunk.waveformBindings.set(waveformBuffer, windowBindings);
      }
      dispatch(encoder, windowPipeline, windowBindings, Math.ceil(BATCH * size / WORKGROUP));
      forwardFft.encode(encoder, forwardInput, spectrum);
      if (scalingPipeline) dispatch(encoder, scalingPipeline, scalingBindings, Math.ceil(BATCH * size / WORKGROUP));
    }
    if (targetMode === 'gpu' && !targetLogMagnitudes) {
      const targetBuffer = upload(Float32Array.from(target), 'STFT target waveform');
      const encoder = device.createCommandEncoder();
      for (const resolution of resolutions) for (const chunk of resolution.chunks) {
        encodeSpectrum(encoder, resolution, chunk, targetBuffer);
        dispatch(encoder, resolution.targetPipeline, chunk.targetBindings, BATCH);
      }
      device.queue.submit([encoder.finish()]);
      await device.queue.onSubmittedWorkDone();
    }
    function encodeForward(encoder, waveformBuffer) {
      if (destroyed) throw new Error('GPU waveform loss has been destroyed.');
      encoder.clearBuffer(gradientBuffer);
      for (const resolution of resolutions) {
        const {size, backwardFft, melPipeline, spectrumGradientPipeline,
          overlapPipeline, reductionPipeline, spectralBindings, inputComplex, timeAdjoint, diagnosticSpectrumPipeline, bins} = resolution;
        for (const chunk of resolution.chunks) {
          encodeSpectrum(encoder, resolution, chunk, waveformBuffer);
          if (diagnosticSpectrumPipeline) dispatch(encoder, diagnosticSpectrumPipeline, chunk.diagnosticSpectrumBindings, Math.ceil(BATCH * bins / WORKGROUP));
          dispatch(encoder, melPipeline, chunk.melBindings, BATCH);
          dispatch(encoder, reductionPipeline, chunk.reductionBindings, 1);
          dispatch(encoder, spectrumGradientPipeline, spectralBindings, Math.ceil(BATCH * size / WORKGROUP));
          backwardFft.encode(encoder, inputComplex, timeAdjoint);
          dispatch(encoder, overlapPipeline, chunk.overlapBindings, Math.ceil((chunk.lastSample - chunk.firstSample) / WORKGROUP));
        }
      }
      dispatch(encoder, finalPipeline, finalBindings, 1);
    }
    return {
      sampleCount, sampleRate, lossBuffer, gradientBuffer,
      backend: 'webgpu-f32', config,
      melBandDiagnostics: reference.melBandDiagnostics,
      numericalOptions: {fftOptions, scalingMode, targetMode, compensatedWindow, suppliedWindows: !!stftWindows, suppliedScales: !!stftScales,
        suppliedTargetLogMagnitudes: !!targetLogMagnitudes},
      diagnosticResolutions: diagnostics ? resolutions.map(resolution => ({
        fftSize: resolution.size, frames: resolution.frames, bins: resolution.bins, windowValues: resolution.windowValues,
        spectrumScale: resolution.spectrumScale, stftBuffer: resolution.diagnosticSpectrum,
        melMagnitudeBuffer: resolution.diagnosticMagnitude, logMelBuffer: resolution.diagnosticLogs,
        targetLogBuffer: resolution.targetLogs,
        stftOrder: 'frame-major complex float32 pairs', melOrder: 'band-major float32',
      })) : undefined,
      scratchBytes: resolutions.reduce((sum, resolution) => sum + BATCH * resolution.size * 8 * (compensatedWindow ? 5 : 3) + BATCH * 128 * 4 * 2, 0)
        + ffts.reduce((sum, item) => sum + (item.scratchBytes ?? 0), 0),
      encodeForward,
      // The scalar loss and its waveform derivative are fused so each batch's
      // FFT scratch can be released immediately. No second encode is required.
      encodeBackward() { if (destroyed) throw new Error('GPU waveform loss has been destroyed.'); },
      destroy() {
        if (destroyed) return;
        destroyed = true;
        for (const item of ffts) item.destroy?.();
        for (const item of ownedBuffers) item.destroy();
      },
    };
  } catch (error) {
    for (const item of ffts) item.destroy?.();
    for (const item of ownedBuffers) item.destroy();
    throw error;
  }
}
