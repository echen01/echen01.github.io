// Port of difftlm/fit_sample.py's MultiResolutionSTFTLoss configuration.
// This objective compares synthesized audio with the target waveform. It has
// no area smoothness, gain elimination, harmonic sampling, or extra penalties.
import {fft} from './faithful-fft.js';

export const FAITHFUL_LOSS_CONFIG = Object.freeze({
  fftSizes: Object.freeze([2048, 1024, 512]),
  hopSizes: Object.freeze([160, 80, 40]),
  winLengths: Object.freeze([800, 400, 160]),
  melBands: 128,
  epsilon: 1e-8,
  window: 'symmetric Hann',
  boundary: 'zeros',
  padded: true,
  scale: 'Slaney mel',
  distance: 'L1 log10 magnitude',
});

const LOSS_CONFIG_16K = Object.freeze({...FAITHFUL_LOSS_CONFIG,
  fftSizes: Object.freeze([1024, 512, 256]),
  hopSizes: Object.freeze([80, 40, 20]),
  winLengths: Object.freeze([400, 200, 80]),
});

export function getFaithfulLossConfig(sampleRate = 32000) {
  if (sampleRate === 32000) return FAITHFUL_LOSS_CONFIG;
  if (sampleRate === 16000) return LOSS_CONFIG_16K;
  throw new RangeError('The faithful loss supports sample rates of 16000 or 32000 Hz.');
}

function hzToMel(hz) {
  return hz < 1000 ? hz / (200 / 3) : 15 + Math.log(hz / 1000) / (Math.log(6.4) / 27);
}

function melToHz(mel) {
  return mel < 15 ? (200 / 3) * mel : 1000 * Math.exp((Math.log(6.4) / 27) * (mel - 15));
}

/** librosa.filters.mel defaults: Slaney spacing, Slaney normalization, float32. */
export function createSlaneyMelBank(sampleRate, fftSize, melBands = 128) {
  if (!(Number.isFinite(sampleRate) && sampleRate > 0)) throw new Error('Invalid sample rate.');
  if (!Number.isInteger(fftSize) || fftSize < 2 || fftSize % 2) throw new Error('Expected an even FFT size.');
  if (!Number.isInteger(melBands) || melBands < 1) throw new Error('Invalid mel band count.');
  const binCount = fftSize / 2 + 1;
  const highestMel = hzToMel(sampleRate / 2);
  const edges = Float64Array.from({length: melBands + 2}, (_, i) => melToHz(i * highestMel / (melBands + 1)));
  const weights = new Float32Array(melBands * binCount);
  const sparse = [];
  for (let band = 0; band < melBands; band++) {
    const indices = [], values = [];
    const normalization = 2 / (edges[band + 2] - edges[band]);
    for (let bin = 0; bin < binCount; bin++) {
      const hz = bin * sampleRate / fftSize;
      const triangle = Math.max(0, Math.min(
        (hz - edges[band]) / (edges[band + 1] - edges[band]),
        (edges[band + 2] - hz) / (edges[band + 2] - edges[band + 1]),
      ));
      // librosa first assigns the triangle to its float32 output array, then
      // applies area normalization in-place, rounding to float32 a second time.
      const weight = Math.fround(Math.fround(triangle) * normalization);
      weights[band * binCount + bin] = weight;
      if (weight !== 0) { indices.push(bin); values.push(weight); }
    }
    sparse.push({indices: Uint32Array.from(indices), values: Float64Array.from(values)});
  }
  return {weights, sparse, binCount, melBands};
}

function createResolution(index, sampleCount, sampleRate, config) {
  const fftSize = config.fftSizes[index];
  const hopSize = config.hopSizes[index];
  const winLength = config.winLengths[index];
  const window = Float64Array.from({length: winLength}, (_, i) => .5 - .5 * Math.cos(2 * Math.PI * i / (winLength - 1)));
  const windowSum = window.reduce((sum, value) => sum + value, 0);
  // scipy.signal.stft extends by winLength/2 at each edge, and pads the tail
  // to complete the final hop. Its 'spectrum' scaling is 1 / sum(window).
  const scaledWindow = Float64Array.from(window, value => value / windowSum);
  return {
    fftSize, hopSize, winLength,
    frameCount: Math.ceil(sampleCount / hopSize) + 1,
    scaledWindow,
    bank: createSlaneyMelBank(sampleRate, fftSize),
    real: new Float64Array(fftSize),
    imag: new Float64Array(fftSize),
    magnitude: new Float64Array(fftSize / 2 + 1),
    magnitudeGradient: new Float64Array(fftSize / 2 + 1),
    adjointReal: new Float64Array(fftSize),
    adjointImag: new Float64Array(fftSize),
    mel: new Float64Array(FAITHFUL_LOSS_CONFIG.melBands),
  };
}

function forwardFrame(waveform, resolution, frame) {
  const {real, imag, scaledWindow, winLength, hopSize, magnitude, mel, bank} = resolution;
  real.fill(0); imag.fill(0);
  const start = frame * hopSize - winLength / 2;
  for (let i = 0; i < winLength; i++) {
    const sample = start + i;
    if (sample >= 0 && sample < waveform.length) real[i] = waveform[sample] * scaledWindow[i];
  }
  fft(real, imag);
  for (let bin = 0; bin < magnitude.length; bin++) magnitude[bin] = Math.hypot(real[bin], imag[bin]);
  for (let band = 0; band < bank.melBands; band++) {
    const {indices, values} = bank.sparse[band];
    let value = 0;
    for (let k = 0; k < indices.length; k++) value += values[k] * magnitude[indices[k]];
    mel[band] = value;
  }
}

function featuresAtResolution(waveform, resolution) {
  const {frameCount, bank, mel} = resolution;
  const magnitudes = new Float64Array(bank.melBands * frameCount);
  const logMagnitudes = new Float64Array(magnitudes.length);
  for (let frame = 0; frame < frameCount; frame++) {
    forwardFrame(waveform, resolution, frame);
    for (let band = 0; band < bank.melBands; band++) {
      const index = band * frameCount + frame;
      magnitudes[index] = mel[band];
      logMagnitudes[index] = Math.log10(mel[band] + FAITHFUL_LOSS_CONFIG.epsilon);
    }
  }
  // Matches the last two axes of upstream's (1, melBands, frameCount) arrays.
  return {magnitudes, logMagnitudes, melBands: bank.melBands, frameCount};
}

/**
 * Exact objective configured by upstream fit_sample(), evaluated in float64.
 * Features of the fixed target are computed once. Each evaluation returns a
 * newly owned waveform gradient; subsequent calls cannot overwrite it.
 */
export function createFaithfulLoss(target, sampleRate = 32000) {
  const config = getFaithfulLossConfig(sampleRate);
  if (!target || target.length < config.winLengths[0]) {
    throw new Error(`The waveform must contain at least ${config.winLengths[0]} samples, as required by the upstream STFT window.`);
  }
  if (!target.every(Number.isFinite)) throw new Error('Target waveform must be finite.');
  const sampleCount = target.length;
  const resolutions = config.fftSizes.map((_, index) => createResolution(index, sampleCount, sampleRate, config));
  const targetFeatures = resolutions.map(resolution => featuresAtResolution(target, resolution));

  function checkWaveform(waveform) {
    if (!waveform || waveform.length !== sampleCount) throw new Error('Predicted waveform must match the target length.');
    if (!waveform.every(Number.isFinite)) throw new Error('Predicted waveform must be finite.');
  }

  function features(waveform) {
    checkWaveform(waveform);
    return resolutions.map(resolution => featuresAtResolution(waveform, resolution));
  }

  function evaluate(waveform) {
    checkWaveform(waveform);
    const gradient = new Float64Array(sampleCount);
    const resolutionLosses = new Float64Array(resolutions.length);
    for (let level = 0; level < resolutions.length; level++) {
      const resolution = resolutions[level];
      const {fftSize, frameCount, hopSize, winLength, scaledWindow, bank,
        real, imag, magnitude, magnitudeGradient, adjointReal, adjointImag, mel} = resolution;
      const targetLogs = targetFeatures[level].logMagnitudes;
      const lossScale = 1 / (frameCount * bank.melBands);
      const gradientScale = lossScale / (resolutions.length * Math.LN10);
      let total = 0;
      for (let frame = 0; frame < frameCount; frame++) {
        forwardFrame(waveform, resolution, frame);
        magnitudeGradient.fill(0);
        for (let band = 0; band < bank.melBands; band++) {
          const denominator = mel[band] + FAITHFUL_LOSS_CONFIG.epsilon;
          const difference = targetLogs[band * frameCount + frame] - Math.log10(denominator);
          total += Math.abs(difference);
          // jax.grad(abs)(0) is +1. Upstream computes abs(target - predicted),
          // so its selected derivative at an exact tie is -1, not zero.
          const cotangent = (difference >= 0 ? -1 : 1) * gradientScale / denominator;
          const {indices, values} = bank.sparse[band];
          for (let k = 0; k < indices.length; k++) magnitudeGradient[indices[k]] += cotangent * values[k];
        }
        adjointReal.fill(0); adjointImag.fill(0);
        for (let bin = 0; bin < magnitude.length; bin++) {
          // JAX's complex abs uses the zero derivative at zero magnitude.
          const scale = magnitude[bin] === 0 ? 0 : magnitudeGradient[bin] / magnitude[bin];
          adjointReal[bin] = scale * real[bin];
          adjointImag[bin] = scale * imag[bin];
        }
        // Only positive frequency cotangents are present. Mirroring these
        // here would incorrectly double the rFFT adjoint's interior bins.
        fft(adjointReal, adjointImag, true);
        const start = frame * hopSize - winLength / 2;
        for (let i = 0; i < winLength; i++) {
          const sample = start + i;
          if (sample >= 0 && sample < sampleCount) gradient[sample] += fftSize * adjointReal[i] * scaledWindow[i];
        }
      }
      resolutionLosses[level] = total * lossScale;
    }
    const loss = resolutionLosses.reduce((sum, value) => sum + value, 0) / resolutions.length;
    return {loss, gradient, resolutionLosses};
  }

  return {evaluate, features, sampleRate, sampleCount, config,
    melBandDiagnostics: resolutions.map(resolution => ({fftSize: resolution.fftSize, melBands: resolution.bank.melBands,
      emptyBandIndices: resolution.bank.sparse.flatMap((band, index) => band.indices.length ? [] : [index])}))};
}
