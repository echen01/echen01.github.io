// Local, framewise speech reconstruction with the validated diffTLM WASM VJP.
// This is a harmonic/envelope approximation, not the paper's turbulence model
// or neural-field + multi-resolution waveform-spectrogram training pipeline.
import {initialAreas, NUM_AREAS} from '../simulator/index.js';
import {SAMPLE_RATE, FFT_SIZE, HOP_LENGTH, windowSpectrum, lfHarmonicMagnitudes} from './dsp.js';

export const DEFAULT_ITERATIONS = 32;
const AREA_MIN = .12, AREA_MAX = 16;
const wait = () => new Promise(resolve => setTimeout(resolve, 0));
function canceled(signal) { if (signal?.aborted) throw new DOMException('Speech reconstruction canceled.', 'AbortError'); }

function nearestPitch(pitch, time) {
  const times = pitch.times;
  let low = 0, high = times.length - 1;
  while (low < high) { const middle = (low + high) >>> 1; if (times[middle] < time) low = middle + 1; else high = middle; }
  let i = low;
  if (i > 0 && Math.abs(times[i - 1] - time) < Math.abs(times[i] - time)) i--;
  const f0 = pitch.pitchHz[i];
  return {f0, voiced: Boolean(pitch.voiced[i]) && Number.isFinite(f0) && f0 >= 40 && f0 <= 2200};
}

function bandMagnitude(magnitude, frequency, halfWidth, sampleRate, fftSize, harmonic = false) {
  const binHz = sampleRate / fftSize;
  const lower = Math.max(1, Math.ceil((frequency - halfWidth) / binHz));
  const upper = Math.min(magnitude.length - 2, Math.floor((frequency + halfWidth) / binHz));
  let energy = 0, count = 0;
  for (let k = lower; k <= upper; k++) { energy += magnitude[k] ** 2; count++; }
  if (!count) {
    const position = Math.max(1, Math.min(magnitude.length - 2, frequency / binHz));
    const left = Math.floor(position), blend = position - left;
    return magnitude[left] * (1 - blend) + magnitude[left + 1] * blend;
  }
  return Math.sqrt(energy / (harmonic ? 1 : count));
}

/** Analyze target recording once. Each frame stores a source-corrected target
 * spectrum and a bounded set of acoustic frequencies. These targets never change
 * while fitting. The original samples never become a reconstructed excitation.
 */
export async function prepareSpeech(samples, pitch, {sampleRate = SAMPLE_RATE, signal, onProgress} = {}) {
  if (!(samples instanceof Float32Array) || !samples.length || !samples.every(Number.isFinite)) throw new Error('Speech fitting requires finite mono PCM samples.');
  if (sampleRate !== SAMPLE_RATE) throw new Error('Speech fitting expects audio resampled to 16 kHz.');
  if (samples.length < sampleRate * .12 || samples.length > sampleRate * 10.01) throw new Error('Choose a recording between 0.12 and 10 seconds.');
  if (!pitch?.times?.length || pitch.pitchHz?.length !== pitch.times.length || pitch.voiced?.length !== pitch.times.length) throw new Error('A complete SwiftF0 pitch and voicing contour is required.');
  let peak = 0;
  for (const sample of samples) peak = Math.max(peak, Math.abs(sample));
  if (peak < 1e-6) throw new Error('The recording is silent. Choose audible speech.');
  const activeThreshold = Math.max(1e-6, peak * .002);
  const harmonicSource = lfHarmonicMagnitudes(128);
  const frames = [], frameTimes = [], framePitchHz = [], frameVoiced = [], frameActive = [];
  const count = Math.ceil(samples.length / HOP_LENGTH);
  let activeFrames = 0;
  for (let frame = 0; frame < count; frame++) {
    canceled(signal);
    const center = frame * HOP_LENGTH, time = center / sampleRate;
    const {magnitude} = windowSpectrum(samples, center, FFT_SIZE);
    let energy = 0, sampleCount = 0;
    for (let i = Math.max(0, center - HOP_LENGTH / 2); i < Math.min(samples.length, center + HOP_LENGTH / 2); i++) { energy += samples[i] ** 2; sampleCount++; }
    const rms = Math.sqrt(energy / Math.max(1, sampleCount));
    const active = rms >= activeThreshold;
    const {f0, voiced} = nearestPitch(pitch, time);
    const frequencies = [], observed = [], source = [];
    if (voiced) {
      const maximumHarmonic = Math.max(1, Math.min(128, Math.floor(5000 / f0)));
      const harmonics = Math.min(36, maximumHarmonic);
      for (let h = 0; h < harmonics; h++) {
        const harmonic = harmonics === 1 ? 1 : 1 + Math.round(h * (maximumHarmonic - 1) / (harmonics - 1));
        const frequency = harmonic * f0;
        frequencies.push(frequency);
        observed.push(bandMagnitude(magnitude, frequency, Math.min(40, .35 * f0), sampleRate, FFT_SIZE, true));
        source.push(Math.max(1e-10, harmonicSource[harmonic - 1]));
      }
    } else {
      // Smooth the noisy recording spectrum into modest-width bands; the source
      // spectrum is flat. This approximates unvoiced excitation at the glottis.
      for (let i = 0; i < 28; i++) {
        const frequency = 150 + i * (5000 - 150) / 27;
        frequencies.push(frequency);
        observed.push(bandMagnitude(magnitude, frequency, 75, sampleRate, FFT_SIZE));
        source.push(1);
      }
    }
    const maximum = Math.max(...observed, 1e-12);
    // Low-energy bins carry little usable shape information. A relative floor
    // avoids forcing the geometry to explain arbitrarily deep spectral nulls.
    const floor = Math.max(1e-10, maximum * .005);
    const targetDb = observed.map((value, i) => 20 * Math.log10(Math.max(value, floor) / source[i]));
    frames.push({frequencies, targetDb, active, rms, voiced, f0: voiced ? f0 : 0});
    frameTimes.push(time); framePitchHz.push(voiced ? f0 : 0); frameVoiced.push(Number(voiced)); frameActive.push(Number(active));
    activeFrames += Number(active);
    if (frame % 12 === 0) { onProgress?.((frame + 1) / count); await wait(); }
  }
  if (!activeFrames) throw new Error('No audible speech frames were found. Try a clearer recording.');
  return {frames, frameTimes, framePitchHz, frameVoiced, frameActive, activeFrames,
    sampleRate, fftSize: FFT_SIZE, hopLength: HOP_LENGTH, duration: samples.length / sampleRate};
}

/** Exact gain-eliminated spectral objective. The best log gain is the mean
 * target-minus-prediction difference. Its derivative cancels because residuals
 * sum to zero; the envelope theorem permits holding this optimum fixed in the
 * VJP. The optimized WASM path centers residuals and Jacobians in one pass.
 * The returned gradient is with respect to log areas.
 */
export function gainInvariantLossAndGradient(sim, areas, targetDb, frequencies, length) {
  if (sim.gainInvariantLossAndGradient) return sim.gainInvariantLossAndGradient(areas, targetDb, length, frequencies);
  // Preserve support for reference/backends implementing only the original API.
  const prediction = sim.response(areas, length, frequencies).magnitudeDb;
  const gainDb = targetDb.reduce((sum, value, i) => sum + value - prediction[i], 0) / targetDb.length;
  const correctedTarget = targetDb.map(value => value - gainDb);
  const result = sim.lossAndGradient(areas, correctedTarget, length, frequencies);
  return {...result, gainDb};
}

/** Squared first differences in log area, normalized independently in space and
 * time. The derivative includes both neighbors, including silent-frame links.
 */
export function regularization(logAreas, frameCount, {spatialWeight = .006, temporalWeight = .08} = {}) {
  const gradient = new Float64Array(logAreas.length);
  let spatialLoss = 0, temporalLoss = 0;
  const spatialScale = spatialWeight / (frameCount * (NUM_AREAS - 1));
  const temporalScale = frameCount > 1 ? temporalWeight / ((frameCount - 1) * NUM_AREAS) : 0;
  for (let f = 0; f < frameCount; f++) for (let i = 0; i < NUM_AREAS; i++) {
    const k = f * NUM_AREAS + i;
    if (i > 0) {
      const difference = logAreas[k] - logAreas[k - 1];
      spatialLoss += spatialScale * difference * difference;
      gradient[k] += 2 * spatialScale * difference; gradient[k - 1] -= 2 * spatialScale * difference;
    }
    if (f > 0) {
      const difference = logAreas[k] - logAreas[k - NUM_AREAS];
      temporalLoss += temporalScale * difference * difference;
      gradient[k] += 2 * temporalScale * difference; gradient[k - NUM_AREAS] -= 2 * temporalScale * difference;
    }
  }
  return {spatialLoss, temporalLoss, gradient};
}

export async function evaluateTrajectory(sim, prepared, logAreas, {length = 17.5, signal, spatialWeight = .006, temporalWeight = .08, yieldEveryFrames = 24} = {}) {
  const frameCount = prepared.frames.length;
  const penalty = regularization(logAreas, frameCount, {spatialWeight, temporalWeight});
  const gradient = penalty.gradient;
  const gainsDb = new Float64Array(frameCount);
  let dataLoss = 0;
  for (let f = 0; f < frameCount; f++) {
    canceled(signal);
    const frame = prepared.frames[f];
    if (frame.active) {
      const areas = Array.from(logAreas.subarray(f * NUM_AREAS, (f + 1) * NUM_AREAS), Math.exp);
      const evaluation = gainInvariantLossAndGradient(sim, areas, frame.targetDb, frame.frequencies, length);
      dataLoss += evaluation.loss / prepared.activeFrames;
      gainsDb[f] = evaluation.gainDb;
      for (let i = 0; i < NUM_AREAS; i++) gradient[f * NUM_AREAS + i] += evaluation.gradient[i] / prepared.activeFrames;
    }
    if (yieldEveryFrames > 0 && (f + 1) % yieldEveryFrames === 0) await wait();
  }
  const totalLoss = dataLoss + penalty.spatialLoss + penalty.temporalLoss;
  if (!Number.isFinite(totalLoss) || !gradient.every(Number.isFinite)) throw new Error('Speech fitting encountered an unstable acoustic response. Try a shorter recording.');
  return {dataLoss, spatialLoss: penalty.spatialLoss, temporalLoss: penalty.temporalLoss, totalLoss, gradient, gainsDb};
}

export async function createSpeechFitter(sim, prepared, {length = 17.5, learningRate = .045, spatialWeight = .006, temporalWeight = .08, signal} = {}) {
  if (!Number.isFinite(length) || length < 12 || length > 22) throw new Error('Choose a tract length between 12 and 22 cm.');
  if (!Number.isFinite(learningRate) || learningRate <= 0 || learningRate > .2) throw new Error('Invalid speech fitting learning rate.');
  const frameCount = prepared.frames.length, base = initialAreas();
  const logs = Float64Array.from({length: frameCount * NUM_AREAS}, (_, k) => Math.log(base[k % NUM_AREAS]));
  const m = new Float64Array(logs.length), v = new Float64Array(logs.length);
  const settings = {length, signal, spatialWeight, temporalWeight};
  let evaluation = await evaluateTrajectory(sim, prepared, logs, settings), iteration = 0;
  const initialLoss = evaluation.totalLoss, history = [initialLoss];
  const snapshot = () => ({iteration, loss: evaluation.totalLoss, totalLoss: evaluation.totalLoss,
    dataLoss: evaluation.dataLoss, spatialLoss: evaluation.spatialLoss, temporalLoss: evaluation.temporalLoss,
    initialLoss, history: [...history], areas: Float64Array.from(logs, Math.exp),
    frames: frameCount, frameTimes: [...prepared.frameTimes], framePitchHz: [...prepared.framePitchHz],
    frameVoiced: [...prepared.frameVoiced], frameActive: [...prepared.frameActive],
    activeFrames: prepared.activeFrames, gainDb: Array.from(evaluation.gainsDb), length,
    framePeriod: prepared.hopLength / prepared.sampleRate, duration: prepared.duration});
  return {
    snapshot,
    async step() {
      canceled(signal);
      iteration++;
      let normSquared = 0;
      for (const g of evaluation.gradient) normSquared += g * g;
      const clip = Math.min(1, 5 / Math.max(1e-12, Math.sqrt(normSquared)));
      const mScale = 1 - Math.pow(.9, iteration), vScale = 1 - Math.pow(.999, iteration);
      for (let k = 0; k < logs.length; k++) {
        const g = evaluation.gradient[k] * clip;
        m[k] = .9 * m[k] + .1 * g; v[k] = .999 * v[k] + .001 * g * g;
        logs[k] = Math.max(Math.log(AREA_MIN), Math.min(Math.log(AREA_MAX), logs[k] - learningRate * (m[k] / mScale) / (Math.sqrt(v[k] / vScale) + 1e-8)));
      }
      evaluation = await evaluateTrajectory(sim, prepared, logs, settings);
      history.push(evaluation.totalLoss);
      return snapshot();
    },
  };
}

export const MODEL_DESCRIPTION = {
  name: 'Framewise harmonic / spectral-envelope reconstruction',
  areasPerFrame: NUM_AREAS,
  excitation: 'Fixed SwiftF0 contour; periodic LF source in voiced regions; independent broadband noise in unvoiced regions',
  objective: 'Mean gain-invariant natural-log spectral error plus spatial and temporal log-area smoothness',
  limitations: 'Approximate unvoiced source; no localized turbulence, learned neural field, waveform phase loss, or anatomical uniqueness guarantee',
};
