/** LF glottal source and RMS controls from pinned diffTLM glottis.py/fit_sample.py.
 * The bundled example uses exported original JAX float32 source samples. This
 * implementation supplies the same source equations for uploaded recordings.
 */
import {createLfFlowPulse} from '../lf-source.js';
import {fractalNoise, jaxKey} from './faithful-noise.js';

export function normalizeFaithfulTarget(samples) {
  if (!samples.length || !samples.every(Number.isFinite)) throw new Error('Expected finite, nonempty audio.');
  const result = Float32Array.from(samples);
  const mean = result.reduce((sum, value) => sum + value, 0) / result.length;
  let peak = 0;
  for (let i = 0; i < result.length; i++) { result[i] -= mean; peak = Math.max(peak, Math.abs(result[i])); }
  for (let i = 0; i < result.length; i++) result[i] /= peak + 1e-8;
  return result;
}

/** librosa.feature.rms(y, frame_length=2048, hop_length=320), center=True,
 * constant-zero padding, followed by fit_sample's peak normalization.
 */
export function faithfulRms(samples, {frameLength = 2048, hopLength = 320, normalize = true} = {}) {
  if (!samples.length || !samples.every(Number.isFinite)) throw new Error('Expected finite, nonempty audio.');
  if (!Number.isInteger(frameLength) || frameLength < 2 || frameLength % 2 || !Number.isInteger(hopLength) || hopLength < 1) {
    throw new Error('Invalid RMS frame/hop length.');
  }
  const result = new Float32Array(Math.floor(samples.length / hopLength) + 1);
  let peak = 0;
  for (let frame = 0; frame < result.length; frame++) {
    const start = frame * hopLength - frameLength / 2;
    let sum = 0;
    for (let i = Math.max(0, start); i < Math.min(samples.length, start + frameLength); i++) {
      sum += Math.fround(samples[i] * samples[i]);
    }
    result[frame] = Math.sqrt(Math.fround(sum / frameLength));
    peak = Math.max(peak, result[frame]);
  }
  if (normalize) for (let i = 0; i < result.length; i++) result[i] /= peak + 1e-8;
  return result;
}

export function interpolateUniform(controls, sampleCount) {
  if (!controls.length || !controls.every(Number.isFinite) || !Number.isInteger(sampleCount) || sampleCount < 1) {
    throw new Error('Uniform interpolation needs finite controls and a positive sample count.');
  }
  const result = new Float64Array(sampleCount);
  if (controls.length === 1) { result.fill(controls[0]); return result; }
  for (let i = 0; i < sampleCount; i++) {
    const position = sampleCount === 1 ? 0 : i * (controls.length - 1) / (sampleCount - 1);
    const left = Math.min(Math.floor(position), controls.length - 2);
    result[i] = controls[left] + (position - left) * (controls[left + 1] - controls[left]);
  }
  return result;
}

export function faithfulLfPulse(pitchHz, rms, opening, {
  sampleRate = 32000, sampleCount, key = jaxKey(0), dtype = 32,
  openQuotient = .5, shapeQuotient = 3, spectralTilt = .02,
} = {}) {
  if (!(sampleRate > 0) || !Number.isInteger(sampleCount) || sampleCount < 16) throw new Error('Invalid LF sample rate/count.');
  const pulseAt = createLfFlowPulse({openQuotient, shapeQuotient, spectralTilt});
  const pitch = interpolateUniform(pitchHz, sampleCount), envelope = interpolateUniform(rms, sampleCount);
  const glottis = interpolateUniform(opening, sampleCount);
  // glottis.py deliberately uses nf//8, distinct from rollout noise's nf//64.
  const noise = fractalNoise(sampleCount, {res: Math.floor(sampleCount / 8), octaves: 5, persistence: .5, lacunarity: 2, key, dtype});
  const result = dtype === 64 ? new Float64Array(sampleCount) : new Float32Array(sampleCount);
  let phase = 0;
  for (let i = 0; i < sampleCount; i++) {
    phase += Math.max(0, Math.min(pitch[i], sampleRate / 2)) / sampleRate;
    const t = phase - Math.floor(phase);
    const pulse = pulseAt(t);
    result[i] = (pulse + (noise[i] + 1) * glottis[i]) * envelope[i];
  }
  return result;
}

export function createFaithfulSource(samples, pitch, options = {}) {
  const sampleRate = options.sampleRate ?? 32000;
  if (![16000, 32000].includes(sampleRate)) throw new Error('The faithful fit_sample source supports 16 or 32 kHz audio.');
  if (pitch.pitchHz.length !== pitch.confidence.length) throw new Error('Pitch and confidence controls must have equal length.');
  // Preserve the original 64 ms RMS window and 10 ms hop at either rate.
  const rms = faithfulRms(samples, {frameLength: sampleRate * .064, hopLength: sampleRate * .01});
  const opening = Float64Array.from(pitch.confidence, confidence => 1 - confidence);
  const uin = faithfulLfPulse(pitch.pitchHz, rms, opening, {...options, sampleRate, sampleCount: samples.length});
  return {uin, rms};
}
