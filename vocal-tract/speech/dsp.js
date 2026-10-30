import {lfPulseCycle} from '../lf-source.js';

// Small, dependency-free DSP helpers for the browser speech experiment.
// This is a source-filter approximation: unvoiced excitation is deterministic
// broadband noise at the glottis, not the paper's distributed turbulence model.
export const SAMPLE_RATE = 16000;
export const FFT_SIZE = 1024;
export const HOP_LENGTH = 320;

function checkFFTSize(size) {
  if (!Number.isInteger(size) || size < 2 || (size & (size - 1)) !== 0) {
    throw new Error('FFT size must be a power of two.');
  }
}

/** In-place radix-2 FFT. The forward transform is unnormalized; inverse is 1/N. */
export function fft(real, imag, inverse = false) {
  const n = real.length;
  checkFFTSize(n);
  if (imag.length !== n) throw new Error('FFT real and imaginary lengths differ.');
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    while (j & bit) { j ^= bit; bit >>= 1; }
    j ^= bit;
    if (i < j) {
      [real[i], real[j]] = [real[j], real[i]];
      [imag[i], imag[j]] = [imag[j], imag[i]];
    }
  }
  for (let size = 2; size <= n; size *= 2) {
    const angle = (inverse ? 2 : -2) * Math.PI / size;
    const stepR = Math.cos(angle), stepI = Math.sin(angle);
    for (let start = 0; start < n; start += size) {
      let wr = 1, wi = 0;
      for (let k = 0; k < size / 2; k++) {
        const a = start + k, b = a + size / 2;
        const br = real[b] * wr - imag[b] * wi;
        const bi = real[b] * wi + imag[b] * wr;
        real[b] = real[a] - br; imag[b] = imag[a] - bi;
        real[a] += br; imag[a] += bi;
        const nextR = wr * stepR - wi * stepI;
        wi = wr * stepI + wi * stepR; wr = nextR;
      }
    }
  }
  if (inverse) for (let i = 0; i < n; i++) { real[i] /= n; imag[i] /= n; }
}

/** Symmetric Hann window, with zero-valued endpoints. */
export function hann(size) {
  if (!Number.isInteger(size) || size < 1) throw new Error('Invalid window size.');
  return Float64Array.from({length: size}, (_, i) => size === 1 ? 1 : .5 - .5 * Math.cos(2 * Math.PI * i / (size - 1)));
}

/**
 * Center is a SAMPLE INDEX. Uses centered zero padding and a Hann window.
 * real/imag contain the full unnormalized FFT; magnitude contains bins 0..N/2.
 */
export function windowSpectrum(samples, center, size = FFT_SIZE) {
  checkFFTSize(size);
  if (!Number.isFinite(center)) throw new Error('Invalid spectrum center.');
  const real = new Float64Array(size), imag = new Float64Array(size);
  const window = hann(size), start = Math.round(center) - size / 2;
  for (let i = 0; i < size; i++) {
    const j = start + i;
    real[i] = (j >= 0 && j < samples.length ? samples[j] : 0) * window[i];
  }
  fft(real, imag);
  const magnitude = Float64Array.from({length: size / 2 + 1}, (_, i) => Math.hypot(real[i], imag[i]));
  return {real, imag, magnitude};
}

/**
 * Magnitudes of positive-frequency LF Fourier-series coefficients h=1..count.
 * They are abs(FFT)/cycleLength: no factor of two, window, or RMS normalization.
 * A real sinusoid's amplitude is twice its coefficient magnitude.
 */
export function lfHarmonicMagnitudes(count = 128) {
  if (!Number.isInteger(count) || count < 1 || count > 4096) throw new Error('Invalid harmonic count.');
  let size = 2048;
  while (size <= count * 2) size *= 2;
  const real = lfPulseCycle(size), imag = new Float64Array(size);
  fft(real, imag);
  return Float64Array.from({length: count}, (_, i) => Math.hypot(real[i + 1], imag[i + 1]) / size);
}

function abortIfNeeded(signal) {
  if (signal?.aborted) {
    const error = new Error('Speech synthesis cancelled.');
    error.name = 'AbortError';
    throw error;
  }
}

const yieldTask = () => new Promise(resolve => setTimeout(resolve, 0));

function checkTimes(times, count, label) {
  if (!times || times.length !== count || !count) throw new Error(`${label} times do not match the trajectory.`);
  for (let i = 0; i < times.length; i++) {
    if (!Number.isFinite(times[i]) || (i > 0 && times[i] <= times[i - 1])) throw new Error(`${label} times must increase.`);
  }
}

// Monotonic cursor interpolation; the caller visits time in ascending order.
function interpolator(times) {
  let i = 0;
  return time => {
    while (i < times.length - 2 && time >= times[i + 1]) i++;
    const next = Math.min(i + 1, times.length - 1);
    const blend = next === i ? 0 : Math.max(0, Math.min(1, (time - times[i]) / (times[next] - times[i])));
    return [i, next, blend];
  };
}

/**
 * Resynthesize a complete area trajectory using a continuous-phase LF source,
 * interpolated F0/voicing and deterministic broadband excitation when unvoiced.
 * Each centered frame is filtered by the complex tract response at its actual
 * FFT frequencies, then Hann overlap-added with squared-window normalization.
 * Original frame RMS supplies a nuisance gain, not original waveform content.
 * This approximation omits Reynolds-gated turbulence and moving-wall dynamics.
 * Returns a Float32Array of the original duration, with a normalizationGain
 * property recording the single global headroom gain (1 when unnecessary).
 * onProgress receives {stage:'synthesis', frame, total, progress}; cancellation
 * is checked before each frame and after cooperative yields every eight frames.
 */
export async function synthesizeTrajectory(sim, {
  samples, pitch, areas, frameTimes, length,
  sampleRate = SAMPLE_RATE, fftSize = FFT_SIZE, hopLength = HOP_LENGTH,
}, {onProgress, signal} = {}) {
  abortIfNeeded(signal);
  checkFFTSize(fftSize);
  if (!samples || !samples.length) {
    const empty = new Float32Array(0);
    empty.normalizationGain = 1;
    return empty;
  }
  if (!Number.isFinite(sampleRate) || sampleRate <= 0 || !Number.isInteger(hopLength) || hopLength < 1 || hopLength > fftSize / 2) throw new Error('Invalid synthesis framing.');
  if (fftSize > 8192) throw new Error('Synthesis FFT exceeds the simulator frequency capacity.');
  if (!samples.every(Number.isFinite)) throw new Error('Speech samples must be finite.');
  checkTimes(frameTimes, areas?.length, 'Area');
  for (const shape of areas) if (shape.length !== 32 || !shape.every(a => Number.isFinite(a) && a > 0)) throw new Error('Each frame needs 32 positive areas.');
  checkTimes(pitch?.times, pitch?.pitchHz?.length, 'Pitch');
  if (pitch.voiced && pitch.voiced.length !== pitch.times.length) throw new Error('Voicing trajectory length differs from pitch.');

  const n = samples.length, source = new Float64Array(n);
  const pulse = lfPulseCycle(2048);
  const mean = pulse.reduce((a, b) => a + b, 0) / pulse.length;
  let pulseEnergy = 0;
  for (let i = 0; i < pulse.length; i++) { pulse[i] -= mean; pulseEnergy += pulse[i] * pulse[i]; }
  const pulseScale = 1 / Math.sqrt(pulseEnergy / pulse.length);
  for (let i = 0; i < pulse.length; i++) pulse[i] *= pulseScale;
  // Carry valid F0 across unvoiced spans; voicing controls whether it is audible.
  const frequencies = Float64Array.from(pitch.pitchHz);
  let lastF0 = frequencies.find(f => Number.isFinite(f) && f >= 40 && f < sampleRate / 4) || 120;
  for (let i = 0; i < frequencies.length; i++) {
    if (Number.isFinite(frequencies[i]) && frequencies[i] >= 40 && frequencies[i] < sampleRate / 4) lastF0 = frequencies[i];
    else frequencies[i] = lastF0;
  }
  const voicing = Float64Array.from(pitch.pitchHz, (f, i) => pitch.voiced ? Math.max(0, Math.min(1, Number(pitch.voiced[i]) || 0)) : (Number.isFinite(f) && f > 0 ? 1 : 0));
  const pitchAt = interpolator(pitch.times);
  let phase = 0, noiseState = 0x61c88647;
  for (let j = 0; j < n; j++) {
    const [a, b, t] = pitchAt(j / sampleRate);
    const f0 = frequencies[a] + t * (frequencies[b] - frequencies[a]);
    const voiced = voicing[a] + t * (voicing[b] - voicing[a]);
    const position = phase * pulse.length, k = Math.floor(position), fraction = position - k;
    const glottal = pulse[k] + fraction * (pulse[(k + 1) % pulse.length] - pulse[k]);
    noiseState ^= noiseState << 13; noiseState ^= noiseState >>> 17; noiseState ^= noiseState << 5;
    const noise = ((noiseState >>> 0) / 0x100000000 * 2 - 1) * Math.sqrt(3);
    source[j] = Math.sqrt(voiced) * glottal + Math.sqrt(1 - voiced) * noise;
    phase = (phase + f0 / sampleRate) % 1;
    if (j > 0 && j % 65536 === 0) { await yieldTask(); abortIfNeeded(signal); }
  }

  const output = new Float64Array(n), weight = new Float64Array(n);
  const window = hann(fftSize), half = fftSize / 2;
  const bins = Float64Array.from({length: half + 1}, (_, i) => i * sampleRate / fftSize);
  const areaAt = interpolator(frameTimes), shape = new Float64Array(32);
  const frameCount = Math.ceil((n - 1) / hopLength) + 1;
  for (let frame = 0; frame < frameCount; frame++) {
    abortIfNeeded(signal);
    const center = frame * hopLength, start = center - half;
    const [a, b, blend] = areaAt(center / sampleRate);
    for (let k = 0; k < shape.length; k++) shape[k] = areas[a][k] + blend * (areas[b][k] - areas[a][k]);
    const real = new Float64Array(fftSize), imag = new Float64Array(fftSize);
    let targetEnergy = 0, validWindowEnergy = 0;
    for (let i = 0; i < fftSize; i++) {
      const j = start + i;
      if (j < 0 || j >= n) continue;
      const w = window[i];
      real[i] = source[j] * w;
      targetEnergy += samples[j] * samples[j] * w * w;
      validWindowEnergy += w * w;
    }
    const targetRMS = Math.sqrt(targetEnergy / Math.max(validWindowEnergy, 1e-20));
    if (targetRMS > 1e-6) {
      fft(real, imag);
      const response = sim.response(shape, length, bins);
      for (let k = 0; k <= half; k++) {
        const hr = response.real[k], hi = response.imag[k];
        if (!Number.isFinite(hr) || !Number.isFinite(hi)) throw new Error('Nonfinite acoustic transfer response.');
        const r = real[k] * hr - imag[k] * hi;
        const im = real[k] * hi + imag[k] * hr;
        real[k] = r; imag[k] = (k === 0 || k === half) ? 0 : im;
        if (k > 0 && k < half) { real[fftSize - k] = r; imag[fftSize - k] = -im; }
      }
      fft(real, imag, true);
      let generatedEnergy = 0;
      for (let i = 0; i < fftSize; i++) if (start + i >= 0 && start + i < n) generatedEnergy += real[i] * real[i];
      const gain = Math.sqrt(targetEnergy / Math.max(generatedEnergy, 1e-24));
      for (let i = 0; i < fftSize; i++) {
        const j = start + i;
        if (j >= 0 && j < n) output[j] += real[i] * window[i] * gain;
      }
    }
    // Silent frames still contribute normalizer weight, preventing adjacent
    // active frames from being spuriously amplified across a silent boundary.
    for (let i = 0; i < fftSize; i++) {
      const j = start + i;
      if (j >= 0 && j < n) weight[j] += window[i] * window[i];
    }
    if ((frame + 1) % 8 === 0 || frame === frameCount - 1) {
      onProgress?.({stage: 'synthesis', frame: frame + 1, total: frameCount, progress: (frame + 1) / frameCount});
      await yieldTask(); abortIfNeeded(signal);
    }
  }
  let peak = 0;
  for (let i = 0; i < n; i++) {
    output[i] = weight[i] > 1e-12 ? output[i] / weight[i] : 0;
    if (!Number.isFinite(output[i])) throw new Error('Nonfinite synthesized sample.');
    peak = Math.max(peak, Math.abs(output[i]));
  }
  // Preserve dynamic range. Apply only one global headroom gain when necessary;
  // never clip samples independently or peak-normalize individual frames.
  const safetyGain = peak > .98 ? .98 / peak : 1;
  const result = Float32Array.from(output, sample => sample * safetyGain);
  result.normalizationGain = safetyGain;
  return result;
}
