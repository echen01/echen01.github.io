import {lfPulseCycle} from '../lf-source.js';

// Both backends filter the same noise-free LF flow pulse at its harmonics.
export function vowelFrequencies(f0 = 120, sampleRate = 32000) {
  if (!Number.isFinite(f0) || f0 < 65 || f0 > 500) throw new Error('Pitch must be between 65 and 500 Hz.');
  const harmonics = Math.floor((sampleRate / 2 - 1) / f0);
  return Array.from({length: harmonics}, (_, i) => (i + 1) * f0);
}

export function synthesizeLfVowel(response, f0 = 120, duration = .8) {
  if (!Number.isFinite(duration) || duration <= 0 || duration > 5) throw new Error('Sound duration must be between 0 and 5 seconds.');
  const sampleRate = 32000, count = Math.floor(sampleRate * duration);
  const harmonics = vowelFrequencies(f0, sampleRate).length;
  if (response.real.length !== harmonics || response.imag.length !== harmonics) throw new Error('Expected one acoustic response per LF harmonic.');
  const pulse = lfPulseCycle(2048);
  const samples = new Float32Array(count);
  for (let h = 1; h <= harmonics; h++) {
    let re = 0, im = 0;
    for (let i = 0; i < pulse.length; i++) {
      const phase = 2 * Math.PI * h * i / pulse.length;
      re += pulse[i] * Math.cos(phase) / pulse.length;
      im -= pulse[i] * Math.sin(phase) / pulse.length;
    }
    const hr = response.real[h - 1], hi = response.imag[h - 1];
    const pr = re * hr - im * hi, pi = re * hi + im * hr;
    const angle = 2 * Math.PI * f0 * h / sampleRate;
    const ca = Math.cos(angle), sa = Math.sin(angle);
    let cos = 1, sin = 0;
    for (let i = 0; i < count; i++) {
      samples[i] += 2 * (pr * cos - pi * sin);
      const nextCos = cos * ca - sin * sa;
      sin = sin * ca + cos * sa;
      cos = nextCos;
    }
  }
  let peak = 0;
  for (const sample of samples) peak = Math.max(peak, Math.abs(sample));
  for (let i = 0; i < count; i++) {
    const fade = Math.min(1, i / (sampleRate * .025), (count - i - 1) / (sampleRate * .04));
    samples[i] *= .3 * Math.max(0, fade) / Math.max(peak, 1e-12);
  }
  return {samples, sampleRate};
}
