/** Exact frame layout/windowing/scatter-add from upstream tract.py. */
import {createFaithfulAcoustic, acousticFftSize} from './faithful-acoustic.js';
import {rolloutNoise} from './faithful-noise.js';

export function rolloutGeometry(sampleCount, sampleRate = 32000, fftSize = acousticFftSize(sampleRate)) {
  const frameLength = Math.floor(sampleRate * .025), hopLength = Math.floor(frameLength / 4);
  const frames = Math.floor((sampleCount - frameLength) / hopLength) + 1;
  if (frames < 1 || fftSize < frameLength) throw new Error('Recording or FFT is too short for the upstream 25 ms analysis window.');
  const analysisWindow = new Float64Array(frameLength), synthesisWindow = new Float64Array(frameLength), alpha = .25;
  for (let i = 0; i < frameLength; i++) {
    const x = i / (frameLength - 1);
    analysisWindow[i] = x < alpha / 2 ? .5 * (1 + Math.cos(Math.PI * (2 * x / alpha - 1)))
      : x >= 1 - alpha / 2 ? .5 * (1 + Math.cos(Math.PI * (2 * x / alpha - 2 / alpha + 1))) : 1;
    synthesisWindow[i] = .5 * (1 - Math.cos(2 * Math.PI * x));
  }
  return {sampleCount, sampleRate, fftSize, frameLength, hopLength, frames, analysisWindow, synthesisWindow};
}

export function createFaithfulRollout(uin, {sampleRate = 32000, fftSize = acousticFftSize(sampleRate), length = 17.5, noiseDtype = 32} = {}) {
  const geometry = rolloutGeometry(uin.length, sampleRate, fftSize);
  const {frames, frameLength, hopLength, analysisWindow, synthesisWindow} = geometry;
  const acoustic = createFaithfulAcoustic({sampleRate, fftSize, length});
  const inputs = Array.from({length: frames}, (_, frame) => {
    const input = new Float64Array(fftSize);
    for (let i = 0; i < frameLength; i++) input[i] = uin[frame * hopLength + i] * analysisWindow[i];
    return input;
  });
  function frameNoise(noise, frame) {
    const result = new Float64Array(31 * fftSize);
    for (let section = 0; section < 31; section++) for (let i = 0; i < frameLength; i++) {
      result[section * fftSize + i] = noise[section][frame * hopLength + i] * analysisWindow[i];
    }
    return result;
  }
  return {
    ...geometry,
    // Recompute frame intermediates in backward to avoid retaining hundreds of
    // large complex transfer-function tapes. This changes memory, not math.
    forward(areas, {iteration = 0, noise = rolloutNoise(uin.length, iteration, {dtype: noiseDtype}), onFrame} = {}) {
      if (areas.length !== frames * 32) throw new Error('Area trajectory must have n_windows × 32 entries.');
      const samples = new Float64Array(uin.length);
      for (let frame = 0; frame < frames; frame++) {
        const local = acoustic.forward(inputs[frame], areas.subarray(frame * 32, (frame + 1) * 32), frameNoise(noise, frame));
        for (let i = 0; i < frameLength; i++) samples[frame * hopLength + i] += local.samples[i] * synthesisWindow[i];
        onFrame?.(frame + 1, frames);
      }
      return {samples, backward(cotangent) {
        if (cotangent.length !== uin.length) throw new Error('Waveform cotangent length mismatch.');
        const gradient = new Float64Array(areas.length), localCotangent = new Float64Array(fftSize);
        for (let frame = 0; frame < frames; frame++) {
          for (let i = 0; i < frameLength; i++) localCotangent[i] = cotangent[frame * hopLength + i] * synthesisWindow[i];
          const local = acoustic.forward(inputs[frame], areas.subarray(frame * 32, (frame + 1) * 32), frameNoise(noise, frame));
          gradient.set(local.backward(localCotangent), frame * 32);
        }
        return gradient;
      }};
    },
  };
}
