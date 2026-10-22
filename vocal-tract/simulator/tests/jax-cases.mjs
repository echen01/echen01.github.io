// Deterministic cross-implementation cases for original JAX step_frame versus
// the shipped WASM stationary, noise-free transfer model. All arrays have 32
// entries, including the final lip-boundary area. Areas are cm², length is cm.
// `bins` indexes rfft(impulse response); exclude DC and Nyquist because irfft
// projects those two complex coefficients onto the real axis.
import {PRESETS, initialAreas} from '../index.js';

const N = 32;
const clamp = x => Math.max(.12, Math.min(16, x));
const constant = value => Array(N).fill(value);

function smoothGeometry(seed) {
  let state = seed >>> 0;
  const random = () => {
    state = (Math.imul(1664525, state) + 1013904223) >>> 0;
    return state / 4294967296;
  };
  const controls = Array.from({length: 7}, () => Math.log(.45) + random() * Math.log(10 / .45));
  return Array.from({length: N}, (_, i) => {
    const position = i / (N - 1) * (controls.length - 1);
    const left = Math.min(controls.length - 2, Math.floor(position));
    const u = position - left;
    const weight = u * u * (3 - 2 * u);
    return Math.exp(controls[left] * (1 - weight) + controls[left + 1] * weight);
  });
}

function localized(base, center, width, minimum) {
  return Array.from({length: N}, (_, i) => {
    const attenuation = Math.exp(-.5 * ((i - center) / width) ** 2);
    return Math.exp(Math.log(base) + attenuation * Math.log(minimum / base));
  });
}

const cases = [];
function add(name, areas, targetAreas, length, fftSize) {
  // Each case includes low-frequency bins, the resonance range, and frequencies
  // above the demo's usual 5 kHz limit, up to the last interior Nyquist bin.
  cases.push({name, areas: [...areas], targetAreas: [...targetAreas], length,
    sampleRate: 16000, fftSize,
    bins: Array.from({length: fftSize / 2 - 1}, (_, i) => i + 1)});
}

for (const [index, preset] of PRESETS.entries()) {
  add(`neutral-to-${preset.id}`, initialAreas(), preset.areas, preset.length, [512, 1024, 2048][index]);
  add(`${preset.id}-to-${PRESETS[(index + 1) % 3].id}`, preset.areas,
    PRESETS[(index + 1) % 3].areas, preset.length, [1024, 2048, 512][index]);
}

for (let i = 0; i < 8; i++) {
  add(`smooth-seed-${i + 1}`, smoothGeometry(0x6a09e667 + 977 * i),
    smoothGeometry(0xbb67ae85 + 1499 * i),
    [12, 14, 16, 17.5, 18, 19.5, 21, 22][i], [512, 1024, 2048][i % 3]);
}

for (const [index, area] of [.12, .1205, 1, 4, 15.999, 16].entries()) {
  add(`uniform-area-${area}`, constant(area), smoothGeometry(0x3c6ef372 + index * 811),
    [12, 22, 17.5, 20, 12, 22][index], [512, 1024, 2048][index % 3]);
}

for (const [name, position, area, fftSize] of [
  ['glottal-constriction', 0, .12, 512],
  ['pharyngeal-constriction', 6, .12, 1024],
  ['central-constriction', 16, .15, 2048],
  ['last-section-constriction', 30, .12, 512],
  ['narrow-lip-boundary', 31, .12, 1024],
]) {
  const areas = smoothGeometry(0xa54ff53a);
  areas[position] = area;
  add(name, areas, PRESETS[0].areas, 17.5, fftSize);
}
const broad = localized(5, 17, 2.4, .16).map((area, i) => clamp(area * (1 + .08 * Math.sin(i * .4))));
broad[N - 1] = 16;
add('broad-constriction-wide-lip', broad, PRESETS[2].areas, 20, 2048);

export default cases;
