import {synthesizeLfVowel, vowelFrequencies} from './vowel-source.js';

// Browser interface to the stationary diffTLM Rust/WASM model.
export const NUM_AREAS = 32;
export const FREQUENCIES = Array.from({length: 160}, (_, i) => 80 + i * (5000 - 80) / 159);
const resample = source => Array.from({length: NUM_AREAS}, (_, i) => {
  const p = i * (source.length - 1) / (NUM_AREAS - 1);
  const j = Math.min(Math.floor(p), source.length - 2);
  return source[j] * (1 - (p - j)) + source[j + 1] * (p - j);
});

// Fant area data distributed with diffTLM (data/fant.py), resampled to 32
// values. The last value is the lip boundary, matching the current JAX model.
export const PRESETS = [
  {id: 'a', name: 'Ah', label: 'ɑ / “ah”', length: 17.5, areas: resample([2.6,1.6,1.3,1,4,2.6,1.6,1,.65,.65,.65,1,1.3,1.6,2,2.6,2.6,1.6,3.2,4,5,6.5,8,8,8,8,8,8,8,8,6.5,5,5,5,5])},
  {id: 'e', name: 'Ay', label: 'e / “ay”', length: 18, areas: resample([2.6,2,1.6,1.3,6.5,6.5,6.5,6.5,8,8,10.5,10.5,10.5,10.5,10.5,8,6.5,8,6.5,8,6.5,5,5,4,4,4,3.2,2.6,2.6,2,2.6,4,5,5,8,8])},
  {id: 'u', name: 'Oo', label: 'u / “oo”', length: 20, areas: resample([2.6,2.6,2,2,10.5,10.5,10.5,8,8,5,3.2,1.6,1.3,1,1,1,1.6,2,1.3,1.6,2,2,2,2.6,3.2,5,6.5,8,10.5,13,13,13,13,10.5,5,2,.32,.32,.65,.65])},
];

export function initialAreas() {
  return Array.from({length: NUM_AREAS}, (_, i) => 2.8 * Math.exp(.12 * Math.cos(i * Math.PI / 31) + .05 * Math.sin(i * .47)));
}

function checkAreas(areas) {
  if (!areas || areas.length !== NUM_AREAS || Array.from(areas).some(v => !Number.isFinite(v) || v <= 0 || v > 100)) {
    throw new Error('The model requires 32 positive, finite cross-sectional areas.');
  }
}
function checkLength(length) {
  if (!Number.isFinite(length) || length < 5 || length > 40) throw new Error('Tract length must be between 5 and 40 cm.');
}

export async function createSimulator(wasmBytes) {
  let instance;
  if (wasmBytes) {
    ({instance} = await WebAssembly.instantiate(wasmBytes, {}));
  } else {
    const response = await fetch(new URL('./difftlm.wasm', import.meta.url));
    if (!response.ok) throw new Error('The acoustic simulator could not be loaded. Please reload the page.');
    ({instance} = await WebAssembly.instantiate(await response.arrayBuffer(), {}));
  }
  const w = instance.exports;
  const view = (name, length) => new Float64Array(w.memory.buffer, w[name](), length);
  function setup(areas, length, frequencies) {
    checkAreas(areas); checkLength(length);
    if (!frequencies?.length || frequencies.length > 4097 || Array.from(frequencies).some(f => !Number.isFinite(f) || f < 0)) {
      throw new Error('Supply between 1 and 4097 nonnegative frequencies.');
    }
    view('areas_ptr', NUM_AREAS).set(areas);
    view('frequencies_ptr', frequencies.length).set(frequencies);
  }
  const api = {
    response(areas, length = 17.5, frequencies = FREQUENCIES) {
      setup(areas, length, frequencies);
      const result = w.evaluate_response(length, frequencies.length);
      if (result !== 0) throw new Error('The acoustic response could not be evaluated.');
      return {frequencies: Array.from(frequencies), real: Array.from(view('real_ptr', frequencies.length)), imag: Array.from(view('imag_ptr', frequencies.length)), magnitudeDb: Array.from(view('db_ptr', frequencies.length))};
    },
    lossAndGradient(areas, targetDb, length = 17.5, frequencies = FREQUENCIES) {
      setup(areas, length, frequencies);
      if (targetDb?.length !== frequencies.length || Array.from(targetDb).some(x => !Number.isFinite(x))) throw new Error('The target spectrum must have one finite value per frequency.');
      view('target_ptr', frequencies.length).set(targetDb);
      const loss = w.evaluate_loss(length, frequencies.length);
      const gradient = Array.from(view('gradient_ptr', NUM_AREAS));
      if (!Number.isFinite(loss) || gradient.some(g => !Number.isFinite(g))) throw new Error('The optimizer encountered a nonfinite acoustic response. Reset to try again.');
      return {loss, gradient};
    },
    gainInvariantLossAndGradient(areas, targetDb, length = 17.5, frequencies = FREQUENCIES) {
      setup(areas, length, frequencies);
      if (targetDb?.length !== frequencies.length || Array.from(targetDb).some(x => !Number.isFinite(x))) throw new Error('The target spectrum must have one finite value per frequency.');
      view('target_ptr', frequencies.length).set(targetDb);
      const loss = w.evaluate_gain_invariant_loss(length, frequencies.length);
      const gradient = Array.from(view('gradient_ptr', NUM_AREAS));
      const gainDb = w.gain_db();
      if (!Number.isFinite(loss) || !Number.isFinite(gainDb) || gradient.some(g => !Number.isFinite(g))) throw new Error('The optimizer encountered a nonfinite acoustic response. Reset to try again.');
      return {loss, gradient, gainDb};
    },
    synthesize(areas, length = 17.5, f0 = 120, duration = .8) {
      checkAreas(areas); checkLength(length);
      if (!Number.isFinite(f0) || f0 < 65 || f0 > 500) throw new Error('Pitch must be between 65 and 500 Hz.');
      if (!Number.isFinite(duration) || duration <= 0 || duration > 5) throw new Error('Sound duration must be between 0 and 5 seconds.');
      const frequencies = vowelFrequencies(f0);
      const response = api.response(areas, length, frequencies);
      return synthesizeLfVowel(response, f0, duration);
    },
  };
  return api;
}

// Shared by the Worker and numerical regression checks. Each returned loss is
// evaluated at the returned geometry; plotting cannot show a stale/fake value.
export function createFitter(sim, {areas, targetAreas, length = 17.5, learningRate = .035}) {
  checkAreas(areas); checkAreas(targetAreas); checkLength(length);
  if (!Number.isFinite(learningRate) || learningRate <= 0 || learningRate > 1) throw new Error('Invalid learning rate.');
  const targetDb = sim.response(targetAreas, length).magnitudeDb;
  const logs = Array.from(areas, Math.log), m = logs.map(() => 0), v = logs.map(() => 0);
  let current = Array.from(areas), iteration = 0;
  let evaluation = sim.lossAndGradient(current, targetDb, length);
  const history = [evaluation.loss];
  function snapshot() {
    return {iteration, areas: [...current], loss: evaluation.loss, gradient: [...evaluation.gradient], gradientNorm: Math.hypot(...evaluation.gradient), magnitudeDb: sim.response(current, length).magnitudeDb, targetDb: [...targetDb], history: [...history]};
  }
  return {
    snapshot,
    step() {
      iteration++;
      const clip = Math.min(1, 5 / Math.max(1e-12, Math.hypot(...evaluation.gradient)));
      for (let i = 0; i < logs.length; i++) {
        const gradient = evaluation.gradient[i] * clip;
        m[i] = .9 * m[i] + .1 * gradient;
        v[i] = .999 * v[i] + .001 * gradient * gradient;
        const mh = m[i] / (1 - Math.pow(.9, iteration)), vh = v[i] / (1 - Math.pow(.999, iteration));
        logs[i] = Math.max(Math.log(.12), Math.min(Math.log(16), logs[i] - learningRate * mh / (Math.sqrt(vh) + 1e-8)));
      }
      current = logs.map(Math.exp);
      evaluation = sim.lossAndGradient(current, targetDb, length);
      history.push(evaluation.loss);
      return snapshot();
    },
  };
}
