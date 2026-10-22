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
  {id: 'i', name: 'Ee', label: 'i / “ee”', length: 17.5, areas: resample([3.2,2.6,2,2,8,8,10.5,10.5,10.5,10.5,10.5,10.5,10.5,10.5,10.5,8,8,6.5,4,2.6,1.3,.65,.65,.65,.65,.65,.65,.65,1,1.3,1.6,3.2,4,4])},
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
    synthesize(areas, length = 17.5, f0 = 120, duration = .8) {
      checkAreas(areas); checkLength(length);
      if (!Number.isFinite(f0) || f0 < 65 || f0 > 500) throw new Error('Pitch must be between 65 and 500 Hz.');
      if (!Number.isFinite(duration) || duration <= 0 || duration > 5) throw new Error('Sound duration must be between 0 and 5 seconds.');
      const sampleRate = 32000;
      const count = Math.floor(sampleRate * duration);
      const harmonics = Math.floor((sampleRate / 2 - 1) / f0);
      const frequencies = Array.from({length: harmonics}, (_, i) => (i + 1) * f0);
      const response = api.response(areas, length, frequencies);
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
    },
  };
  return api;
}

// Liljencrants–Fant flow pulse, matching difftlm/glottis.py's OQ/SQ/TL.
// The stationary demonstration uses a periodic source with no breath noise.
function lfPulseCycle(count) {
  const te = .5, tp = .375, ta = .02, w = Math.PI / tp;
  let epsilon = 1 / ta;
  for (let i = 0; i < 40; i++) {
    const h = 1 - Math.exp(-epsilon * (1 - te)) - epsilon * ta;
    if (Math.abs(h) < 1e-5) break;
    epsilon -= h / ((1 - te) * Math.exp(-epsilon * (1 - te)) - ta);
  }
  const sw = Math.sin(w * te), cw = Math.cos(w * te);
  const x = sw * Math.exp(epsilon * te) * (-Math.exp(-epsilon) / epsilon - Math.exp(-epsilon)) / (epsilon * ta);
  const y = sw * Math.exp(epsilon * te) * (-Math.exp(-epsilon * te) / epsilon - te * Math.exp(-epsilon)) / (epsilon * ta);
  const objective = a => w / (w * w + a * a) + Math.exp(a * te) * ((a * sw - w * cw) / (w * w + a * a) + x - y);
  let lo = 0, hi = 0, alpha = 0;
  for (let i = 0; i < 20; i++) { hi++; if (objective(hi) < 0) break; }
  if (objective(hi) < 0) for (let i = 0; i < 20; i++) {
    const flo = objective(lo), fhi = objective(hi);
    alpha = lo - flo * (hi - lo) / (fhi - flo);
    const f = objective(alpha);
    if (Math.abs(f) <= 1e-5) break;
    if (f < 0) hi = alpha; else lo = alpha;
  }
  const b = 1 / ((Math.exp(alpha * tp) * (alpha * Math.sin(w * tp) - w * Math.cos(w * tp)) + w) / (w * w + alpha * alpha));
  const open = t => b * (Math.exp(alpha * t) * (alpha * Math.sin(w * t) - w * Math.cos(w * t)) + w) / (w * w + alpha * alpha);
  const factor = b * Math.exp(alpha * te) * sw * Math.exp(epsilon * te) / (epsilon * ta);
  const ret = t => factor * (-Math.exp(-epsilon * t) / epsilon - t * Math.exp(-epsilon));
  return Array.from({length: count}, (_, i) => {const t = i / count; return t < te ? open(t) : open(te) + ret(t) - ret(te);});
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
