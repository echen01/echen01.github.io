import {createRffField} from '../speech/rff-field.js?v=vowel-rff-1';

export const VOWEL_RFF_CONFIG = Object.freeze({frequencies: 16, width: 64, spaceScale: 1, seed: 42});
export const MIN_LOG_AREA = Math.log(.12), MAX_LOG_AREA = Math.log(16);

// A stationary 1D RFF MLP. The initial tube is a frozen, continuous baseline;
// only the shared network weights are optimized, never individual area values.
// Zero output weights make the first forward exactly match the starting tube.
export function createVowelRffField(areas, options = {}) {
  if (areas?.length !== 32 || !Array.from(areas).every(a => Number.isFinite(a) && a >= .12 && a <= 16)) {
    throw new Error('The initial vowel tube needs 32 areas between 0.12 and 16 cm².');
  }
  const network = createRffField({...VOWEL_RFF_CONFIG, ...options, frameTimes: [0], duration: 1, numAreas: 32, timeScale: 0});
  const baseLogAreas = Float64Array.from(areas, Math.log), final = network.layout.at(-1);
  network.parameters.fill(0, final.weightOffset);
  const positions = Float64Array.from({length: 32}, (_, i) => i / 31);
  let derivative;
  const metadata = {...network.metadata, inputs: ['section / 31'], family: 'rff', fixedGlottisArea: null,
    areaTransform: {type: 'residual-log-area', minimum: .12, maximum: 16},
    initialization: 'Seeded Xavier hidden layers, zero output layer; fixed starting log-area profile'};
  function forwardAt(xs = positions) {
    if (!xs?.length || !Array.from(xs).every(x => Number.isFinite(x) && x >= 0 && x <= 1)) throw new Error('Positions must lie between 0 and 1.');
    const coordinates = new Float64Array(xs.length * 2), baseline = new Float64Array(xs.length);
    for (let i = 0; i < xs.length; i++) {
      coordinates[2 * i + 1] = xs[i];
      const p = xs[i] * 31, left = Math.min(30, Math.floor(p)), fraction = p - left;
      baseline[i] = baseLogAreas[left] * (1 - fraction) + baseLogAreas[left + 1] * fraction;
    }
    const raw = network.forwardRawAt(coordinates);
    derivative = new Float64Array(xs.length);
    return Float64Array.from(raw, (value, i) => {
      const log = baseline[i] + value;
      derivative[i] = log > MIN_LOG_AREA && log < MAX_LOG_AREA ? 1 : 0;
      return Math.max(MIN_LOG_AREA, Math.min(MAX_LOG_AREA, log));
    });
  }
  const {config, parameters, encoding, B, layout, parameterCount, refreshEncoding} = network;
  return {config, parameters, encoding, B, layout, parameterCount, refreshEncoding, metadata, baseLogAreas,
    forward: () => forwardAt(), forwardAt,
    backward(cotangent) {
      if (!derivative || cotangent?.length !== derivative.length) throw new Error('Run the vowel field forward before a matching backward pass.');
      return network.backwardRaw(Float64Array.from(cotangent, (g, i) => g * derivative[i]));
    },
  };
}

export function vowelLearningRate(step, iterations = 240, peak = .005) {
  // Cosine decay reduces late oscillation while retaining a small final step.
  return peak * (.1 + .9 * .5 * (1 + Math.cos(Math.PI * step / Math.max(1, iterations - 1))));
}

export function createVowelRffFitter(sim, {areas, targetAreas, length = 17.5, iterations = 240, learningRate = .005, fieldConfig = {}}) {
  return createVowelFieldFitter(sim, createVowelRffField(areas, fieldConfig), {targetAreas, length, iterations, learningRate});
}

export function createVowelFieldFitter(sim, field, {targetAreas, length = 17.5, iterations = 240, learningRate = .005}) {
  const family = field.metadata.family;
  if (!Number.isInteger(iterations) || iterations < 1 || iterations > 1000) throw new Error('Choose between 1 and 1000 updates.');
  if (!Number.isFinite(learningRate) || learningRate <= 0 || learningRate > .1) throw new Error('Invalid field learning rate.');
  const targetDb = sim.response(targetAreas, length).magnitudeDb;
  const first = new Float64Array(field.parameterCount), second = new Float64Array(field.parameterCount);
  let iteration = 0, current, evaluation, parameterGradient;
  const history = [];
  function evaluate() {
    current = Array.from(field.forward(), Math.exp);
    evaluation = sim.lossAndGradient(current, targetDb, length);
    parameterGradient = field.backward(evaluation.gradient);
    if (!parameterGradient.every(Number.isFinite)) throw new Error('The field gradient is not finite. Reset to try again.');
    history.push(evaluation.loss);
  }
  evaluate();
  function snapshot({diagnostic = false} = {}) {
    return {parameterization: family, field: field.metadata, iteration, areas: [...current], loss: evaluation.loss,
      gradient: [...evaluation.gradient], gradientNorm: Math.hypot(...evaluation.gradient),
      ...(diagnostic ? {parameterGradient: parameterGradient.slice(), parameters: field.parameters.slice()} : {}),
      magnitudeDb: sim.response(current, length).magnitudeDb, targetDb: [...targetDb], history: [...history]};
  }
  return {field, snapshot,
    step() {
      if (iteration >= iterations) throw new Error('The requested field fit is complete.');
      const rate = vowelLearningRate(iteration, iterations, learningRate), count = iteration + 1;
      let squared = 0;
      for (const g of parameterGradient) squared += g * g;
      const clip = Math.min(1, 5 / Math.max(1e-12, Math.sqrt(squared)));
      for (let i = 0; i < field.parameterCount; i++) {
        const g = parameterGradient[i] * clip;
        first[i] = .9 * first[i] + .1 * g; second[i] = .999 * second[i] + .001 * g * g;
        field.parameters[i] -= rate * (first[i] / (1 - .9 ** count)) / (Math.sqrt(second[i] / (1 - .999 ** count)) + 1e-8);
      }
      iteration++; evaluate();
      return {iteration};
    },
  };
}
