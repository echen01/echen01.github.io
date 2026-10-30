import {createRffField} from './rff-field.js?v=fixed-grid';

export const UPSTREAM_COMMIT = 'b05116e06afbd7a569f3b9ac1d03a52c9d18efed';
export const FAITHFUL_HIDDEN_DIMS = Object.freeze([64, 128, 256, 512]);
export function validateFaithfulHiddenDim(hiddenDim) {
  if (!FAITHFUL_HIDDEN_DIMS.includes(hiddenDim)) throw new Error('RFF hidden width must be 64, 128, 256, or 512.');
  return hiddenDim;
}

function validateManifest(manifest, hiddenDim = manifest?.hiddenDim) {
  validateFaithfulHiddenDim(hiddenDim);
  if (manifest?.hiddenDim !== hiddenDim || manifest.encodingDim !== 256 || manifest.hiddenLayers !== 3 || manifest.activation !== 'tanh') {
    throw new Error(`The ${hiddenDim}-wide RFF bundle must use 512 frozen Fourier features and three tanh hidden layers.`);
  }
  const sizes = [512, hiddenDim, hiddenDim, hiddenDim, 1];
  let offset = 0;
  if (manifest.layout?.length !== 4) throw new Error('The RFF initialization must contain four affine layers.');
  for (let i = 0; i < 4; i++) {
    const inputSize = sizes[i], outputSize = sizes[i + 1], weightCount = inputSize * outputSize;
    const expected = {inputSize, outputSize, weightOffset: offset, weightCount, biasOffset: offset + weightCount, biasCount: outputSize};
    if (Object.entries(expected).some(([key, value]) => manifest.layout[i]?.[key] !== value)) throw new Error(`Invalid RFF initialization layout at layer ${i}.`);
    offset += weightCount + outputSize;
  }
  if (manifest.parameterCount !== offset) throw new Error('RFF initialization parameter count does not match its architecture.');
  return offset;
}

function float32Bundle(bytes, count, label) {
  const raw = bytes instanceof ArrayBuffer ? bytes : ArrayBuffer.isView(bytes) ? bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) : undefined;
  if (!raw || raw.byteLength !== count * 4) throw new Error(`Invalid ${label} bundle: expected ${count * 4} bytes.`);
  const values = new Float32Array(raw);
  if (!values.every(Number.isFinite)) throw new Error(`Invalid ${label} bundle: nonfinite values.`);
  return values;
}

// An injected read(name) still receives paths relative to the selected bundle,
// preserving the original callback contract for Node tests and local fixtures.
export async function loadFaithfulInitialization(read, {hiddenDim = 512} = {}) {
  validateFaithfulHiddenDim(hiddenDim);
  const directory = hiddenDim === 512 ? 'reference' : 'reference-small';
  const manifestName = hiddenDim === 512 ? 'rff-init.json' : `rff-init-${hiddenDim}.json`;
  const readFile = read ?? (async name => {
    const response = await fetch(new URL(`./upstream/${directory}/${name}`, import.meta.url));
    if (!response.ok) throw new Error(`Could not load ${hiddenDim}-wide RFF initialization: ${name} (HTTP ${response.status}).`);
    return response.arrayBuffer();
  });
  if (typeof readFile !== 'function') throw new Error('Initialization reader must be a function.');
  let manifest;
  try { manifest = JSON.parse(new TextDecoder().decode(await readFile(manifestName))); }
  catch (error) { throw new Error(`Could not read ${manifestName}: ${error.message}`); }
  const parameterCount = validateManifest(manifest, hiddenDim);
  if (typeof manifest.parameters?.file !== 'string' || typeof manifest.encoding?.file !== 'string') throw new Error('RFF initialization is missing parameter or Fourier-encoding files.');
  const [parameters, encoding] = await Promise.all([readFile(manifest.parameters.file), readFile(manifest.encoding.file)]);
  return {manifest, parameters: float32Bundle(parameters, parameterCount, 'RFF parameters'), encoding: float32Bundle(encoding, 512, 'frozen Fourier encoding')};
}

/** Original NNX weights and frozen B; original endpoint-aligned coordinates. */
export function createFaithfulField(frames, duration, initialization, {lazyFeatures = false} = {}) {
  const {manifest, parameters, encoding} = initialization;
  const expectedParameters = validateManifest(manifest);
  if (parameters?.length !== expectedParameters || encoding?.length !== 512 || !parameters.every(Number.isFinite) || !encoding.every(Number.isFinite)) throw new Error('Invalid RFF initialization tensor shapes or values.');
  const times = Array.from({length: frames}, (_, i) => frames > 1 ? duration * (i === frames - 1 ? 1 : Math.fround(i * Math.fround(1 / (frames - 1)))) : 0);
  const field = createRffField({frameTimes: times, duration, numAreas: 32,
    frequencies: manifest.encodingDim, width: manifest.hiddenDim, timeScale: 10, spaceScale: 2, seed: 0, coordinateDtype: 32, lazyFeatures});
  if (field.parameters.length !== parameters.length || field.encoding.length !== encoding.length) throw new Error('Upstream initialization does not match the RFF architecture.');
  field.parameters.set(parameters); field.encoding.set(encoding); field.refreshEncoding();
  field.metadata.initialization = manifest.initialization;
  field.metadata.inputs = ['frame / (n_windows - 1)', 'section / 31'];
  field.metadata.timeNormalization = 'Endpoint-aligned linspace(0,1,n_windows), as in difftlm.fields._coordinates.';
  field.metadata.upstreamCommit = UPSTREAM_COMMIT;
  field.metadata.faithful = true;
  field.metadata.family = 'rff';
  return field;
}

/** optax.warmup_cosine_decay_schedule, evaluated at the zero-based update. */
export function faithfulLearningRate(step, iterations = 500, peak = 1e-3) {
  const warmup = Math.min(50, iterations - 1), initial = 1e-6;
  if (step < warmup) return initial + (peak - initial) * step / warmup;
  const progress = Math.min(1, Math.max(0, (step - warmup) / (iterations - warmup)));
  return peak * .5 * (1 + Math.cos(Math.PI * progress));
}
export function createFaithfulAdam(parameters, {iterations = 500, peak = 1e-3} = {}) {
  const first = new Float64Array(parameters.length), second = new Float64Array(parameters.length);
  let step = 0;
  return {update(gradient) {
    const rate = faithfulLearningRate(step, iterations, peak), count = step + 1;
    const b1Correction = 1 - .9 ** count, b2Correction = 1 - .999 ** count;
    for (let i = 0; i < parameters.length; i++) {
      const g = gradient[i];
      first[i] = .9 * first[i] + .1 * g;
      second[i] = .999 * second[i] + .001 * g * g;
      parameters[i] -= rate * (first[i] / b1Correction) / (Math.sqrt(second[i] / b2Correction) + 1e-8);
    }
    step++; return rate;
  }, get step() { return step; }, first, second};
}
