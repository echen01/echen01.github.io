import {UPSTREAM_COMMIT} from './faithful-field.js';

const WIDTH = 256, SECTIONS = 32;
const AREA_BIAS = .1, AREA_FLOOR = 1e-4, AREA_CEILING = 16;
const softplus = x => Math.max(x, 0) + Math.log1p(Math.exp(-Math.abs(x)));
const sigmoid = x => x >= 0 ? 1 / (1 + Math.exp(-x)) : Math.exp(x) / (1 + Math.exp(x));
const FIXED_AREA = softplus(AREA_BIAS) + AREA_FLOOR;

function validateManifest(manifest, hiddenDim = WIDTH) {
  if (hiddenDim !== WIDTH) throw new Error('The faithful MFN currently uses the original 256-wide FourierNet.');
  if (manifest?.family !== 'mfn' || manifest.hiddenDim !== hiddenDim || manifest.hiddenLayers !== 3 || manifest.filterCount !== 4 || manifest.inputScale !== 256 || manifest.outputAct !== false) throw new Error('Invalid original FourierNet initialization architecture.');
  let offset = 0;
  const validate = (layer, inputSize, outputSize) => {
    const weightCount = inputSize * outputSize;
    const expected = {inputSize, outputSize, weightOffset: offset, weightCount, biasOffset: offset + weightCount, biasCount: outputSize};
    if (Object.entries(expected).some(([key, value]) => layer?.[key] !== value)) throw new Error('Invalid MFN parameter layout.');
    offset += weightCount + outputSize;
  };
  if (manifest.layout?.length !== 4 || manifest.filterLayout?.length !== 4) throw new Error('MFN initialization must contain four affine layers and four sine filters.');
  for (let i = 0; i < 4; i++) validate(manifest.layout[i], hiddenDim, i === 3 ? 1 : hiddenDim);
  for (const layer of manifest.filterLayout) validate(layer, 2, hiddenDim);
  if (manifest.parameterCount !== offset) throw new Error('Invalid MFN parameter count.');
  return offset;
}

export async function loadFaithfulMfnInitialization(read, {hiddenDim = WIDTH} = {}) {
  if (hiddenDim !== WIDTH) throw new Error('The faithful MFN currently uses the original 256-wide FourierNet.');
  const readFile = read ?? (async name => {
    const response = await fetch(new URL(`./upstream/reference-mfn/${name}`, import.meta.url));
    if (!response.ok) throw new Error(`Could not load MFN initialization: ${name} (HTTP ${response.status}).`);
    return response.arrayBuffer();
  });
  if (typeof readFile !== 'function') throw new Error('Initialization reader must be a function.');
  const manifest = JSON.parse(new TextDecoder().decode(await readFile(`mfn-init-${hiddenDim}.json`)));
  const parameterCount = validateManifest(manifest, hiddenDim);
  if (typeof manifest.parameters?.file !== 'string') throw new Error('MFN initialization is missing its parameter file.');
  const bytes = await readFile(manifest.parameters.file);
  const buffer = bytes instanceof ArrayBuffer ? bytes : ArrayBuffer.isView(bytes) ? bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) : undefined;
  if (!buffer || buffer.byteLength !== parameterCount * 4) throw new Error('Invalid MFN parameter bundle size.');
  const parameters = new Float32Array(buffer);
  if (!parameters.every(Number.isFinite)) throw new Error('Nonfinite MFN initialization.');
  return {manifest, parameters};
}

function workspace(count, width) {
  return {
    activations: Array.from({length: 4}, () => new Float64Array(count * width)),
    affine: Array.from({length: 3}, () => new Float64Array(count * width)),
    sine: Array.from({length: 4}, () => new Float64Array(count * width)),
    cosine: Array.from({length: 4}, () => new Float64Array(count * width)),
    adjoints: Array.from({length: 4}, () => new Float64Array(count * width)),
    logAreaDerivative: new Float64Array(count),
    rawField: new Float64Array(count),
  };
}

/** Original FourierNet: learned sin(F_k c + b_k) filters multiply three
 * hidden affine transforms. All eight affine modules are trainable; there is
 * no frozen RFF encoding and no additional 2*pi in the filter phases.
 * Backward consumes cotangents of log(area), matching the acoustic solver.
 */
export function createFaithfulMfnField(frames, duration, initialization, {inputScale = 256, areaScale = .35} = {}) {
  if (!Number.isInteger(frames) || frames < 1 || !Number.isFinite(duration) || duration <= 0) throw new Error('MFN requires a positive frame count and duration.');
  if (!Number.isFinite(inputScale) || inputScale <= 0) throw new Error('MFN inputScale must be positive and finite.');
  if (!Number.isFinite(areaScale) || areaScale <= 0) throw new Error('MFN areaScale must be positive and finite.');
  const {manifest, parameters: initial} = initialization;
  const parameterCount = validateManifest(manifest);
  if (initial?.length !== parameterCount || !initial.every(Number.isFinite)) throw new Error('Invalid MFN initialization parameters.');
  const parameters = Float64Array.from(initial), layout = structuredClone(manifest.layout), filterLayout = structuredClone(manifest.filterLayout);
  // FourierNet applies input_scale / sqrt(n_layers + 1) only to each learned
  // coordinate-filter kernel at initialization. Preserve the original random
  // draws and biases; the power-of-two sweep matches a fresh float32 NNX
  // constructor byte for byte. These remain trainable weights, not a fixed
  // coordinate rescaling applied on every forward call.
  if (inputScale !== manifest.inputScale) {
    const factor = inputScale / manifest.inputScale;
    for (const filter of filterLayout) for (let index = filter.weightOffset; index < filter.biasOffset; index++) {
      parameters[index] = Math.fround(initial[index] * factor);
      if (!Number.isFinite(parameters[index])) throw new Error('MFN inputScale produces nonfinite filter weights.');
    }
  }
  const width = WIDTH, count = frames * SECTIONS, coordinates = new Float64Array(count * 2);
  const frameTimes = Array.from({length: frames}, (_, frame) => duration * (frames === 1 ? 0 : frame === frames - 1 ? 1 : Math.fround(frame * Math.fround(1 / (frames - 1)))));
  for (let frame = 0; frame < frames; frame++) for (let section = 0; section < SECTIONS; section++) {
    const row = frame * SECTIONS + section;
    coordinates[2 * row] = Math.fround(frameTimes[frame] / duration);
    coordinates[2 * row + 1] = section === SECTIONS - 1 ? 1 : Math.fround(section * Math.fround(1 / (SECTIONS - 1)));
  }
  const config = {width, numAreas: SECTIONS, sections: SECTIONS, frameTimes, duration, coordinateDtype: 32, inputScale, areaScale, hiddenLayers: 3, filterCount: 4};
  const metadata = {family: 'mfn', type: 'fourier-mfn', faithful: true, hiddenWidth: width, hiddenLayers: 3,
    filterCount: 4, activation: 'multiplicative learned sine filters', inputScale, areaScale, outputAct: false,
    parameterCount, trainableCount: parameterCount, frozenCount: 0, seed: 0, peakLearningRate: .005,
    inputs: ['frame / (n_windows - 1)', 'section / 31'], weightOrder: 'output-major: [output,input]',
    initialization: inputScale === manifest.inputScale ? manifest.initialization : `Original FourierNet(N=32,T=${frames},input_scale=${inputScale},rngs=nnx.Rngs(0)); only the learned coordinate-filter kernels are rescaled from the original float32 initialization. All other constructor settings and parameters are unchanged.`,
    initializationInputScale: manifest.inputScale, upstreamCommit: UPSTREAM_COMMIT,
    timeNormalization: 'Endpoint-aligned linspace(0,1,n_windows), as in difftlm.fields._coordinates.',
    areaTransform: {scale: areaScale, bias: AREA_BIAS, epsilon: AREA_FLOOR, maximum: AREA_CEILING}, fixedGlottisArea: FIXED_AREA};
  // Allocate the CPU tape only when first used; full-GPU fitting only needs
  // coordinates, parameters and layout, so it never retains this CPU tape.
  let tape = null, hasForward = false;
  return {
    parameters, parameterCount, layout, filterLayout, coordinates, config, metadata,
    forward() {
      tape ??= workspace(count, width);
      for (let filter = 0; filter < 4; filter++) {
        const spec = filterLayout[filter], sin = tape.sine[filter], cos = tape.cosine[filter];
        for (let row = 0; row < count; row++) for (let out = 0; out < width; out++) {
          const phase = parameters[spec.weightOffset + 2 * out] * coordinates[2 * row] + parameters[spec.weightOffset + 2 * out + 1] * coordinates[2 * row + 1] + parameters[spec.biasOffset + out];
          sin[row * width + out] = Math.sin(phase); cos[row * width + out] = Math.cos(phase);
        }
      }
      tape.activations[0].set(tape.sine[0]);
      for (let layer = 0; layer < 3; layer++) {
        const spec = layout[layer], input = tape.activations[layer], output = tape.activations[layer + 1], affine = tape.affine[layer], filter = tape.sine[layer + 1];
        for (let row = 0; row < count; row++) for (let out = 0; out < width; out++) {
          let value = parameters[spec.biasOffset + out];
          const weight = spec.weightOffset + out * width, start = row * width;
          for (let k = 0; k < width; k++) value += parameters[weight + k] * input[start + k];
          affine[start + out] = value; output[start + out] = value * filter[start + out];
        }
      }
      const logAreas = new Float64Array(count), output = layout[3], activation = tape.activations[3];
      for (let row = 0; row < count; row++) {
        let raw = parameters[output.biasOffset];
        for (let k = 0; k < width; k++) raw += parameters[output.weightOffset + k] * activation[row * width + k];
        tape.rawField[row] = raw;
        if (row % SECTIONS === 0) { logAreas[row] = Math.log(FIXED_AREA); tape.logAreaDerivative[row] = 0; continue; }
        const z = areaScale * raw + AREA_BIAS, unclipped = softplus(z) + AREA_FLOOR, area = Math.min(AREA_CEILING, unclipped);
        logAreas[row] = Math.log(area);
        tape.logAreaDerivative[row] = (unclipped > AREA_CEILING ? 0 : unclipped === AREA_CEILING ? .5 : 1) * areaScale * sigmoid(z) / area;
      }
      hasForward = true;
      return logAreas;
    },
    // The stationary vowel demo supplies its own residual log-area transform.
    forwardRaw() { this.forward(); return tape.rawField.slice(); },
    backwardRaw(cotangent) { return this.backward(cotangent, {raw: true}); },
    backward(cotangent, {raw = false} = {}) {
      if (!hasForward) throw new Error('Call MFN forward() before backward().');
      if (cotangent?.length !== count) throw new Error('The MFN log-area cotangent has the wrong shape.');
      const gradient = new Float64Array(parameterCount), output = layout[3], delta = tape.adjoints[3], activation = tape.activations[3];
      for (let row = 0; row < count; row++) {
        const d = cotangent[row] * (raw ? 1 : tape.logAreaDerivative[row]), start = row * width;
        gradient[output.biasOffset] += d;
        for (let k = 0; k < width; k++) {
          gradient[output.weightOffset + k] += d * activation[start + k];
          delta[start + k] = d * parameters[output.weightOffset + k];
        }
      }
      for (let layer = 2; layer >= 0; layer--) {
        const spec = layout[layer], filterSpec = filterLayout[layer + 1], input = tape.activations[layer];
        const nextDelta = tape.adjoints[layer + 1], previousDelta = tape.adjoints[layer];
        const sine = tape.sine[layer + 1], cosine = tape.cosine[layer + 1], affine = tape.affine[layer];
        previousDelta.fill(0);
        for (let row = 0; row < count; row++) for (let out = 0; out < width; out++) {
          const start = row * width, index = start + out, d = nextDelta[index] * sine[index];
          const phaseDelta = nextDelta[index] * affine[index] * cosine[index];
          gradient[filterSpec.weightOffset + 2 * out] += phaseDelta * coordinates[2 * row];
          gradient[filterSpec.weightOffset + 2 * out + 1] += phaseDelta * coordinates[2 * row + 1];
          gradient[filterSpec.biasOffset + out] += phaseDelta;
          gradient[spec.biasOffset + out] += d;
          const weight = spec.weightOffset + out * width;
          for (let k = 0; k < width; k++) {
            gradient[weight + k] += d * input[start + k];
            previousDelta[start + k] += d * parameters[weight + k];
          }
        }
      }
      const firstFilter = filterLayout[0], firstDelta = tape.adjoints[0], firstCosine = tape.cosine[0];
      for (let row = 0; row < count; row++) for (let out = 0; out < width; out++) {
        const phaseDelta = firstDelta[row * width + out] * firstCosine[row * width + out];
        gradient[firstFilter.weightOffset + 2 * out] += phaseDelta * coordinates[2 * row];
        gradient[firstFilter.weightOffset + 2 * out + 1] += phaseDelta * coordinates[2 * row + 1];
        gradient[firstFilter.biasOffset + out] += phaseDelta;
      }
      return gradient;
    },
    serialize() { return {version: 1, config: structuredClone(config), parameters: parameters.slice(), layout: structuredClone(layout), filterLayout: structuredClone(filterLayout), metadata: structuredClone(metadata)}; },
  };
}
