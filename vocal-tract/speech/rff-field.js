/** A continuous random-Fourier-feature neural area field.
 *
 * q = [time / clipDuration, section / (numAreas - 1)]. Frozen Gaussian B
 * embeds q as [cos(2πqB), sin(2πqB)]. Three tanh hidden layers feed one
 * scalar affine output f. Area = min(16, softplus(.35 f + .1) + 1e-4).
 * Section zero is fixed to softplus(.1) + 1e-4, as in the reference model.
 * Unlike endpoint-aligned linspace training coordinates, time / duration keeps
 * the final analysis frame at its actual time and permits continuous queries.
 * All arithmetic and reverse-mode derivatives use JavaScript float64 numbers.
 */
const TAU = 2 * Math.PI;
const AREA_SCALE = .35, AREA_BIAS = .1, AREA_EPSILON = 1e-4, AREA_MAX = 16;
const softplus = value => Math.max(value, 0) + Math.log1p(Math.exp(-Math.abs(value)));
const sigmoid = value => value >= 0 ? 1 / (1 + Math.exp(-value)) : Math.exp(value) / (1 + Math.exp(value));
const FIXED_AREA = softplus(AREA_BIAS) + AREA_EPSILON;

function randomGenerator(seed) {
  let state = seed >>> 0;
  return () => {
    state += 0x6D2B79F5;
    let value = state;
    value = Math.imul(value ^ value >>> 15, value | 1);
    value ^= value + Math.imul(value ^ value >>> 7, value | 61);
    return ((value ^ value >>> 14) >>> 0) / 4294967296;
  };
}

function normalGenerator(random) {
  let spare;
  return () => {
    if (spare !== undefined) { const value = spare; spare = undefined; return value; }
    const radius = Math.sqrt(-2 * Math.log(1 - random())), angle = TAU * random();
    spare = radius * Math.sin(angle);
    return radius * Math.cos(angle);
  };
}

function makeLayout(frequencies, width) {
  const sizes = [2 * frequencies, width, width, width, 1];
  let offset = 0;
  return sizes.slice(1).map((outputSize, index) => {
    const inputSize = sizes[index], weightCount = inputSize * outputSize;
    const layer = {inputSize, outputSize, weightOffset: offset, biasOffset: offset + weightCount,
      weightCount, biasCount: outputSize, activation: index < 3 ? 'tanh' : 'linear', weightShape: [outputSize, inputSize]};
    offset += weightCount + outputSize;
    return layer;
  });
}

function encode(coordinates, encoding, frequencies, destination) {
  const count = coordinates.length / 2, featureSize = frequencies * 2;
  const features = destination ?? new Float64Array(count * featureSize);
  for (let row = 0; row < count; row++) {
    const t = coordinates[row * 2], x = coordinates[row * 2 + 1], offset = row * featureSize;
    for (let k = 0; k < frequencies; k++) {
      const phase = TAU * (t * encoding[k] + x * encoding[frequencies + k]);
      features[offset + k] = Math.cos(phase);
      features[offset + frequencies + k] = Math.sin(phase);
    }
  }
  return features;
}

// Keep hidden activations and adjoints between iterations. Returned log areas
// and parameter gradients stay independently owned by their callers.
function batchWorkspace(layout, count) {
  return {count, activations: [null, ...layout.map(layer => new Float64Array(count * layer.outputSize))],
    adjoints: layout.map(layer => new Float64Array(count * layer.outputSize)),
    logAreaDerivative: new Float64Array(count)};
}

function evaluateBatch(parameters, layout, features, pinnedMask, count, workspace) {
  if (!workspace || workspace.count !== count) workspace = batchWorkspace(layout, count);
  const {activations, logAreaDerivative} = workspace;
  activations[0] = features;
  let input = features;
  for (let layerIndex = 0; layerIndex < layout.length; layerIndex++) {
    const layer = layout[layerIndex], output = activations[layerIndex + 1];
    const {inputSize, outputSize, weightOffset: weights, biasOffset: biases} = layer;
    const tanh = layer.activation === 'tanh', bulk = outputSize - outputSize % 4;
    for (let row = 0; row < count; row++) {
      const inputOffset = row * inputSize, outputOffset = row * outputSize;
      let out = 0;
      // Four independent accumulators reuse each input and preserve each
      // output's original summation order (no reassociation or reduced precision).
      for (; out < bulk; out += 4) {
        let v0 = parameters[biases + out], v1 = parameters[biases + out + 1];
        let v2 = parameters[biases + out + 2], v3 = parameters[biases + out + 3];
        const w0 = weights + out * inputSize, w1 = w0 + inputSize, w2 = w1 + inputSize, w3 = w2 + inputSize;
        for (let k = 0; k < inputSize; k++) {
          const x = input[inputOffset + k];
          v0 += parameters[w0 + k] * x;
          v1 += parameters[w1 + k] * x;
          v2 += parameters[w2 + k] * x;
          v3 += parameters[w3 + k] * x;
        }
        output[outputOffset + out] = tanh ? Math.tanh(v0) : v0;
        output[outputOffset + out + 1] = tanh ? Math.tanh(v1) : v1;
        output[outputOffset + out + 2] = tanh ? Math.tanh(v2) : v2;
        output[outputOffset + out + 3] = tanh ? Math.tanh(v3) : v3;
      }
      for (; out < outputSize; out++) {
        let value = parameters[biases + out];
        const weightOffset = weights + out * inputSize;
        for (let k = 0; k < inputSize; k++) value += parameters[weightOffset + k] * input[inputOffset + k];
        output[outputOffset + out] = tanh ? Math.tanh(value) : value;
      }
    }
    input = output;
  }
  const logAreas = new Float64Array(count);
  for (let row = 0; row < count; row++) {
    if (pinnedMask?.[row]) { logAreas[row] = Math.log(FIXED_AREA); logAreaDerivative[row] = 0; continue; }
    const z = AREA_SCALE * input[row] + AREA_BIAS;
    const rawArea = softplus(z) + AREA_EPSILON, area = Math.min(AREA_MAX, rawArea);
    logAreas[row] = Math.log(area);
    logAreaDerivative[row] = rawArea > AREA_MAX ? 0 : (rawArea === AREA_MAX ? .5 : 1) * AREA_SCALE * sigmoid(z) / area;
  }
  return {...workspace, logAreas};
}

function reverseBatch(parameters, layout, cached, cotangent) {
  if (!cached) throw new Error('Call forward() or forwardAt() before backward().');
  if (cotangent.length !== cached.count) throw new Error('The log-area cotangent must match the most recent forward batch.');
  const gradient = new Float64Array(parameters.length);
  let delta = cached.adjoints[layout.length - 1];
  for (let i = 0; i < cached.count; i++) delta[i] = cotangent[i] * cached.logAreaDerivative[i];
  for (let layerIndex = layout.length - 1; layerIndex >= 0; layerIndex--) {
    const layer = layout[layerIndex], input = cached.activations[layerIndex];
    const {inputSize, outputSize, weightOffset: weights, biasOffset: biases} = layer;
    const previousDelta = layerIndex > 0 ? cached.adjoints[layerIndex - 1] : null;
    const bulk = outputSize - outputSize % 4;
    if (previousDelta) previousDelta.fill(0);
    for (let row = 0; row < cached.count; row++) {
      const inputOffset = row * inputSize, outputOffset = row * outputSize;
      let out = 0;
      for (; out < bulk; out += 4) {
        const d0 = delta[outputOffset + out], d1 = delta[outputOffset + out + 1];
        const d2 = delta[outputOffset + out + 2], d3 = delta[outputOffset + out + 3];
        const w0 = weights + out * inputSize, w1 = w0 + inputSize, w2 = w1 + inputSize, w3 = w2 + inputSize;
        gradient[biases + out] += d0; gradient[biases + out + 1] += d1;
        gradient[biases + out + 2] += d2; gradient[biases + out + 3] += d3;
        if (previousDelta) {
          for (let k = 0; k < inputSize; k++) {
            const x = input[inputOffset + k];
            gradient[w0 + k] += d0 * x; gradient[w1 + k] += d1 * x;
            gradient[w2 + k] += d2 * x; gradient[w3 + k] += d3 * x;
            let d = previousDelta[inputOffset + k];
            d += d0 * parameters[w0 + k]; d += d1 * parameters[w1 + k];
            d += d2 * parameters[w2 + k]; d += d3 * parameters[w3 + k];
            previousDelta[inputOffset + k] = d;
          }
        } else {
          for (let k = 0; k < inputSize; k++) {
            const x = input[inputOffset + k];
            gradient[w0 + k] += d0 * x; gradient[w1 + k] += d1 * x;
            gradient[w2 + k] += d2 * x; gradient[w3 + k] += d3 * x;
          }
        }
      }
      for (; out < outputSize; out++) {
        const d = delta[outputOffset + out], weightOffset = weights + out * inputSize;
        gradient[biases + out] += d;
        for (let k = 0; k < inputSize; k++) {
          gradient[weightOffset + k] += d * input[inputOffset + k];
          if (previousDelta) previousDelta[inputOffset + k] += d * parameters[weightOffset + k];
        }
      }
    }
    if (previousDelta) {
      // Every non-input predecessor is a tanh hidden layer.
      for (let i = 0; i < previousDelta.length; i++) previousDelta[i] *= 1 - input[i] * input[i];
      delta = previousDelta;
    }
  }
  return gradient;
}

function gridCoordinates(frameTimes, duration, numAreas, coordinateDtype = 64) {
  const coordinates = new Float64Array(frameTimes.length * numAreas * 2), pinnedMask = new Uint8Array(frameTimes.length * numAreas);
  for (let frame = 0; frame < frameTimes.length; frame++) for (let section = 0; section < numAreas; section++) {
    const row = frame * numAreas + section;
    coordinates[2 * row] = coordinateDtype === 32 ? Math.fround(frameTimes[frame] / duration) : frameTimes[frame] / duration;
    coordinates[2 * row + 1] = coordinateDtype === 32 ? (section === numAreas - 1 ? 1 : Math.fround(section * Math.fround(1 / (numAreas - 1)))) : section / (numAreas - 1);
    pinnedMask[row] = Number(section === 0);
  }
  return {coordinates, pinnedMask};
}

function checkCoordinates(coordinates, pinnedMask) {
  if (!coordinates?.length || coordinates.length % 2 || !Array.from(coordinates).every(Number.isFinite)) throw new Error('Supply finite [normalized time, normalized position] pairs.');
  if (pinnedMask && pinnedMask.length !== coordinates.length / 2) throw new Error('The pinned mask must have one entry per coordinate pair.');
}

/** Trainable parameters are a flat mutable buffer. Each layer has output-major
 * weights [output,input], then biases. B is frozen (outside this buffer).
 * backward() differentiates the most recent forward()/forwardAt() evaluation;
 * sampleAreas() is read-only and does not replace that differentiation cache.
 */
export function createRffField({frameTimes, duration, numAreas = 32, frequencies = 32, width = 32, timeScale = 10, spaceScale = 2, seed = 42, coordinateDtype = 64, lazyFeatures = false} = {}) {
  if (!Number.isFinite(duration) || duration <= 0) throw new Error('A positive clip duration is required.');
  if (!frameTimes?.length || !Array.from(frameTimes).every(t => Number.isFinite(t) && t >= 0 && t <= duration)) throw new Error('Frame times must be finite and inside the clip.');
  if (!Number.isInteger(numAreas) || numAreas < 2 || !Number.isInteger(frequencies) || frequencies < 1 || !Number.isInteger(width) || width < 1) throw new Error('Field dimensions must be positive integers (at least two areas).');
  if (![timeScale, spaceScale].every(value => Number.isFinite(value) && value >= 0) || !Number.isInteger(seed)) throw new Error('Encoding scales must be nonnegative and the seed must be an integer.');
  const config = {frameTimes: Array.from(frameTimes), duration, numAreas, frequencies, width, timeScale, spaceScale, seed, coordinateDtype};
  const layout = makeLayout(frequencies, width), last = layout.at(-1), parameterCount = last.biasOffset + last.biasCount;
  const parameters = new Float64Array(parameterCount), encoding = new Float64Array(2 * frequencies);
  const random = randomGenerator(seed), normal = normalGenerator(random);
  for (let k = 0; k < frequencies; k++) { encoding[k] = normal() * timeScale; encoding[frequencies + k] = normal() * spaceScale; }
  for (const layer of layout) {
    const bound = Math.sqrt(6 / (layer.inputSize + layer.outputSize));
    for (let k = 0; k < layer.weightCount; k++) parameters[layer.weightOffset + k] = (2 * random() - 1) * bound;
  }
  const {coordinates, pinnedMask} = gridCoordinates(frameTimes, duration, numAreas, coordinateDtype);
  // GPU fitting builds its own float32 features. Defer this much larger CPU
  // matrix until a CPU forward is requested. Capture B at each refresh so its
  // explicit refresh semantics also hold before that first lazy forward.
  let featureEncoding = encoding.slice();
  let features = lazyFeatures ? null : encode(coordinates, encoding, frequencies), cached = null;
  const metadata = {type: 'rff-mlp', inputs: ['time / duration', 'section / (numAreas - 1)'],
    frequencies, featureCount: frequencies * 2, hiddenLayers: 3, hiddenWidth: width, activation: 'tanh', outputSize: 1,
    timeScale, spaceScale, seed, parameterCount, trainableCount: parameterCount, frozenCount: encoding.length,
    encodingShape: [2, frequencies], encodingOrder: 'row-major: time row, space row',
    weightOrder: 'output-major: [output,input]', initialization: 'Seeded Xavier uniform weights; zero biases; frozen Gaussian Fourier frequencies',
    areaTransform: {scale: AREA_SCALE, bias: AREA_BIAS, epsilon: AREA_EPSILON, maximum: AREA_MAX},
    fixedGlottisArea: FIXED_AREA, timeNormalization: 'Physical clip time divided by full duration; analysis-frame endpoints are not stretched.'};
  const field = {
    config, parameters, encoding, B: encoding, layout, parameterCount, metadata,
    forward() {
      features ??= encode(coordinates, featureEncoding, frequencies);
      cached = evaluateBatch(parameters, layout, features, pinnedMask, frameTimes.length * numAreas, cached);
      return cached.logAreas;
    },
    forwardAt(normalizedCoordinates, {pinnedMask: customMask} = {}) {
      checkCoordinates(normalizedCoordinates, customMask);
      cached = evaluateBatch(parameters, layout, encode(normalizedCoordinates, encoding, frequencies), customMask, normalizedCoordinates.length / 2, cached);
      return cached.logAreas;
    },
    // The stationary vowel field supplies its own log-area transform. These
    // methods expose the same MLP before the speech-specific softplus/pinning.
    forwardRawAt(normalizedCoordinates) {
      checkCoordinates(normalizedCoordinates);
      cached = evaluateBatch(parameters, layout, encode(normalizedCoordinates, encoding, frequencies), null, normalizedCoordinates.length / 2, cached);
      return cached.activations.at(-1).slice();
    },
    backwardRaw(cotangent) {
      if (!cached) throw new Error('Call forwardRawAt() before backwardRaw().');
      return reverseBatch(parameters, layout, {...cached, logAreaDerivative: new Float64Array(cached.count).fill(1)}, cotangent);
    },
    backward(logAreaCotangent) { return reverseBatch(parameters, layout, cached, logAreaCotangent); },
    sampleAreas(timeSeconds) {
      if (!Number.isFinite(timeSeconds)) throw new Error('The query time must be finite.');
      const query = gridCoordinates([Math.max(0, Math.min(duration, timeSeconds))], duration, numAreas);
      const result = evaluateBatch(parameters, layout, encode(query.coordinates, encoding, frequencies), query.pinnedMask, numAreas);
      return Float64Array.from(result.logAreas, Math.exp);
    },
    // This is useful only when loading explicit test/checkpoint encoding. Normal
    // training never mutates B. Rebuild the cached training features afterwards.
    refreshEncoding() {
      featureEncoding = encoding.slice();
      features = lazyFeatures ? null : encode(coordinates, encoding, frequencies);
      cached = null;
    },
    serialize() { return {version: 1, config: {...config, frameTimes: [...config.frameTimes]}, parameters: parameters.slice(), encoding: encoding.slice(), layout: structuredClone(layout), metadata: structuredClone(metadata)}; },
  };
  return field;
}

/** Query a worker checkpoint at any physical time, without frame interpolation.
 * The serialization is trusted local model data; no executable code is loaded.
 */
export function evaluateSerializedField(serialized, timeSeconds) {
  if (!Number.isFinite(timeSeconds)) throw new Error('The query time must be finite.');
  const {duration, numAreas, frequencies, width} = serialized.config;
  const layout = makeLayout(frequencies, width);
  const query = gridCoordinates([Math.max(0, Math.min(duration, timeSeconds))], duration, numAreas);
  const result = evaluateBatch(serialized.parameters, layout, encode(query.coordinates, serialized.encoding, frequencies), query.pinnedMask, numAreas);
  return Float64Array.from(result.logAreas, Math.exp);
}
