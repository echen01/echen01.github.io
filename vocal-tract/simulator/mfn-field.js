import {createFaithfulMfnField, loadFaithfulMfnInitialization} from '../speech/faithful-mfn-field.js?v=vowel-mfn-1';
import {createVowelFieldFitter, MIN_LOG_AREA, MAX_LOG_AREA} from './rff-field.js?v=vowel-mfn-1';

export {loadFaithfulMfnInitialization as loadVowelMfnInitialization};
export const VOWEL_MFN_CONFIG = Object.freeze({inputScale: 32, areaScale: .1});

// Same learned sine filters and multiplicative affine layers as the upstream
// FourierNet. The stationary demo predicts a residual on a frozen starting tube.
export function createVowelMfnField(areas, initialization, options = {}) {
  if (areas?.length !== 32 || !Array.from(areas).every(a => Number.isFinite(a) && a >= .12 && a <= 16)) {
    throw new Error('The initial vowel tube needs 32 areas between 0.12 and 16 cm².');
  }
  const network = createFaithfulMfnField(1, 1, initialization, {...VOWEL_MFN_CONFIG, ...options});
  const output = network.layout.at(-1), baseLogAreas = Float64Array.from(areas, Math.log);
  network.parameters.fill(0, output.weightOffset, output.biasOffset + output.biasCount);
  const derivative = new Float64Array(32);
  let hasForward = false;
  const {parameters, parameterCount, config, layout, filterLayout, coordinates} = network;
  const {areaScale} = config;
  const metadata = {...network.metadata, faithful: false, inputs: ['section / 31'], fixedGlottisArea: null,
    areaTransform: {type: 'residual-log-area', scale: areaScale, minimum: .12, maximum: 16},
    initialization: 'Upstream FourierNet hidden layers and learned filters; zero output layer; fixed starting log-area profile'};
  return {parameters, parameterCount, config, layout, filterLayout, coordinates, metadata, baseLogAreas,
    forward() {
      const raw = network.forwardRaw(); hasForward = true;
      return Float64Array.from(raw, (value, i) => {
        const log = baseLogAreas[i] + areaScale * value;
        derivative[i] = log > MIN_LOG_AREA && log < MAX_LOG_AREA ? areaScale : 0;
        return Math.max(MIN_LOG_AREA, Math.min(MAX_LOG_AREA, log));
      });
    },
    backward(cotangent) {
      if (!hasForward || cotangent?.length !== 32) throw new Error('Run the MFN field forward before a matching backward pass.');
      return network.backwardRaw(Float64Array.from(cotangent, (g, i) => g * derivative[i]));
    },
  };
}

export async function createVowelMfnFitter(sim, {areas, fieldConfig = {}, initialization, ...options}) {
  const field = createVowelMfnField(areas, initialization ?? await loadFaithfulMfnInitialization(), fieldConfig);
  return createVowelFieldFitter(sim, field, options);
}
