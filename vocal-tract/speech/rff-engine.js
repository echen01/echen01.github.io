// RFF-MLP parameterization of the browser acoustic inverse problem.
// All trainable values are shared network weights; no free per-frame areas.
import {createRffField} from './rff-field.js';
import {evaluateTrajectory, MODEL_DESCRIPTION} from './engine.js';

export const RFF_DEFAULT_ITERATIONS = 120;
export const RFF_MODEL_DESCRIPTION = {
  ...MODEL_DESCRIPTION,
  name: 'RFF neural field with harmonic / spectral-envelope reconstruction',
  parameterization: 'Frozen Gaussian Fourier features of normalized time and position; three tanh hidden layers and a scalar area output',
  limitations: 'Browser-sized network and approximate unvoiced source; no localized turbulence, waveform phase loss, or anatomical uniqueness guarantee',
};

export async function createRffSpeechFitter(sim, prepared, {
  length = 17.5, iterations = RFF_DEFAULT_ITERATIONS, learningRate = .01,
  spatialWeight = .006, temporalWeight = .08, signal, fieldConfig = {}, fieldFactory = createRffField,
} = {}) {
  if (!Number.isFinite(length) || length < 12 || length > 22) throw new Error('Choose a tract length between 12 and 22 cm.');
  if (!Number.isInteger(iterations) || iterations < 1 || iterations > 500) throw new Error('Choose between 1 and 500 optimization steps.');
  if (!Number.isFinite(learningRate) || learningRate <= 0 || learningRate > .05) throw new Error('Invalid RFF learning rate.');
  const check = () => { if (signal?.aborted) throw new DOMException('RFF fitting canceled.', 'AbortError'); };
  check();
  const field = await fieldFactory({...fieldConfig, frameTimes: prepared.frameTimes, duration: prepared.duration, numAreas: 32});
  // The worker yields between whole steps, so six nested timer yields per
  // trajectory are redundant. No frequencies, frames, or gradients are skipped.
  const settings = {length, spatialWeight, temporalWeight, signal, yieldEveryFrames: 0};
  let logs, evaluation, gradient;
  try {
    check();
    logs = await field.forward();
    evaluation = await evaluateTrajectory(sim, prepared, logs, settings);
    gradient = await field.backward(evaluation.gradient);
    check();
  } catch (error) {
    field.destroy?.();
    throw error;
  }
  const m = new Float64Array(field.parameterCount), v = new Float64Array(field.parameterCount);
  let iteration = 0, currentLearningRate = 0;
  const initialLoss = evaluation.totalLoss, history = [initialLoss];
  const snapshot = () => ({iteration, iterations, loss: evaluation.totalLoss, totalLoss: evaluation.totalLoss,
    dataLoss: evaluation.dataLoss, spatialLoss: evaluation.spatialLoss, temporalLoss: evaluation.temporalLoss,
    initialLoss, history: [...history], areas: Float64Array.from(logs, Math.exp),
    frames: prepared.frames.length, frameTimes: [...prepared.frameTimes], framePitchHz: [...prepared.framePitchHz],
    frameVoiced: [...prepared.frameVoiced], frameActive: [...prepared.frameActive], activeFrames: prepared.activeFrames,
    gainDb: Array.from(evaluation.gainsDb), length, framePeriod: prepared.hopLength / prepared.sampleRate,
    duration: prepared.duration, parameterization: 'rff', field: field.serialize(),
    fieldMetadata: field.metadata, parameterCount: field.parameterCount, learningRate: currentLearningRate,
    backend: field.backend || 'JavaScript float64 + Rust/WASM float64',
    fallbackReason: field.fallbackReason || null,
  });
  return {
    snapshot,
    dispose() { field.destroy?.(); },
    async step() {
      check();
      if (iteration >= iterations) throw new Error('The requested RFF optimization steps are complete.');
      iteration++;
      // Short warmup followed by cosine decay, ending at 10% of the peak rate.
      const warmup = Math.min(10, Math.max(1, Math.floor(iterations / 10)));
      const progress = Math.max(0, (iteration - warmup) / Math.max(1, iterations - warmup));
      currentLearningRate = learningRate * (iteration <= warmup ? iteration / warmup : .1 + .9 * .5 * (1 + Math.cos(Math.PI * progress)));
      let normSquared = 0;
      for (const value of gradient) normSquared += value * value;
      const clip = Math.min(1, 5 / Math.max(1e-12, Math.sqrt(normSquared)));
      const mScale = 1 - .9 ** iteration, vScale = 1 - .999 ** iteration;
      for (let i = 0; i < field.parameterCount; i++) {
        const g = gradient[i] * clip;
        m[i] = .9 * m[i] + .1 * g; v[i] = .999 * v[i] + .001 * g * g;
        field.parameters[i] -= currentLearningRate * (m[i] / mScale) / (Math.sqrt(v[i] / vScale) + 1e-8);
      }
      logs = await field.forward();
      evaluation = await evaluateTrajectory(sim, prepared, logs, settings);
      check();
      gradient = await field.backward(evaluation.gradient);
      if (!gradient.every(Number.isFinite)) throw new Error('The RFF field produced an invalid gradient.');
      history.push(evaluation.totalLoss);
      return snapshot();
    },
  };
}
