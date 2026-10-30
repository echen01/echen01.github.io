import {createFaithfulRollout} from './faithful-rollout.js';
import {createFaithfulLoss} from './faithful-loss.js';
import {createFaithfulAdam} from './faithful-field.js?v=fixed-grid';
import {faithfulNetwork} from './faithful-networks.js?v=mfn-area-scale-1';

export async function loadFaithfulExample({sampleRate = 16000} = {}) {
  if (![16000, 32000].includes(sampleRate)) throw new Error('The bundled example supports 16 or 32 kHz.');
  const base = new URL(sampleRate === 16000 ? './upstream/reference-16k/' : './upstream/reference/', import.meta.url);
  const metadata = await (await fetch(new URL('speech-sample.json', base))).json();
  const read = async descriptor => new Float32Array(await (await fetch(new URL(descriptor.file, base))).arrayBuffer());
  const [target, uin] = await Promise.all([read(metadata.target), read(metadata.uin)]);
  return {target, uin, metadata, sampleRate: metadata.sampleRate};
}

/** CPU64 reference pipeline; optionally a GPU field, with explicit backend. */
export async function createFaithfulFitter({target, uin, sampleRate = 32000, metadata = {}}, {
  iterations = 500, fieldType = 'rff', hiddenDim = fieldType === 'mfn' ? 256 : 512, mfnInputScale = 256, mfnAreaScale = .35, initialization, fieldFactory = field => field, signal, onStage,
} = {}) {
  const network = faithfulNetwork(fieldType, hiddenDim);
  if (target.length !== uin.length) throw new Error('Target and glottal source must have equal length.');
  const rollout = createFaithfulRollout(uin, {sampleRate}), objective = createFaithfulLoss(target, sampleRate);
  const initial = initialization ?? await network.load(undefined, {hiddenDim});
  if (initial.manifest.hiddenDim !== hiddenDim) throw new Error('Requested hidden width does not match the supplied initialization.');
  const field = await fieldFactory(network.create(rollout.frames, target.length / sampleRate, initial, {inputScale: mfnInputScale, areaScale: mfnAreaScale}));

  const adam = createFaithfulAdam(field.parameters, {iterations, peak: network.peak});
  const history = []; let lastAreas, lastLoss = NaN;
  function check() { if (signal?.aborted) throw new DOMException('Fitting canceled.', 'AbortError'); }
  const frameTimes = Float64Array.from({length: rollout.frames}, (_, i) => (i * rollout.hopLength + rollout.frameLength / 2) / sampleRate);
  const framePitchHz = Float64Array.from(frameTimes, t => metadata.pitchHz?.[Math.min(metadata.pitchHz.length - 1, Math.floor(t / .016))] ?? 0);
  const frameVoiced = Uint8Array.from(frameTimes, t => Number((metadata.confidence?.[Math.min(metadata.confidence.length - 1, Math.floor(t / .016))] ?? 0) >= .5));
  function snapshot() {
    const serialized = field.serialize();
    // Acoustic window centers map to the trained endpoint coordinates when
    // querying the continuous field in the visualization.
    serialized.timeMapping = {start: frameTimes[0], end: frameTimes.at(-1)};
    return {iteration: adam.step, frames: rollout.frames, areas: lastAreas?.slice(), frameTimes, framePitchHz, frameVoiced,
      frameActive: new Uint8Array(rollout.frames).fill(1), length: 17.5, duration: target.length / sampleRate,
      history: [...history], loss: lastLoss, initialLoss: history[0], field: serialized, fieldMetadata: field.metadata, backend: field.backend === 'webgpu-f32' ? 'faithful-hybrid' : 'faithful-cpu',
      model: `Pinned diffTLM ${network.label} waveform objective; no auxiliary loss terms.`};
  }
  return {
    field, rollout, history,
    async step() {
      check(); onStage?.(`Evaluating the ${network.label} field…`);
      const logs = await field.forward(); lastAreas = Float64Array.from(logs, Math.exp);
      check(); onStage?.('Synthesizing the complete waveform with turbulence…');
      const synthesis = rollout.forward(lastAreas, {iteration: adam.step});
      const loss = objective.evaluate(synthesis.samples); lastLoss = loss.loss;
      check(); onStage?.('Backpropagating the mel-STFT waveform loss…');
      const areaGradient = synthesis.backward(loss.gradient);
      const gradient = await field.backward(areaGradient);
      if (!gradient.every(Number.isFinite) || !Number.isFinite(lastLoss)) throw new Error('Nonfinite faithful objective or gradient.');
      history.push(lastLoss); adam.update(gradient);
      return {iteration: adam.step, loss: lastLoss};
    },
    async snapshot() { lastAreas = Float64Array.from(await field.forward(), Math.exp); return snapshot(); },
    async synthesize() {
      check(); lastAreas = Float64Array.from(await field.forward(), Math.exp);
      return rollout.forward(lastAreas, {iteration: 0}).samples;
    },
    dispose() { field.destroy?.(); },
  };
}
