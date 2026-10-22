// Run: node vocal-tract/speech/test.mjs
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import {createSimulator, PRESETS, initialAreas} from '../simulator/index.js';
import {gainInvariantLossAndGradient, regularization, prepareSpeech, createSpeechFitter, evaluateTrajectory} from './engine.js';
import {fft, windowSpectrum, synthesizeTrajectory} from './dsp.js';

const sim = await createSimulator(await fs.readFile(new URL('../simulator/difftlm.wasm', import.meta.url)));
const frequencies = Array.from({length: 30}, (_, i) => 90 + i * 151);
const areas = initialAreas();
const target = sim.response(PRESETS[0].areas, 17.5, frequencies).magnitudeDb.map(x => x + 37.2);
const evaluation = gainInvariantLossAndGradient(sim, areas, target, frequencies, 17.5);
const shifted = gainInvariantLossAndGradient(sim, areas, target.map(x => x - 83), frequencies, 17.5);
assert.ok(Math.abs(evaluation.loss - shifted.loss) < 1e-12);
assert.ok(evaluation.gradient.every((g, i) => Math.abs(g - shifted.gradient[i]) < 1e-11));
let gradientError = 0;
for (let i = 0; i < 32; i++) {
  const plus = [...areas], minus = [...areas], epsilon = 1e-5;
  plus[i] *= Math.exp(epsilon); minus[i] *= Math.exp(-epsilon);
  const finite = (gainInvariantLossAndGradient(sim, plus, target, frequencies, 17.5).loss - gainInvariantLossAndGradient(sim, minus, target, frequencies, 17.5).loss) / (2 * epsilon);
  gradientError = Math.max(gradientError, Math.abs(finite - evaluation.gradient[i]) / (1 + Math.abs(finite)));
}
assert.ok(gradientError < 2e-6);
console.log(`Gain invariance and 32 log-area derivatives passed; maximum scaled gradient error ${gradientError.toExponential(3)}.`);

const logs = Float64Array.from({length: 96}, (_, i) => Math.sin(i * .3) + .2 * Math.cos(i * .9));
const penalty = regularization(logs, 3);
for (let i = 0; i < logs.length; i++) {
  const plus = logs.slice(), minus = logs.slice(), epsilon = 1e-6;
  plus[i] += epsilon; minus[i] -= epsilon;
  const a = regularization(plus, 3), b = regularization(minus, 3);
  const numerical = ((a.spatialLoss + a.temporalLoss) - (b.spatialLoss + b.temporalLoss)) / (2 * epsilon);
  assert.ok(Math.abs(numerical - penalty.gradient[i]) < 1e-9);
}
console.log('Spatial and temporal smoothness gradients passed at all 96 coordinates.');

const smallPrepared = {
  frames: [
    {active: true, frequencies, targetDb: target},
    {active: false, frequencies, targetDb: target},
    {active: true, frequencies, targetDb: sim.response(PRESETS[2].areas, 17.5, frequencies).magnitudeDb},
  ], activeFrames: 2,
};
const combined = await evaluateTrajectory(sim, smallPrepared, logs);
for (const i of [0, 16, 31, 32, 55, 63, 64, 95]) {
  const plus = logs.slice(), minus = logs.slice(), epsilon = 1e-5;
  plus[i] += epsilon; minus[i] -= epsilon;
  const a = await evaluateTrajectory(sim, smallPrepared, plus), b = await evaluateTrajectory(sim, smallPrepared, minus);
  const numerical = (a.totalLoss - b.totalLoss) / (2 * epsilon);
  assert.ok(Math.abs(numerical - combined.gradient[i]) / (1 + Math.abs(numerical)) < 2e-6);
}
console.log('Combined normalized objective gradients passed, including silent frames and tract/time boundaries.');

const raw = Float64Array.from({length: 1024}, (_, i) => Math.sin(i * .17) + .3 * Math.cos(i * .49)), original = raw.slice();
const imaginary = new Float64Array(1024);
fft(raw, imaginary); fft(raw, imaginary, true);
assert.ok(raw.every((x, i) => Math.abs(x - original[i]) < 1e-11));
assert.equal(windowSpectrum(new Float32Array(3200), 0).magnitude.length, 513);

const sampleRate = 16000, duration = 1.0;
const envelope = Float32Array.from({length: sampleRate * duration}, (_, i) => .12 * Math.sin(2 * Math.PI * 173 * i / sampleRate));
const pitch = {
  times: Float64Array.from({length: 64}, (_, i) => i * .016),
  pitchHz: Float32Array.from({length: 64}, (_, i) => 150 + 60 * i / 63),
  voiced: Uint8Array.from({length: 64}, (_, i) => i < 25 || i > 38 ? 1 : 0),
};
const knownAreas = [PRESETS[0].areas, PRESETS[1].areas, PRESETS[2].areas];
const targetAudio = await synthesizeTrajectory(sim, {samples: envelope, pitch, areas: knownAreas, frameTimes: [0, .5, 1], length: 17.5});
assert.equal(targetAudio.length, envelope.length);
assert.ok(targetAudio.every(Number.isFinite));
const prepared = await prepareSpeech(targetAudio, pitch);
assert.equal(prepared.frames.length, 50);
assert.ok(prepared.frameVoiced.some(x => x === 0) && prepared.frameVoiced.some(x => x === 1));
assert.equal(prepared.activeFrames, 50);
const fitter = await createSpeechFitter(sim, prepared, {length: 17.5});
const initial = fitter.snapshot();
let result;
const started = performance.now();
for (let i = 0; i < 24; i++) result = await fitter.step();
assert.ok(result.loss < initial.loss * .8, `Loss did not improve enough: ${initial.loss} → ${result.loss}`);
assert.equal(result.history.length, 25);
assert.equal(result.history.at(-1), result.loss);
assert.ok(result.areas.every(a => Number.isFinite(a) && a >= .12 - 1e-12 && a <= 16 + 1e-12));
const repeated = await evaluateTrajectory(sim, prepared, Float64Array.from(result.areas, Math.log), {length: 17.5});
assert.ok(Math.abs(repeated.totalLoss - result.loss) < 1e-12, 'Displayed loss must match returned areas.');
const trajectory = Array.from({length: result.frames}, (_, f) => Array.from(result.areas.subarray(f * 32, (f + 1) * 32)));
let motion = 0;
for (let i = 0; i < 32; i++) motion += Math.abs(trajectory[5][i] - trajectory[40][i]);
assert.ok(motion > .5, 'Different speech frames must learn different tract areas.');
const audio = await synthesizeTrajectory(sim, {samples: targetAudio, pitch, areas: trajectory, frameTimes: result.frameTimes, length: 17.5});
assert.equal(audio.length, targetAudio.length);
assert.ok(audio.every(x => Number.isFinite(x) && Math.abs(x) <= .981));
assert.ok(audio.some(x => Math.abs(x) > .01));
const negated = Float32Array.from(targetAudio, x => -x);
const noPhaseLeak = await synthesizeTrajectory(sim, {samples: negated, pitch, areas: trajectory, frameTimes: result.frameTimes, length: 17.5});
assert.ok(audio.every((x, i) => x === noPhaseLeak[i]), 'Reconstruction must not copy target waveform phase.');
console.log(`50 voiced/unvoiced frames: 24 updates reduced total loss ${initial.loss.toFixed(5)} → ${result.loss.toFixed(5)} (${(100 * (1 - result.loss / initial.loss)).toFixed(1)}%) in ${((performance.now() - started) / 1000).toFixed(2)} s; area motion ${motion.toFixed(2)} cm².`);
console.log('Reconstruction duration, finite bounds, moving areas, exact reported loss, and absence of copied target phase passed.');

await assert.rejects(() => prepareSpeech(new Float32Array(16000), pitch), /silent/);
const controller = new AbortController(); controller.abort();
await assert.rejects(() => prepareSpeech(envelope, pitch, {signal: controller.signal}), /canceled/);
const silence = await synthesizeTrajectory(sim, {samples: new Float32Array(16000), pitch, areas: knownAreas, frameTimes: [0, .5, 1], length: 17.5});
assert.ok(silence.every(x => x === 0));
console.log('Silence and cancellation guards passed.');
