import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import {execFileSync} from 'node:child_process';
import {fileURLToPath} from 'node:url';
import {createSimulator, createFitter, PRESETS, initialAreas, FREQUENCIES} from '../index.js';

const simulator = await createSimulator(await fs.readFile(new URL('../difftlm.wasm', import.meta.url)));
const fixture = JSON.parse(execFileSync(process.env.PYTHON || 'python3', [fileURLToPath(new URL('./reference.py', import.meta.url))], {encoding: 'utf8'}));
let maxRelativeError = 0;
for (const test of fixture) {
  const actual = simulator.response(test.areas, test.length, test.frequencies);
  for (let i = 0; i < test.frequencies.length; i++) {
    const error = Math.hypot(actual.real[i] - test.real[i], actual.imag[i] - test.imag[i]) / Math.max(1e-25, Math.hypot(test.real[i], test.imag[i]));
    maxRelativeError = Math.max(maxRelativeError, error);
    assert.ok(error < 1e-10, `NumPy transfer mismatch at ${test.frequencies[i]} Hz: ${error}`);
  }
}
console.log(`50 full-matrix NumPy transfer checks passed; max relative error ${maxRelativeError.toExponential(3)}.`);

const areas = initialAreas(), target = simulator.response(PRESETS[0].areas, 17.5).magnitudeDb;
const {gradient} = simulator.lossAndGradient(areas, target, 17.5);
let maxGradientError = 0;
function objective(a) {
  const db = simulator.response(a, 17.5).magnitudeDb;
  return db.reduce((sum, value, i) => sum + ((value - target[i]) * Math.LN10 / 20) ** 2, 0) / db.length;
}
for (let i = 0; i < areas.length; i++) {
  const plus = [...areas], minus = [...areas], epsilon = 1e-5;
  plus[i] *= Math.exp(epsilon); minus[i] *= Math.exp(-epsilon);
  const numeric = (objective(plus) - objective(minus)) / (2 * epsilon);
  const error = Math.abs(gradient[i] - numeric) / (1 + Math.abs(numeric));
  maxGradientError = Math.max(maxGradientError, error);
  assert.ok(error < 2e-6, `Gradient mismatch at area ${i}`);
}
console.log(`32 WASM reverse-mode gradient checks passed; max scaled error ${maxGradientError.toExponential(3)}.`);

for (const preset of PRESETS) {
  const fitter = createFitter(simulator, {areas: initialAreas(), targetAreas: preset.areas, length: preset.length});
  const initialLoss = fitter.snapshot().loss;
  let result;
  const start = performance.now();
  for (let i = 0; i < 240; i++) result = fitter.step();
  assert.ok(result.loss < initialLoss * .2, `${preset.id}: fit should reduce loss by at least 80%`);
  assert.equal(result.history.length, 241);
  assert.equal(result.loss, result.history.at(-1));
  const measuredLoss = simulator.lossAndGradient(result.areas, targetFor(preset), preset.length).loss;
  assert.ok(Math.abs(measuredLoss - result.loss) < 1e-12);
  const audio = simulator.synthesize(result.areas, preset.length);
  assert.equal(audio.sampleRate, 32000);
  assert.equal(audio.samples.length, 25600);
  assert.ok(audio.samples.every(Number.isFinite));
  assert.ok(audio.samples.some(x => Math.abs(x) > .1));
  assert.ok(audio.samples.every(x => Math.abs(x) <= .301));
  assert.equal(Math.abs(audio.samples[0]), 0);
  assert.equal(Math.abs(audio.samples.at(-1)), 0);
  console.log(`${preset.id}: loss ${initialLoss.toFixed(6)} → ${result.loss.toFixed(6)}, ${(100 * (1 - result.loss / initialLoss)).toFixed(2)}% reduction; 240 steps in ${(performance.now() - start).toFixed(0)} ms; audio finite.`);
}
function targetFor(preset) {return simulator.response(preset.areas, preset.length).magnitudeDb;}

assert.throws(() => simulator.response([1, 2], 17.5));
assert.throws(() => simulator.response(areas, NaN));
assert.throws(() => simulator.response(areas, 17.5, [NaN]));
assert.throws(() => simulator.lossAndGradient(areas, [1], 17.5));
assert.throws(() => simulator.synthesize(areas, 17.5, 0));
assert.equal(FREQUENCIES.length, 160);
console.log('Input guards passed.');
