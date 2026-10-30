import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {createSimulator, initialAreas, PRESETS} from '../index.js';
import {createVowelRffField, createVowelRffFitter} from '../rff-field.js';

const sim = await createSimulator(await readFile(new URL('../difftlm.wasm', import.meta.url)));
const areas = initialAreas(), field = createVowelRffField(areas), twin = createVowelRffField(areas);
assert.deepEqual(field.parameters, twin.parameters);
assert.equal(field.parameterCount, 10497);
assert.equal(field.metadata.fixedGlottisArea, null);
for (const [i, log] of field.forward().entries()) assert.ok(Math.abs(Math.exp(log) - areas[i]) < 1e-12);
// Continuous position queries interpolate the frozen starting log-area profile.
const x = .371, p = 31 * x, i = Math.floor(p);
assert.ok(Math.abs(field.forwardAt([x])[0] - ((1 - p + i) * Math.log(areas[i]) + (p - i) * Math.log(areas[i + 1]))) < 1e-12);
// Nonzero output weights exercise all hidden layers in the chained derivative.
const small = createVowelRffField(areas, {frequencies: 3, width: 4});
const last = small.layout.at(-1);
for (let k = 0; k < last.weightCount; k++) small.parameters[last.weightOffset + k] = .15 * Math.sin(k + 1);
let maximumError = 0;
for (const vowel of PRESETS) {
  const target = sim.response(vowel.areas, vowel.length).magnitudeDb;
  const objective = () => sim.lossAndGradient(Array.from(small.forward(), Math.exp), target, vowel.length);
  const analytic = small.backward(objective().gradient), epsilon = 1e-5;
  for (let k = 0; k < small.parameterCount; k++) {
    const original = small.parameters[k];
    small.parameters[k] = original + epsilon; const plus = objective().loss;
    small.parameters[k] = original - epsilon; const minus = objective().loss;
    small.parameters[k] = original;
    const expected = (plus - minus) / (2 * epsilon);
    const error = Math.abs(analytic[k] - expected) / Math.max(1, Math.abs(expected));
    maximumError = Math.max(maximumError, error);
    assert.ok(error < 2e-7, `${vowel.id} parameter ${k}: ${analytic[k]} vs ${expected}`);
  }
}
small.parameters[last.biasOffset] = 100;
small.forward(); assert.ok(small.backward(new Float64Array(32).fill(1)).every(g => g === 0));
const fits = [];
for (const preset of PRESETS) {
  const fitter = createVowelRffFitter(sim, {areas, targetAreas: preset.areas, length: preset.length});
  const first = fitter.snapshot();
  for (let k = 0; k < 240; k++) fitter.step();
  const final = fitter.snapshot(), checked = sim.lossAndGradient(final.areas, final.targetDb, preset.length);
  assert.equal(final.iteration, 240); assert.equal(final.history.length, 241);
  assert.equal(final.parameterization, 'rff'); assert.equal(final.history.at(-1), final.loss);
  assert.ok(final.loss < first.loss * .05); assert.ok(Math.abs(checked.loss - final.loss) < 1e-12);
  assert.ok(final.areas.every(a => a >= .12 && a <= 16));
  assert.throws(() => fitter.step(), /complete/);
  fits.push({vowel: preset.id, initialLoss: first.loss, finalLoss: final.loss});
}
console.log(JSON.stringify({passed: true, checkedGradientCoordinates: 3 * small.parameterCount, maximumError, fits}, null, 2));
