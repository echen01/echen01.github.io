import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {createSimulator, initialAreas, PRESETS} from '../index.js';
import {createVowelMfnField, createVowelMfnFitter, loadVowelMfnInitialization} from '../mfn-field.js';

const initialization = await loadVowelMfnInitialization(name => readFile(new URL(`../../speech/upstream/reference-mfn/${name}`, import.meta.url)));
const sim = await createSimulator(await readFile(new URL('../difftlm.wasm', import.meta.url)));
const areas = initialAreas(), field = createVowelMfnField(areas, initialization);
assert.equal(field.parameterCount, 200705); assert.equal(field.metadata.family, 'mfn');
assert.equal(field.config.inputScale, 32); assert.equal(field.metadata.fixedGlottisArea, null);
assert.equal(field.config.areaScale, .1); assert.equal(field.metadata.areaTransform.scale, .1);
for (const [i, log] of field.forward().entries()) assert.ok(Math.abs(Math.exp(log) - areas[i]) < 1e-12);
// Use nonzero output weights to exercise gradients through every affine layer
// and every learned sine filter (including frequency weights and phase biases).
const output = field.layout.at(-1);
for (let k = 0; k < output.weightCount; k++) field.parameters[output.weightOffset + k] = .05 * Math.sin(k + 1);
const indices = [...new Set([...field.layout, ...field.filterLayout].flatMap(l => [l.weightOffset, l.weightOffset + 1, l.weightOffset + Math.floor(l.weightCount / 2), l.biasOffset, l.biasOffset + l.biasCount - 1]))];
let maximumError = 0;
for (const vowel of PRESETS) {
  const target = sim.response(vowel.areas, vowel.length).magnitudeDb;
  const objective = () => sim.lossAndGradient(Array.from(field.forward(), Math.exp), target, vowel.length);
  const analytic = field.backward(objective().gradient), epsilon = 1e-5;
  for (const k of indices) {
    const original = field.parameters[k];
    field.parameters[k] = original + epsilon; const plus = objective().loss;
    field.parameters[k] = original - epsilon; const minus = objective().loss;
    field.parameters[k] = original;
    const expected = (plus - minus) / (2 * epsilon);
    const error = Math.abs(analytic[k] - expected) / Math.max(1, Math.abs(expected));
    maximumError = Math.max(maximumError, error);
    assert.ok(error < 2e-7, `${vowel.id} parameter ${k}: ${analytic[k]} vs ${expected}`);
  }
}
field.parameters[output.biasOffset] = 100; field.forward();
assert.ok(field.backward(new Float64Array(32).fill(1)).every(g => g === 0));
const fits = [];
for (const vowel of PRESETS) {
  const fit = await createVowelMfnFitter(sim, {areas, targetAreas: vowel.areas, length: vowel.length, initialization});
  for (let k = 0; k < 240; k++) fit.step();
  const final = fit.snapshot();
  assert.equal(final.parameterization, 'mfn'); assert.equal(final.iteration, 240);
  assert.equal(final.history.length, 241); assert.equal(final.loss, final.history.at(-1));
  assert.ok(final.loss < final.history[0] * .1); assert.ok(final.areas.every(a => a >= .12 && a <= 16));
  assert.ok(Math.abs(sim.lossAndGradient(final.areas, final.targetDb, vowel.length).loss - final.loss) < 1e-12);
  fits.push({vowel: vowel.id, initialLoss: final.history[0], finalLoss: final.loss,
    clippedSections: final.areas.filter(a => a <= .120001 || a >= 15.99999).length});
}
console.log(JSON.stringify({passed: true, checkedGradientCoordinates: 3 * indices.length, maximumError, fits}, null, 2));
