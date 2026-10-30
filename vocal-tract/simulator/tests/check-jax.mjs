// DIFFTLM_JAX_PYTHON=/path/to/python node tests/check-jax.mjs [--upstream /checkout] [--report /result.json]
import assert from 'node:assert/strict';
import {spawn} from 'node:child_process';
import {createHash} from 'node:crypto';
import fs from 'node:fs/promises';
import {fileURLToPath} from 'node:url';
import {createSimulator} from '../index.js';
import cases from './jax-cases.mjs';

const options = new Map();
for (let i = 2; i < process.argv.length; i += 2) {
  assert(['--upstream', '--report'].includes(process.argv[i]) && process.argv[i + 1], 'Unknown or incomplete option');
  options.set(process.argv[i], process.argv[i + 1]);
}
const script = fileURLToPath(new URL('./jax-reference.py', import.meta.url));
const args = [script, ...(options.has('--upstream') ? ['--upstream', options.get('--upstream')] : [])];
const python = spawn(process.env.DIFFTLM_JAX_PYTHON || 'python3', args, {stdio: ['pipe', 'pipe', 'inherit']});
const output = [];
python.stdout.on('data', data => output.push(data));
const finished = new Promise((resolve, reject) => {
  python.on('error', reject);
  python.on('close', code => code === 0 ? resolve() : reject(new Error(`JAX reference exited with ${code}`)));
});
python.stdin.end(JSON.stringify(cases));
await finished;
const reference = JSON.parse(Buffer.concat(output).toString('utf8'));
const wasm = await fs.readFile(new URL('../difftlm.wasm', import.meta.url));
const sim = await createSimulator(wasm);
const tolerances = {forwardRelative: 1e-8, lossScaled: 1e-8, gradientScaled: 1e-7, gradientRelativeL2: 1e-6};
const results = [];
assert.equal(reference.cases.length, cases.length);
for (let i = 0; i < cases.length; i++) {
  const test = cases[i], expected = reference.cases[i];
  assert.equal(test.name, expected.name);
  const h = sim.response(test.areas, test.length, expected.frequencies);
  const actual = sim.lossAndGradient(test.areas, expected.targetDb, test.length, expected.frequencies);
  let forwardRelative = 0, gradientScaled = 0, gradientAbsolute = 0, worstCoordinate = 0;
  for (let k = 0; k < h.real.length; k++) {
    const error = Math.hypot(h.real[k] - expected.real[k], h.imag[k] - expected.imag[k]);
    forwardRelative = Math.max(forwardRelative, error / Math.max(1e-30, Math.hypot(expected.real[k], expected.imag[k])));
  }
  const delta = actual.gradient.map((g, j) => g - expected.gradient[j]);
  for (let j = 0; j < delta.length; j++) {
    const scaled = Math.abs(delta[j]) / (1 + Math.abs(expected.gradient[j]));
    if (scaled > gradientScaled) { gradientScaled = scaled; worstCoordinate = j; }
    gradientAbsolute = Math.max(gradientAbsolute, Math.abs(delta[j]));
  }
  const gradientNorm = Math.hypot(...expected.gradient);
  const gradientRelativeL2 = Math.hypot(...delta) / Math.max(1e-12, gradientNorm);
  const lossScaled = Math.abs(actual.loss - expected.loss) / (1 + Math.abs(expected.loss));
  const result = {name: test.name, bins: expected.frequencies.length, length: test.length,
    lossJax: expected.loss, lossWasm: actual.loss, forwardRelative, lossScaled,
    gradientAbsolute, gradientScaled, gradientRelativeL2, gradientNorm, worstCoordinate};
  result.passed = Object.entries(tolerances).every(([key, tolerance]) => Number.isFinite(result[key]) && result[key] <= tolerance);
  const centered = sim.gainInvariantLossAndGradient(test.areas, expected.targetDb, test.length, expected.frequencies);
  const centeredDelta = centered.gradient.map((g, j) => g - expected.gainInvariantGradient[j]);
  result.gainInvariantLossScaled = Math.abs(centered.loss - expected.gainInvariantLoss) / (1 + Math.abs(expected.gainInvariantLoss));
  result.gainInvariantGradientScaled = Math.max(...centeredDelta.map((d, j) => Math.abs(d) / (1 + Math.abs(expected.gainInvariantGradient[j]))));
  result.gainInvariantGradientRelativeL2 = Math.hypot(...centeredDelta) / Math.max(1e-12, Math.hypot(...expected.gainInvariantGradient));
  result.gainDbAbsolute = Math.abs(centered.gainDb - expected.gainDb);
  result.passed &&= result.gainInvariantLossScaled <= tolerances.lossScaled
    && result.gainInvariantGradientScaled <= tolerances.gradientScaled
    && result.gainInvariantGradientRelativeL2 <= tolerances.gradientRelativeL2
    && result.gainDbAbsolute <= 1e-8;
  results.push(result);
  console.log(`${result.passed ? 'PASS' : 'FAIL'} ${test.name}: forward ${forwardRelative.toExponential(2)}, gradient scaled ${gradientScaled.toExponential(2)}, relative L2 ${gradientRelativeL2.toExponential(2)}`);
}
const {cases: unused, ...metadata} = reference;
const report = {
  ...metadata,
  wasmSha256: createHash('sha256').update(wasm).digest('hex'),
  caseSpecificationSha256: createHash('sha256').update(JSON.stringify(cases)).digest('hex'),
  tolerances,
  summary: {
    passed: results.every(result => result.passed),
    cases: results.length,
    frequencyComparisons: results.reduce((sum, result) => sum + result.bins, 0),
    gradientCoordinates: results.length * 32 * 2,
    objectivesPerCase: 2,
    maxForwardRelative: Math.max(...results.map(r => r.forwardRelative)),
    maxLossScaled: Math.max(...results.map(r => r.lossScaled)),
    maxGradientScaled: Math.max(...results.map(r => r.gradientScaled)),
    maxGradientRelativeL2: Math.max(...results.map(r => r.gradientRelativeL2)),
    maxGainInvariantGradientScaled: Math.max(...results.map(r => r.gainInvariantGradientScaled)),
    maxGainInvariantGradientRelativeL2: Math.max(...results.map(r => r.gainInvariantGradientRelativeL2)),
    maxGainInvariantLossScaled: Math.max(...results.map(r => r.gainInvariantLossScaled)),
  },
  results,
};
if (options.has('--report')) await fs.writeFile(options.get('--report'), JSON.stringify(report, null, 2) + '\n');
console.log(JSON.stringify(report.summary, null, 2));
assert(report.summary.passed, 'JAX/WASM comparison failed; see per-case errors above.');
