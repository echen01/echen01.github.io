// node vocal-tract/speech/test-recording.mjs /path/pitch.json /path/audio.f32le
// The fixture must contain actual SwiftF0 output and its 16 kHz mono recording.
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import {createSimulator} from '../simulator/index.js';
import {prepareSpeech, createSpeechFitter} from './engine.js';
import {synthesizeTrajectory} from './dsp.js';

assert(process.argv[2] && process.argv[3], 'Supply a real pitch JSON and raw float32-le PCM fixture.');
const pitch = JSON.parse(await fs.readFile(process.argv[2], 'utf8'));
const bytes = await fs.readFile(process.argv[3]);
const samples = Float32Array.from({length: bytes.length / 4}, (_, i) => bytes.readFloatLE(i * 4));
assert.equal(pitch.sampleRate, 16000);
const sim = await createSimulator(await fs.readFile(new URL('../simulator/difftlm.wasm', import.meta.url)));
const start = performance.now();
const prepared = await prepareSpeech(samples, pitch);
const fitter = await createSpeechFitter(sim, prepared);
const initial = fitter.snapshot();
let result;
for (let i = 0; i < 32; i++) result = await fitter.step();
assert.ok(result.loss < initial.loss * .9, 'Real recording fitting must measurably lower the actual objective.');
assert.ok(result.dataLoss < initial.dataLoss);
assert.ok(result.areas.every(Number.isFinite));
assert.ok(prepared.frameVoiced.some(x => x === 1) && prepared.frameVoiced.some(x => x === 0));
const shapes = Array.from({length: result.frames}, (_, f) => Array.from(result.areas.subarray(f * 32, (f + 1) * 32)));
const audio = await synthesizeTrajectory(sim, {samples, pitch, areas: shapes, frameTimes: result.frameTimes, length: 17.5});
assert.equal(audio.length, samples.length);
assert.ok(audio.every(x => Number.isFinite(x) && Math.abs(x) < .981));
assert.ok(audio.some(x => Math.abs(x) > .01));
const metrics = {
  duration: samples.length / 16000, frames: result.frames, activeFrames: result.activeFrames,
  voicedFrames: result.frameVoiced.reduce((sum, x) => sum + x, 0), iterations: result.iteration,
  initialLoss: initial.loss, finalLoss: result.loss, initialDataLoss: initial.dataLoss, finalDataLoss: result.dataLoss,
  reductionPercent: 100 * (1 - result.loss / initial.loss), elapsedMs: performance.now() - start,
  outputSamples: audio.length, normalizationGain: audio.normalizationGain,
};
console.log(JSON.stringify(metrics, null, 2));
