import assert from 'node:assert/strict';
import {computeSpectrogram, sharedSpectrogramScale, spectralDecibels, spectrogramColor, createSpectrogramComparison, quantizeSpectrogram} from './spectrogram.js';

const sampleRate = 16000;
const tone = Float32Array.from({length: sampleRate}, (_, i) => Math.sin(2 * Math.PI * 1000 * i / sampleRate));
const target = computeSpectrogram(tone), identical = computeSpectrogram(tone);
assert.equal(target.frames, 125); assert.equal(target.bins, 257); assert.equal(target.duration, 1);
assert.deepEqual(target.power, identical.power);
const frame = 20, powers = target.power.subarray(frame * target.bins, (frame + 1) * target.bins);
const peakBin = powers.indexOf(Math.max(...powers));
assert.equal(peakBin * sampleRate / target.fftSize, 1000);
assert(Math.abs(powers[peakBin] - 1) < 1e-6, 'A full-amplitude bin-centered tone should have unit spectral power.');
const scale = sharedSpectrogramScale(target, identical);
for (let i = 0; i < target.power.length; i++) {
  const db = spectralDecibels(target.power[i], scale);
  assert.equal(db, spectralDecibels(identical.power[i], scale));
  assert.deepEqual(spectrogramColor(db), spectrogramColor(spectralDecibels(identical.power[i], scale)));
}

// One shared peak preserves level differences instead of independently
// normalizing both panels and hiding a reconstruction amplitude mismatch.
const quieter = computeSpectrogram(Float32Array.from(tone, x => x / 2));
const pairScale = sharedSpectrogramScale(target, quieter);
const levelDifference = spectralDecibels(quieter.power[frame * quieter.bins + peakBin], pairScale)
  - spectralDecibels(target.power[frame * target.bins + peakBin], pairScale);
assert(Math.abs(levelDifference + 6.020599913) < 1e-7);
assert.equal(pairScale.referencePower, target.peakPower);
const restored = computeSpectrogram(Float32Array.from(tone, x => 4 * x), {amplitudeScale: 1 / 4});
assert.deepEqual(restored.power, target.power, 'Removing playback gain should restore the training waveform spectrum.');

// Check time ordering as well as frequency: switch from 1 to 2 kHz halfway.
const changing = Float32Array.from(tone, (x, i) => i < sampleRate / 2 ? x : Math.sin(2 * Math.PI * 2000 * i / sampleRate));
const sequence = computeSpectrogram(changing);
const dominant = n => { const row = sequence.power.subarray(n * sequence.bins, (n + 1) * sequence.bins); return row.indexOf(Math.max(...row)); };
assert.equal(dominant(20), 32); assert.equal(dominant(100), 64);
const silent = computeSpectrogram(new Float32Array(321));
assert(silent.power.every(value => value === 0));
assert.equal(spectralDecibels(0, sharedSpectrogramScale(silent)), -80);
assert.throws(() => computeSpectrogram(new Float32Array([NaN])), /finite/);
assert.throws(() => computeSpectrogram(tone, {fftSize: 511}), /FFT/);
assert.throws(() => computeSpectrogram(tone, {amplitudeScale: 0}), /amplitude/);

const fixedScale = {...sharedSpectrogramScale(target), maximumDb:20};
assert.equal(spectralDecibels(target.peakPower * 100, fixedScale), 20);
assert.equal(spectralDecibels(target.peakPower * 10000, fixedScale), 20, 'Values outside the displayed range clip to endpoint colors.');
const compact = quantizeSpectrogram(target, fixedScale);
assert(compact instanceof Uint8Array);
assert.equal(compact.byteLength, target.power.length, 'Checkpoint history keeps one byte per bin.');

// Minimal canvas fixture verifies shared axes, identical pixel colors, correct
// reconstruction gain handling, pending/reset states, and cached-only resize.
function fakeCanvas() {
  const canvas = {clientWidth: 360, clientHeight: 170, attributes: {}, setAttribute(name, value) { this.attributes[name] = value; }};
  const context = {texts: [], setTransform() { this.texts = []; this.bitmap = null; }, fillRect() {},
    fillText(text, x, y) { this.texts.push({text, x, y}); },
    createImageData(width, height) { return {data: new Uint8ClampedArray(width * height * 4)}; },
    putImageData(image) { canvas.pixels = image.data; },
    drawImage(bitmap, ...bounds) { this.bitmap = bitmap; this.bounds = bounds; },
  };
  canvas.getContext = () => context; canvas.context = context; return canvas;
}
function fakeControl() {
  return {listeners:{},attributes:{},setAttribute(name,value){this.attributes[name]=value;},
    addEventListener(name,callback){this.listeners[name]=callback;},removeEventListener(name){delete this.listeners[name];},
    dispatch(name){this.listeners[name]?.();}};
}
const oldDocument = globalThis.document, oldWindow = globalThis.window, oldObserver = globalThis.ResizeObserver;
let resize;
try {
  globalThis.document = {createElement: fakeCanvas}; globalThis.window = {devicePixelRatio: 2};
  globalThis.ResizeObserver = class { constructor(callback) { resize = callback; } observe() {} disconnect() {} };
  const targetCanvas = fakeCanvas(), reconstructedCanvas = fakeCanvas(), status = {};
  const comparison = createSpectrogramComparison({targetCanvas, reconstructedCanvas, status});
  comparison.reset(); assert.equal(status.textContent, 'Awaiting audio');
  comparison.setTarget(tone, sampleRate);
  assert.equal(status.textContent, 'Target ready · reconstruction pending');
  assert(!reconstructedCanvas.context.bitmap);
  comparison.setReconstruction(Float32Array.from(tone, x => x * 4), {sampleRate, normalizationGain: 4});
  assert.deepEqual(targetCanvas.context.bitmap.pixels, reconstructedCanvas.context.bitmap.pixels);
  assert.deepEqual(targetCanvas.context.texts, reconstructedCanvas.context.texts, 'Time and frequency axes must match.');
  assert.deepEqual(targetCanvas.context.bounds, reconstructedCanvas.context.bounds);
  const cached = targetCanvas.context.bitmap;
  targetCanvas.clientWidth = 310; reconstructedCanvas.clientWidth = 310; resize();
  assert.equal(targetCanvas.context.bitmap, cached, 'Resizing must reuse the cached raster, without STFT computation.');
  assert.equal(targetCanvas.width, 620); assert.equal(reconstructedCanvas.width, 620);
  comparison.clearReconstruction('Fitting stopped. Fit again to compare the reconstruction.');
  assert(!reconstructedCanvas.context.bitmap);
  assert.match(reconstructedCanvas.attributes['aria-label'], /Fitting stopped/);
  assert(targetCanvas.context.bitmap, 'Cancellation retains the current target.');
  comparison.reset(); assert(!targetCanvas.context.bitmap); assert(!reconstructedCanvas.context.bitmap);
  comparison.dispose();

  const controls=fakeControl(), slider=fakeControl(), label={}, follow=fakeControl();
  const training=createSpectrogramComparison({targetCanvas,reconstructedCanvas,status,
    trainingControls:controls,stepSlider:slider,stepLabel:label,followLatest:follow});
  training.setTarget(tone,sampleRate);
  assert(controls.hidden);
  training.addCheckpoint(target,{iteration:0,iterations:200});
  const initialPixels=reconstructedCanvas.context.bitmap.pixels.slice(), targetPixels=targetCanvas.context.bitmap.pixels.slice();
  assert.equal(label.textContent,'0 / 200'); assert(!controls.hidden); assert(slider.disabled); assert(follow.checked);
  training.addCheckpoint(quieter,{iteration:10,iterations:200});
  const quietPixels=reconstructedCanvas.context.bitmap.pixels.slice();
  assert.notDeepEqual(quietPixels,initialPixels); assert.equal(label.textContent,'10 / 200'); assert(!slider.disabled);
  slider.value=0;slider.dispatch('input');
  assert(!follow.checked);assert.equal(label.textContent,'0 / 200');
  assert.deepEqual(reconstructedCanvas.context.bitmap.pixels,initialPixels);
  const louder=computeSpectrogram(Float32Array.from(tone,x=>10*x));
  training.addCheckpoint(louder,{iteration:25,iterations:200});
  assert.equal(label.textContent,'0 / 200','Incoming previews must not steal manual history selection.');
  assert.deepEqual(reconstructedCanvas.context.bitmap.pixels,initialPixels);
  assert.deepEqual(targetCanvas.context.bitmap.pixels,targetPixels,'Louder checkpoints must not recolor the target.');
  follow.checked=true;follow.dispatch('change');assert.equal(label.textContent,'25 / 200');
  slider.value=1;slider.dispatch('input');assert.equal(label.textContent,'10 / 200');
  assert.deepEqual(reconstructedCanvas.context.bitmap.pixels,quietPixels);
  assert.match(slider.attributes['aria-valuetext'],/step 10 of 200/);
  assert.match(reconstructedCanvas.attributes['aria-label'],/training step 10/);
  training.completeTraining();assert.match(status.textContent,/Fit complete/);assert.equal(label.textContent,'10 / 200');
  training.stopTraining();assert.match(status.textContent,/Fitting stopped/);assert.deepEqual(reconstructedCanvas.context.bitmap.pixels,quietPixels);
  training.clearReconstruction('Restarting…');assert(controls.hidden);assert(!reconstructedCanvas.context.bitmap);assert(targetCanvas.context.bitmap);
  assert(follow.checked);

  // Input powers can be released/reused after rasterization. History keeps its
  // own compact data, including when old checkpoints are evicted defensively.
  const disposable=computeSpectrogram(tone);
  training.addCheckpoint(disposable,{iteration:0,iterations:500});disposable.power.fill(0);
  training.addCheckpoint(quieter,{iteration:10,iterations:500});
  slider.value=0;slider.dispatch('input');assert.deepEqual(reconstructedCanvas.context.bitmap.pixels,initialPixels);
  slider.value=1;slider.dispatch('input');
  for(let i=2;i<=22;i++)training.addCheckpoint(target,{iteration:i*10,iterations:500});
  assert.equal(slider.max,21);assert.equal(label.textContent,'20 / 500');
  assert.deepEqual(reconstructedCanvas.context.bitmap.pixels,initialPixels,'Eviction must refresh the selected image along with its step label.');
  training.setTarget(changing,sampleRate);assert(controls.hidden);assert(!reconstructedCanvas.context.bitmap);
  training.dispose();assert.equal(Object.keys(slider.listeners).length,0);assert.equal(Object.keys(follow.listeners).length,0);
} finally {
  if (oldDocument === undefined) delete globalThis.document; else globalThis.document = oldDocument;
  if (oldWindow === undefined) delete globalThis.window; else globalThis.window = oldWindow;
  if (oldObserver === undefined) delete globalThis.ResizeObserver; else globalThis.ResizeObserver = oldObserver;
}
console.log(JSON.stringify({passed: true, frequencyPeakHz: 1000, halfAmplitudeDbDifference: levelDifference,
  identicalWaveformsIdenticalPixels: true, playbackGainRemoved: true, axesAligned: true, resizeUsesCachedRaster: true,
  fixedColorsAcrossTraining: true, compactHistory: true, scrubAndFollowLatest: true,
  cancellationRetainsPreviews: true, resetClearsHistory: true, evictionKeepsLabelAndImageAligned: true}));
