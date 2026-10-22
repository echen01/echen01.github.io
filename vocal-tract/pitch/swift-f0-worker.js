// Adapted from the MIT-licensed official SwiftF0 browser demo. See README.md.
import * as ort from './ort/ort.wasm.min.mjs';
import { SAMPLE_RATE, FRAME_PERIOD, MODEL_VERSION, CONFIDENCE_THRESHOLD, FMIN, FMAX } from './swift-f0.js';

// One thread works on ordinary static hosting, without cross-origin isolation.
ort.env.wasm.numThreads = 1;
ort.env.wasm.proxy = false;
ort.env.wasm.wasmPaths = {
  mjs: new URL('./ort/ort-wasm-simd-threaded.mjs', import.meta.url).href,
  wasm: new URL('./ort/ort-wasm-simd-threaded.wasm', import.meta.url).href,
};

const HOP = 256;
const LEFT_FRAMES = 11;
const LOOKAHEAD_FRAMES = 10;
const WINDOW_FRAMES = 1875; // 30 seconds, matching the Python library.
const SILENCE_PEAK = 1e-3;
let session = null;
let queue = Promise.resolve();
const canceled = new Set();
const active = new Set();

function stage(id, value) { self.postMessage({ id, type: 'stage', stage: value }); }
function progress(id, fraction) { self.postMessage({ id, type: 'progress', fraction }); }
function checkCanceled(id) {
  if (canceled.has(id)) throw new Error('Pitch estimation canceled.');
}

async function download(path) {
  const response = await fetch(new URL(path, import.meta.url));
  if (!response.ok) throw new Error(`The pitch estimator could not load (${response.status}). Please retry.`);
  return new Uint8Array(await response.arrayBuffer());
}

async function loadModel(id) {
  if (session) return;
  stage(id, 'loading');
  const [model, wasm] = await Promise.all([download('./model.onnx'), download('./ort/ort-wasm-simd-threaded.wasm')]);
  checkCanceled(id);
  stage(id, 'preparing');
  ort.env.wasm.wasmBinary = wasm;
  try {
    session = await ort.InferenceSession.create(model, { executionProviders: ['wasm'] });
  } finally {
    delete ort.env.wasm.wasmBinary;
  }
}

async function infer(samples) {
  const input = new ort.Tensor('float32', samples, [1, samples.length]);
  const fmin = new ort.Tensor('float32', Float32Array.of(FMIN), []);
  const fmax = new ort.Tensor('float32', Float32Array.of(FMAX), []);
  let output;
  try {
    output = await session.run({ audio: input, fmin, fmax });
    return {
      pitch: Float32Array.from(output.pitch.data),
      confidence: Float32Array.from(output.confidence.data),
    };
  } finally {
    input.dispose();
    fmin.dispose();
    fmax.dispose();
    if (output) for (const tensor of Object.values(output)) tensor.dispose();
  }
}

async function estimate(samples, id) {
  checkCanceled(id);
  await loadModel(id);
  checkCanceled(id);
  stage(id, 'analyzing');
  progress(id, 0);
  const n = Math.max(1, Math.floor(samples.length / HOP));
  const pitchHz = new Float32Array(n);
  const confidence = new Float32Array(n);
  // Include receptive-field context at chunk boundaries, so chunking does not
  // change the result compared with a single pass through the official model.
  for (let first = 0; first < n; first += WINDOW_FRAMES) {
    checkCanceled(id);
    const last = Math.min(n, first + WINDOW_FRAMES);
    const left = Math.max(0, first - LEFT_FRAMES);
    const end = last === n ? samples.length : (last + LOOKAHEAD_FRAMES) * HOP;
    const part = await infer(samples.subarray(left * HOP, end));
    const from = first - left;
    if (part.pitch.length < from + last - first || part.confidence.length !== part.pitch.length) {
      throw new Error('The pitch estimator returned an unexpected number of frames.');
    }
    pitchHz.set(part.pitch.subarray(from, from + last - first), first);
    confidence.set(part.confidence.subarray(from, from + last - first), first);
    progress(id, last / n);
  }
  checkCanceled(id);
  const times = new Float64Array(n);
  const voiced = new Uint8Array(n);
  const loudnessDb = new Float32Array(n);
  let previousPower = 0;
  for (let t = 0; t < n; t++) {
    let peak = 0;
    let power = 0;
    const end = Math.min(samples.length, (t + 1) * HOP);
    for (let j = t * HOP; j < end; j++) {
      peak = Math.max(peak, Math.abs(samples[j]));
      power += samples[j] * samples[j];
    }
    // Digital silence can otherwise produce spurious voiced predictions.
    if (peak < SILENCE_PEAK) confidence[t] = 0;
    if (!Number.isFinite(pitchHz[t]) || !Number.isFinite(confidence[t])) {
      throw new Error('The pitch estimator returned invalid values.');
    }
    times[t] = t * FRAME_PERIOD;
    voiced[t] = Number(confidence[t] >= CONFIDENCE_THRESHOLD);
    loudnessDb[t] = 20 * Math.log10(Math.max(Math.sqrt((previousPower + power) / 512), 1e-7));
    previousPower = power;
  }
  return { times, pitchHz, confidence, voiced, loudnessDb, sampleRate: SAMPLE_RATE, framePeriod: FRAME_PERIOD, modelVersion: MODEL_VERSION };
}

self.onmessage = ({ data }) => {
  if (data.type === 'cancel') {
    if (active.has(data.id)) canceled.add(data.id);
    return;
  }
  if (data.type !== 'estimate') return;
  active.add(data.id);
  // Serialize requests because an ONNX session owns reusable WASM buffers.
  queue = queue.then(async () => {
    try {
      const result = await estimate(data.samples, data.id);
      self.postMessage({ id: data.id, result }, [result.times.buffer, result.pitchHz.buffer, result.confidence.buffer, result.voiced.buffer, result.loudnessDb.buffer]);
    } catch (error) {
      self.postMessage({ id: data.id, error: error instanceof Error ? error.message : String(error) });
    } finally {
      canceled.delete(data.id);
      active.delete(data.id);
    }
  });
};
