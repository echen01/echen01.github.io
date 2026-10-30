// SwiftF0's official model runs entirely in a lazily created local Web Worker.
export const SAMPLE_RATE = 16000;
export const FRAME_PERIOD = 0.016;
export const MODEL_VERSION = '0.3.0';
export const CONFIDENCE_THRESHOLD = 0.5;
export const FMIN = 46.875;
export const FMAX = 2093.75;

const TIMEOUT_MS = 180000;
const pending = new Map();
let worker = null;
let nextId = 0;

function failWorker(error) {
  worker?.terminate();
  worker = null;
  for (const request of pending.values()) request.reject(error);
  pending.clear();
}

function getWorker() {
  if (worker) return worker;
  worker = new Worker(new URL('./swift-f0-worker.js', import.meta.url), { type: 'module' });
  worker.onmessage = ({ data }) => {
    const request = pending.get(data.id);
    if (!request) return;
    if (data.type === 'stage') {
      request.onStage?.(data.stage);
    } else if (data.type === 'progress') {
      request.onProgress?.(data.fraction);
    } else if (data.error) {
      request.reject(new Error(data.error));
    } else {
      request.resolve(data.result);
    }
  };
  worker.onerror = (event) => {
    failWorker(new Error(event.message || 'The pitch estimator could not start. Please retry.'));
  };
  worker.onmessageerror = () => failWorker(new Error('The pitch estimator returned an unreadable result. Please retry.'));
  return worker;
}

/**
 * Estimate monophonic pitch from finite, mono Float32 PCM sampled at 16 kHz.
 * This function preserves the caller's samples. It does not normalize audio.
 * Options: onStage('loading'|'preparing'|'analyzing'), onProgress(0..1), signal,
 * fmin/fmax within the model's supported range (defaults FMIN/FMAX).
 * Returns {times, pitchHz, confidence, voiced, loudnessDb, sampleRate,
 * framePeriod, modelVersion}. Pitch values at unvoiced frames are not reliable;
 * use the voiced mask (confidence >= 0.5, after SwiftF0's silence gate).
 */
export function estimatePitch(samples, { onStage, onProgress, signal, fmin = FMIN, fmax = FMAX } = {}) {
  if (!(samples instanceof Float32Array) || !samples.length) {
    return Promise.reject(new TypeError('Pitch estimation needs nonempty mono Float32 audio at 16 kHz.'));
  }
  if (!samples.every(Number.isFinite)) {
    return Promise.reject(new TypeError('The audio contains invalid sample values.'));
  }
  if (!Number.isFinite(fmin) || !Number.isFinite(fmax) || fmin < FMIN || fmax > FMAX || fmin >= fmax) return Promise.reject(new Error('Pitch bounds must lie inside SwiftF0’s supported frequency range.'));
  if (signal?.aborted) return Promise.reject(new DOMException('Pitch estimation canceled.', 'AbortError'));

  return new Promise((resolve, reject) => {
    const id = ++nextId;
    const cleanup = () => {
      clearTimeout(timer);
      signal?.removeEventListener('abort', abort);
      pending.delete(id);
    };
    const abort = () => {
      worker?.postMessage({ type: 'cancel', id });
      cleanup();
      reject(new DOMException('Pitch estimation canceled.', 'AbortError'));
    };
    const timer = setTimeout(() => {
      failWorker(new Error('Pitch estimation timed out. Please retry with a shorter recording.'));
    }, TIMEOUT_MS);
    pending.set(id, {
      onStage,
      onProgress,
      resolve: (result) => { cleanup(); resolve(result); },
      reject: (error) => { cleanup(); reject(error); },
    });
    signal?.addEventListener('abort', abort, { once: true });
    try {
      const copy = samples.slice();
      getWorker().postMessage({ type: 'estimate', id, samples: copy, fmin, fmax }, [copy.buffer]);
    } catch (error) {
      cleanup();
      reject(error);
    }
  });
}

// Release the cached model, for example when disposing a single-page view.
export function disposePitchEstimator() {
  failWorker(new DOMException('Pitch estimator closed.', 'AbortError'));
}
