# Browser pitch estimation

`swift-f0.js` lazily creates a module worker containing the official **SwiftF0 0.3.0** model and the official demo's reduced **ONNX Runtime Web 1.29.0** WASM build. Audio never leaves the browser. Runtime/model requests are same-origin and relative to this directory; there is no CDN dependency. One WASM thread works on GitHub Pages without COOP/COEP headers. The neural model is not differentiated through; its estimated F0 is an input to the separate tract simulator.

```js
import { estimatePitch } from './pitch/swift-f0.js';
const result = await estimatePitch(monoAudioAt16k, {
  onStage: stage => console.log(stage), // loading, preparing, analyzing
  onProgress: fraction => console.log(fraction), // 0..1 during inference
  signal: abortController.signal, // optional
});
// result.times: Float64Array, seconds at the start of each 16 ms frame
// result.pitchHz, result.confidence, result.loudnessDb: Float32Array
// result.voiced: Uint8Array, 1 when confidence >= 0.5
// result.sampleRate = 16000; result.framePeriod = 0.016
// result.modelVersion = '0.3.0'
```

Supply finite mono `Float32Array` PCM at **16,000 Hz**, conventionally in [-1, 1]. The caller decodes/resamples; the module preserves input and does not change recording gain. SwiftF0 supports 46.875–2093.75 Hz; values from unvoiced frames should not be used as pitch estimates. Like the official Python and web implementations, a 256-sample frame with peak amplitude below 0.001 receives confidence zero. Loudness uses the previous and current hop (512 samples), floored at -140 dBFS. Clips shorter than one hop yield one frame. Analysis uses 30-second windows with 11 preceding and 10 following context frames, matching the Python implementation's bounded-memory analysis. Cancellation rejects immediately and queued/in-flight inference is discarded at the next window boundary.

The page currently summarizes reliable voiced pitch to drive its stationary vowel source. That summary is separate from SwiftF0's full per-frame contour; it does not turn the stationary inverse demo into full speech fitting.

## Upstream and licenses

- [SwiftF0 Python/model repository](https://github.com/lars76/swift-f0/tree/d1ba77fe5331310b778ebe126b6cb3656d41d24f), version 0.3.0.
- [Official browser demo source](https://github.com/swift-f0/swift-f0.github.io/tree/1ffc621afa141bc6a14a3677bddf38896b43e309), pinned commit `1ffc621afa141bc6a14a3677bddf38896b43e309`. `model.onnx` and every file in `ort/` are copied unchanged from its `src/` directory. The worker adapts its `src/onnx-worker.js`, using the Python library's 30-second window length.
- SwiftF0 and the adapted worker: MIT, see `LICENSE.swift-f0.txt` (Lars Nieradzik).
- ONNX Runtime Web: MIT, see `ort/LICENSE.txt` and `ort/ThirdPartyNotices.txt`.
- Exact source URLs, byte lengths and SHA-256 hashes: `vendor-manifest.json`.
- Reference: Lars Nieradzik, [SwiftF0: Fast and Accurate Monophonic Pitch Detection](https://arxiv.org/abs/2508.18440), 2025. The paper describes the original model; version 0.3.0 uses the newer network documented in the [official article](https://swift-f0.github.io/how/).

## Validation

Serve the repository and open `vocal-tract/pitch/test.html`, then press **Run pitch tests**. The browser suite checks harmonic tones at three known pitches, digital silence, short audio, real bundled speech, invalid input, and cancellation. It loads the real ONNX model and WASM files; there are no mocked predictions.

For automated checks, run `node vocal-tract/pitch/test-browser.mjs` with Playwright installed and Chrome available. Alternatively supply the absolute path to Playwright's `index.mjs` as the first argument; `CHROMIUM_PATH` can select a browser executable. The runner starts a temporary local HTTP server with no isolation headers and additionally verifies vendor hashes, lazy downloads, cached inference, and the absence of external requests.
