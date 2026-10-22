# Whole-clip browser speech experiment

This module fits a moving vocal-tract area function to every audible frame of a
recording, using genuine reverse-mode acoustic gradients from the validated
Rust/WASM simulator. It then generates a new waveform from the fitted geometry.
Computation and user audio stay in the browser.

## Model and objective

Input is mono 16 kHz PCM (0.12–10 seconds) and the complete SwiftF0 pitch/voicing
contour. Analysis uses centered 64 ms Hann windows at a 20 ms hop. Each frame has
32 positive area values, including the separate lip boundary used by current
diffTLM. Tract length is fixed during a fit.

For voiced frames, the target features are harmonic-band magnitudes divided by
the harmonic coefficients of the fixed Liljencrants–Fant source (OQ=.5, SQ=3,
TL=.02). At most 36 harmonics through 5 kHz are used. For unvoiced frames, 28
smoothed bands approximate excitation by broadband noise at the glottis. Quiet
frames are excluded from the data term but remain connected by smoothness.

For each frame, an additive log gain is solved analytically. The data loss is
the mean squared natural-log magnitude error after removing that gain. The
gradient of the optimal gain cancels because the residuals have zero mean;
the existing WASM area-gradient routine therefore gives the exact derivative
of this objective. Spatial and temporal squared differences of log areas are
added with analytic gradients. Adam jointly updates all frame areas, projected
to 0.12–16 cm². Every reported loss is re-evaluated at the returned trajectory.

This is a **harmonic / spectral-envelope approximation**, not a port of the
paper's complete training pipeline. It does not implement localized
Reynolds-gated turbulence, neural fields, a multi-resolution waveform-STFT loss,
or waveform phase matching. Noise at the glottis is a limited model for
consonants, and a good spectral fit does not establish anatomical correctness.
The spectrum optimized at each frame approximates the output of the final
windowed waveform synthesis; the two objectives are not identical.

## Resynthesis

The source is newly generated: a continuous-phase LF pulse follows the full
interpolated pitch contour, with deterministic broadband noise during unvoiced
regions. The moving tract filters this source with the complex transfer
function at FFT frequencies. Hann overlap-add combines the frames, with squared
window normalization. The recording supplies only the observed frame RMS as a
nuisance gain. No original samples or phase are inserted into the excitation.
One global attenuation is applied only if needed to avoid output clipping.

## Worker API

Create `fit-worker.js` as a module Worker. Send:

```js
worker.postMessage({
  type: 'start', samples, sampleRate: 16000,
  pitch: {times, pitchHz, voiced, confidence},
  length: 17.5, iterations: 32,
});
```

`stage` messages report analysis, fitting, and synthesis. `progress` messages
include `iteration`, `loss`/`totalLoss`, `dataLoss`, spatial and temporal losses,
`initialLoss`, `history`, `elapsedMs`, flattened frame-major `areas` (Float64),
`frames`, `frameTimes`, `framePitchHz`, `frameVoiced`, `frameActive`, and
`activeFrames`. Areas are measured at the same state as the reported loss.
`done` includes these fields plus generated `samples` (Float32), `sampleRate`,
`normalizationGain`, and model/scope metadata. Errors return `{type:'error',
message}`. Send `stop` or terminate the Worker to cancel.

The pure functions in `engine.js` allow independent testing without a DOM.
`dsp.js` contains the radix-2 FFT, LF source, and overlap-add synthesis.

## Verification

```sh
node vocal-tract/speech/test.mjs
node vocal-tract/speech/test-recording.mjs /path/pitch.json /path/audio.f32le
```

The first test verifies gain invariance, all 32 gain-eliminated area derivatives
against finite differences, 96 spatial/temporal derivatives, FFT round trips,
voiced/unvoiced trajectory fitting, loss consistency, duration, finite bounded
audio, silence, and cancellation. Negating the recording leaves the generated
waveform unchanged, demonstrating that recording phase is not copied.

The second test requires a real recording and its actual SwiftF0 contour, then
checks improvement, complete trajectory coverage, and finite resynthesis. It
does not substitute a constant or fabricated contour when a fixture is absent.

Physics, LF equations, and attribution follow the accompanying
[`simulator` module](../simulator/README.md) and its [MIT license](../simulator/LICENSE).
