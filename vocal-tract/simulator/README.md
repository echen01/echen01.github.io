# The browser acoustic experiment

This directory contains a small, reproducible browser demonstration of
the **stationary voiced** part of [diffTLM](https://github.com/echen01/diffTLM).
It fits a synthetic target transfer spectrum by running actual reverse-mode
backpropagation through a frequency-domain transmission-line model in the
visitor's browser. No fitting trajectory is recorded or downloaded.
The vowel demo uses WebGPU for transfer responses, reverse gradients, loss
reduction, gradient clipping, and Adam. The Rust/WebAssembly CPU implementation
with float64 arithmetic remains available through the numerical reference APIs
and validation tests. LF harmonic synthesis and browser audio playback run on
the host using the WebGPU simulator's complex response.

## Physics and scope

The reference is `difftlm/tract.py` at commit
`b05116e06afbd7a569f3b9ac1d03a52c9d18efed`. Air viscosity, wall mass/resistance/
stiffness, and lip radiation impedance follow that code in CGS units. The current
JAX convention has **32 area values**, of which the last is a lip boundary and the
first 31 are propagating segments of length `L/32`; the browser matches that
convention exactly. Areas are in cm²; length is in cm.

The three target shapes resample the Fant area data supplied in
`diffTLM/data/fant.py`. Their lengths are 17.5 cm for /ɑ/, 18 cm for /e/, and 20 cm for /u/.
Playback uses a periodic Liljencrants–Fant flow source with the reference
OQ=0.5, SQ=3, and TL=0.02, filtered by the same complex transfer function. A short
fade prevents clicks; playback peak is normalized for comfortable comparison.

The loss is mean squared natural-log transfer-magnitude error across 160
frequencies from 80 Hz to 5 kHz. The demo defaults to a spatial MFN neural field:
four learned sine filters multiply three 256-wide affine layers, followed by
one scalar output. Its 200,705 shared weights and biases predict a residual on
the starting tube's log-area profile. The hidden layers and filters use the
upstream FourierNet initialization, with input scale 32. An area scale of 0.1
multiplies the network output before adding it to the starting log-area profile:
`log(A) = clamp(log(A_start) + 0.1 * F(x), log(0.12), log(16))`.
The backward pass includes the same factor. The initial
profile is frozen and interpolated continuously in log area; zero output-layer
weights preserve it exactly at step zero. All 32 values, including the glottal
and lip boundaries, can change through the field. Areas stay within 0.12–16 cm².

The acoustic reverse pass supplies derivatives with respect to log area, which
backpropagate through the field to its parameters. Adam uses a global parameter
gradient-norm limit of 5 and a cosine learning-rate schedule from 0.005 to 0.0005
over 240 updates. Both CPU and WebGPU use this architecture, initialization,
objective, and schedule; WebGPU keeps the field, acoustic reverse pass, and
optimizer on the GPU. There is no extra smoothness penalty or coordinate jitter.
Both geometry and displayed loss are evaluated at the same updated parameters.

This experiment deliberately does not implement the paper's moving tract,
Reynolds-gated turbulence, glottal/noise inference, multiresolution mel-STFT
objective, or a time-varying neural field. It is not an implementation of the
full utterance reconstruction pipeline, and a good spectral fit does not imply
unique or anatomical geometry. These limitations are also stated on the page.

Compared with the older `vocal-wasm` implementation, this browser module follows
the latest boundary-area convention, exposes shape and tract length, and adds
reverse-mode differentiation and optimization. It is a separate, dependency-free
module; it does not modify the upstream repository.

## Build and verification

Requires Rust with the `wasm32-unknown-unknown` target, Node.js, and Python with
NumPy for an independent reference. No Rust packages are fetched.

```sh
sh build.sh
```

The deployed `difftlm.wasm` is built from `src/lib.rs`. The tests cover:

- Complex transfer-function parity with independent NumPy full-matrix products
  across 50 shape/frequency combinations, including DC and the area limits.
- All 32 reverse-mode log-area derivatives against centered finite differences,
  in both native Rust and the compiled WebAssembly binary.
- Zero loss/gradient for an identical target and finite responses at area limits.
- At least 80% spectral-loss reduction for each preset over 240 actual steps.
- Finite, bounded, click-faded audio and rejected invalid model inputs.

### Direct comparison with upstream JAX autodiff

The optional JAX suite calls the **unmodified upstream
`TransmissionLineSimulator.step_frame`**, rather than a second transcription of
the equations. It checks the exact source Git blob hashes at the commit above
before importing. An impulse input and zero turbulence noise recover the voiced
transfer function through `rfft(step_frame(...))`. `jax.value_and_grad`
differentiates the same mean squared natural-log magnitude loss with respect to
all 32 log areas. The identical target spectrum is passed to the shipped WASM.
JAX runs on CPU with float64 enabled to match the WASM's `f64` arithmetic.

There are 26 deterministic cases: vowel targets, eight seeded smooth shapes,
uniform areas at and near the bounds, and localized/broad constrictions. Tract
lengths span 12–22 cm; FFT sizes are 512, 1024, and 2048 at 16 kHz. The suite
compares 15,078 complex response values and 832 gradient coordinates. DC and
Nyquist are excluded because the upstream `irfft` projects those two bins onto
the real axis. This does not test turbulence, overlap-add synthesis, neural
fields, derivatives with respect to length or pitch, or default float32 behavior.

From this simulator directory:

```sh
python3.12 -m venv /tmp/difftlm-jax-validation
/tmp/difftlm-jax-validation/bin/python -m pip install -r tests/requirements-jax.txt
DIFFTLM_JAX_PYTHON=/tmp/difftlm-jax-validation/bin/python node tests/check-jax.mjs --report tests/jax-validation.json
```

The first run downloads only three pinned upstream source files into a temporary
directory. For an offline run, append `--upstream /path/to/diffTLM`; those source
files must match the pinned hashes. The report records dependency versions,
source identity, the tested WASM SHA-256, per-case errors, and pass/fail results.
Gradient scaled error is `abs(wasm - jax) / (1 + abs(jax))`; the suite also checks
the relative L2 error of each full gradient vector. This test is separate from
`build.sh` so routine browser builds do not install JAX.

The saved `tests/jax-validation.json` reports **26/26 cases passed** with JAX
0.12.0.1. Maximum complex-response relative error is `3.73e-13`, maximum
coordinatewise scaled gradient error is `1.46e-13`, and maximum gradient-vector
relative L2 error is `5.78e-14`. No simulator changes were needed. These values
apply to the exact WASM and case specification identified by hashes in the report.

## WebGPU browser API

```js
import {createWebGpuSimulator} from './webgpu.js';
import {PRESETS, initialAreas} from './index.js';
const sim = await createWebGpuSimulator();
const target = PRESETS[0];
const spectrum = await sim.response(target.areas, target.length);
const fitter = await sim.createMfnFitter({
  areas: initialAreas(), targetAreas: target.areas, length: target.length,
  iterations: 240, learningRate: .005,
});
await fitter.step(4); // 1–8 updates per bounded submission
const snapshot = await fitter.snapshot();
fitter.dispose();
const audio = await sim.synthesize(snapshot.areas, target.length, 120, .8);
sim.dispose();
```

The WebGPU MFN fitter uses the same 160-frequency objective, area bounds, clipping,
and learning-rate schedule as `createVowelMfnFitter` in `mfn-field.js` on CPU.
Its parameters, gradients, Adam moments,
and complete loss history stay on the GPU between snapshots. Float32 rounding
can produce different later optimization trajectories in this nonconvex problem.
For both backends, the interactive worker displays every real optimization step, with a minimum
25 ms pause between snapshots: a 240-step fit takes about six seconds. It leaves
the GPU idle between updates and accepts cancellation during those pauses. This
presentation pacing does not change the objective or Adam updates; the direct
fitter API remains unpaced. Cancellation explicitly frees its GPU resources.
WebGPU requires a supported browser and a secure context (HTTPS or localhost).
If it is unavailable or the device is lost, the demo reports the error. There
is no CPU option or automatic CPU fallback in the vowel fitting UI.

Open `tests/webgpu.html` through the local server and click **Run validation**.
It checks six transfer/gradient cases against the float64 reference, early Adam
updates, complete 240-step fits for all three vowels, nine LF audio cases, the
existing gain-invariant speech objective, cancellation, disposal, unsupported
WebGPU, and device loss. The latest local results are in
`tests/webgpu-validation.json`; this is a numerical check, not a speed benchmark.
That suite retains the independent-area `createFitter` API as a numerical baseline.
The main demo uses the MFN fitter instead. The RFF implementation remains
available through `createRffFitter` and `createVowelRffFitter` for comparisons.

For the default neural field, run `node tests/mfn.test.mjs` and open
`tests/mfn.html`. The Node test checks chained acoustic/MFN derivatives by finite
differences across every affine and learned-filter block, initial geometry,
area limits, and all three 240-step CPU fits. The browser test checks all 200,705
parameter derivatives against the CPU implementation at steps 0, 10, and 240
for each vowel, early optimizer trajectories, complete GPU fits, and
completion/cancellation of the default CPU and WebGPU workers. Corresponding
RFF checks remain in `tests/rff.test.mjs` and `tests/rff.html`.

## CPU reference API

```js
import {createSimulator, PRESETS, initialAreas, FREQUENCIES} from './index.js';
import {createVowelMfnFitter} from './mfn-field.js';
const sim = await createSimulator();
const target = PRESETS[0];
const spectrum = sim.response(target.areas, target.length);
const {loss, gradient} = sim.lossAndGradient(initialAreas(), spectrum.magnitudeDb, target.length);
const {samples, sampleRate} = sim.synthesize(target.areas, target.length, 120, .8);
const fitter = await createVowelMfnFitter(sim, {
  areas: initialAreas(), targetAreas: target.areas, length: target.length,
  iterations: 240, learningRate: .005,
});
fitter.step();
const snapshot = fitter.snapshot();
```

`response` returns `frequencies`, `real`, `imag`, and `magnitudeDb` arrays. dB
values retain the source CGS scale; if offset for a chart, use one common offset
for both curves. `lossAndGradient` returns the true loss and its derivative with
respect to each log area.

Use `fit-worker.js` as a module Worker. Send `{type:'start', backend:'cpu', areas,
targetAreas, length, iterations:240, learningRate:.005}`. `backend` accepts `cpu`
(the default) or `webgpu`. The field defaults to MFN; pass `fieldType:'rff'` to
compare the previous RFF field. Each `progress` and final `done` message includes
`backend`, `parameterization:'mfn'` (or `'rff'`), `field` metadata, `iteration`, `areas`, `loss`, `gradient`, `gradientNorm`,
`magnitudeDb`, `targetDb`, `history`, and `elapsedMs`. `stop` cancels between
updates; a `stopped` message confirms resource cleanup.
Terminating the Worker also cancels immediately. Failures
produce `{type:'error', message}`. All computations are local.

The model and source data are distributed under the accompanying MIT license.
