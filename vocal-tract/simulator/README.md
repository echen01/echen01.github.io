# The browser acoustic experiment

This directory contains a small, reproducible Rust/WebAssembly demonstration of
the **stationary voiced** part of [diffTLM](https://github.com/echen01/diffTLM).
It fits a synthetic target transfer spectrum by running actual reverse-mode
backpropagation through a frequency-domain transmission-line model in the
visitor's browser. No fitting trajectory is recorded or downloaded.

## Physics and scope

The reference is `difftlm/tract.py` at commit
`b05116e06afbd7a569f3b9ac1d03a52c9d18efed`. Air viscosity, wall mass/resistance/
stiffness, and lip radiation impedance follow that code in CGS units. The current
JAX convention has **32 area values**, of which the last is a lip boundary and the
first 31 are propagating segments of length `L/32`; the browser matches that
convention exactly. Areas are in cm²; length is in cm.

The three target shapes resample the Fant area data supplied in
`diffTLM/data/fant.py`. Their lengths are 17.5 cm for /ɑ/ and /i/, and 20 cm for /u/.
Playback uses a periodic Liljencrants–Fant flow source with the reference
OQ=0.5, SQ=3, and TL=0.02, filtered by the same complex transfer function. A short
fade prevents clicks; playback peak is normalized for comfortable comparison.

The loss is mean squared natural-log transfer-magnitude error across 160
frequencies from 80 Hz to 5 kHz. The optimizer is Adam on 32 log-area parameters;
areas remain between 0.12 and 16 cm². Both the geometry and displayed loss are
computed at every step. The derivative is with respect to log area.

This experiment deliberately does not implement the paper's moving tract,
Reynolds-gated turbulence, glottal/noise inference, multiresolution mel-STFT
objective, or neural field parameterization. It is not an implementation of the
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

## Browser API

```js
import {createSimulator, PRESETS, initialAreas, FREQUENCIES} from './index.js';
const sim = await createSimulator();
const target = PRESETS[0];
const spectrum = sim.response(target.areas, target.length);
const {loss, gradient} = sim.lossAndGradient(initialAreas(), spectrum.magnitudeDb, target.length);
const {samples, sampleRate} = sim.synthesize(target.areas, target.length, 120, .8);
```

`response` returns `frequencies`, `real`, `imag`, and `magnitudeDb` arrays. dB
values retain the source CGS scale; if offset for a chart, use one common offset
for both curves. `lossAndGradient` returns the true loss and its derivative with
respect to each log area.

Use `fit-worker.js` as a module Worker. Send `{type:'start', areas, targetAreas,
length, iterations:240, learningRate:.035}`. Each `progress` and final `done`
message includes `iteration`, `areas`, `loss`, `gradient`, `gradientNorm`,
`magnitudeDb`, `targetDb`, `history`, and `elapsedMs`. `stop` cancels at the next
iteration boundary; terminating the Worker also cancels immediately. Failures
produce `{type:'error', message}`. All computations are local.

The model and source data are distributed under the accompanying MIT license.
