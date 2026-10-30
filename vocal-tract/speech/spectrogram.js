import {fft} from './faithful-fft.js';

export const SPECTROGRAM_MIN_DB = -80;
export const SPECTROGRAM_MAX_DB = 20;
const MAX_CHECKPOINTS = 22;
const COLORS = [[22, 26, 51], [65, 75, 135], [181, 95, 86], [232, 164, 85], [244, 237, 176]];

/** Display-only STFT. Each column is centered in an 8 ms interval; the 32 ms
 * periodic Hann window is zero-padded at the recording boundaries. This does
 * not reuse or alter the simulator's analysis windows or training objective.
 */
export function computeSpectrogram(samples, {sampleRate = 16000, fftSize = 512, hopSize = 128, amplitudeScale = 1} = {}) {
  if (!samples?.length || !samples.every(Number.isFinite)) throw new Error('Spectrogram audio must contain finite samples.');
  if (!Number.isFinite(sampleRate) || sampleRate <= 0 || !Number.isInteger(fftSize) || fftSize < 2 || (fftSize & (fftSize - 1))) throw new Error('Invalid spectrogram sample rate or FFT size.');
  if (!Number.isInteger(hopSize) || hopSize < 1 || hopSize > fftSize || !Number.isFinite(amplitudeScale) || amplitudeScale <= 0) throw new Error('Invalid spectrogram hop or amplitude scale.');
  const frames = Math.ceil(samples.length / hopSize), bins = fftSize / 2 + 1;
  const power = new Float32Array(frames * bins), window = new Float64Array(fftSize);
  const real = new Float64Array(fftSize), imag = new Float64Array(fftSize);
  let windowSum = 0, peakPower = 0;
  for (let i = 0; i < fftSize; i++) { window[i] = .5 - .5 * Math.cos(2 * Math.PI * i / fftSize); windowSum += window[i]; }
  for (let frame = 0; frame < frames; frame++) {
    const start = frame * hopSize + Math.floor(hopSize / 2) - fftSize / 2;
    for (let i = 0; i < fftSize; i++) real[i] = (samples[start + i] ?? 0) * window[i] * amplitudeScale;
    imag.fill(0); fft(real, imag);
    for (let bin = 0; bin < bins; bin++) {
      const scale = (bin === 0 || bin === bins - 1 ? 1 : 2) / windowSum;
      const value = (real[bin] ** 2 + imag[bin] ** 2) * scale ** 2;
      power[frame * bins + bin] = value;
      peakPower = Math.max(peakPower, power[frame * bins + bin]);
    }
  }
  return {power, peakPower, frames, bins, sampleRate, fftSize, hopSize, duration: samples.length / sampleRate};
}

export function sharedSpectrogramScale(...spectrograms) {
  const peak = Math.max(0, ...spectrograms.filter(Boolean).map(value => value.peakPower));
  return {referencePower: peak || 1, minimumDb: SPECTROGRAM_MIN_DB, maximumDb: 0};
}

export function spectralDecibels(power, scale) {
  return Math.max(scale.minimumDb, Math.min(scale.maximumDb, 10 * Math.log10(Math.max(Number.MIN_VALUE, power) / scale.referencePower)));
}

export function spectrogramColor(db) {
  const position = Math.max(0, Math.min(1, (db - SPECTROGRAM_MIN_DB) / -SPECTROGRAM_MIN_DB)) * (COLORS.length - 1);
  const low = Math.min(COLORS.length - 2, Math.floor(position)), fraction = position - low;
  return COLORS[low].map((value, channel) => Math.round(value + (COLORS[low + 1][channel] - value) * fraction));
}
const COLOR_LOOKUP = Array.from({length: 256}, (_, i) => spectrogramColor(SPECTROGRAM_MIN_DB * (1 - i / 255)));

/** Fixed target-relative colors make checkpoints comparable. Store one byte per
 * time/frequency bin rather than retaining float powers and RGBA canvases for
 * every checkpoint. Only the selected reconstruction is expanded to a raster.
 */
export function quantizeSpectrogram(spectrum, scale) {
  return Uint8Array.from(spectrum.power, power => Math.round(255 *
    (spectralDecibels(power, scale) - scale.minimumDb) / (scale.maximumDb - scale.minimumDb)));
}

/** Cached display panels and bounded training history. Resizing and playback
 * never compute an FFT. Worker previews arrive as display-only STFT powers.
 */
export function createSpectrogramComparison({targetCanvas, reconstructedCanvas, status, trainingControls, stepSlider, stepLabel, followLatest}) {
  let target = null, reconstructed = null, targetRaster = null, reconstructedRaster = null;
  let pending = 'Start fitting to see the reconstruction evolve.', scale = null;
  let checkpoints = [], selected = -1, following = true, trainingState = 'Fitting';
  const metadata = ({power, ...rest}) => rest;
  function raster(spectrum, pixels, reuse) {
    if (!spectrum) return null;
    const canvas = reuse || document.createElement('canvas'); canvas.width = spectrum.frames; canvas.height = spectrum.bins;
    const context = canvas.getContext('2d'), image = context.createImageData(canvas.width, canvas.height);
    for (let frame = 0; frame < spectrum.frames; frame++) for (let bin = 0; bin < spectrum.bins; bin++) {
      const color = COLOR_LOOKUP[pixels[frame * spectrum.bins + bin]];
      const offset = ((spectrum.bins - 1 - bin) * spectrum.frames + frame) * 4;
      image.data.set(color, offset); image.data[offset + 3] = 255;
    }
    context.putImageData(image, 0, 0); return canvas;
  }
  function paint(canvas, spectrum, bitmap, title, empty) {
    const width = canvas.clientWidth || 760, height = canvas.clientHeight || 200, pixelRatio = Math.min(2, window.devicePixelRatio || 1);
    canvas.width = Math.round(width * pixelRatio); canvas.height = Math.round(height * pixelRatio);
    const context = canvas.getContext('2d'); context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
    const left = 41, top = 18, right = 10, bottom = 31, plotWidth = width - left - right, plotHeight = height - top - bottom;
    const duration = Math.max(target?.duration || 0, reconstructed?.duration || 0), nyquist = (target?.sampleRate || 16000) / 2;
    context.fillStyle = bitmap ? 'rgb(22,26,51)' : '#f3f5f1'; context.fillRect(left, top, plotWidth, plotHeight);
    if (bitmap) {
      context.imageSmoothingEnabled = false;
      context.drawImage(bitmap, left, top, plotWidth * spectrum.duration / duration, plotHeight);
    } else {
      context.fillStyle = '#626b6b'; context.font = '11px system-ui, sans-serif'; context.textAlign = 'center';
      context.fillText(empty, left + plotWidth / 2, top + plotHeight / 2, plotWidth - 16);
    }
    context.font = '10px system-ui, sans-serif'; context.fillStyle = '#626b6b';
    context.textAlign = 'right';
    for (let n = 0; n <= 4; n++) {
      const y = top + plotHeight * (1 - n / 4), frequency = nyquist * n / 4;
      context.fillText(String(frequency / 1000), left - 8, y + 3);
    }
    context.textAlign = 'left'; context.fillText('kHz', 4, 10);
    if (duration) {
      for (let n = 0; n <= 4; n++) {
        const x = left + plotWidth * n / 4;
        context.textAlign = n === 0 ? 'left' : n === 4 ? 'right' : 'center';
        context.fillText(`${(duration * n / 4).toFixed(duration < 1 ? 2 : 1)}${n === 4 ? ' s' : ''}`, x, height - 9);
      }
    } else { context.textAlign = 'right'; context.fillText('Time · s', width - right, height - 9); }
    canvas.setAttribute('aria-label', spectrum
      ? `${title}: ${duration.toFixed(2)} seconds, 0 to ${nyquist / 1000} kilohertz. Fixed color scale from −80 to +20 decibels relative to the target peak.`
      : `${title}: ${empty}`);
  }
  function draw() {
    paint(targetCanvas, target, targetRaster, 'Target spectrogram', 'The target appears when you start fitting.');
    const checkpoint = checkpoints[selected];
    paint(reconstructedCanvas, reconstructed, reconstructedRaster,
      checkpoint ? `Reconstruction spectrogram at training step ${checkpoint.iteration}` : 'Reconstruction spectrogram', pending);
  }
  function updateControls() {
    const checkpoint = checkpoints[selected];
    if (trainingControls) trainingControls.hidden = !checkpoint;
    if (stepSlider) {
      stepSlider.min = 0; stepSlider.max = Math.max(0, checkpoints.length - 1); stepSlider.value = Math.max(0, selected);
      stepSlider.disabled = checkpoints.length < 2;
      stepSlider.setAttribute('aria-valuetext', checkpoint ? `Training step ${checkpoint.iteration} of ${checkpoint.iterations}` : 'No training previews yet');
    }
    if (stepLabel) stepLabel.textContent = checkpoint ? `${checkpoint.iteration} / ${checkpoint.iterations}` : '—';
    if (followLatest) followLatest.checked = following;
    status.textContent = checkpoint ? `${trainingState} · ${checkpoints.length} saved previews`
      : reconstructed ? 'Fixed time, frequency and level scales' : target ? 'Target ready · reconstruction pending' : 'Awaiting audio';
  }
  function select(index) {
    selected = Math.max(0, Math.min(checkpoints.length - 1, index));
    const checkpoint = checkpoints[selected];
    if (checkpoint) {
      reconstructed = checkpoint.spectrum;
      reconstructedRaster = raster(reconstructed, checkpoint.pixels, reconstructedRaster);
    }
    updateControls(); draw();
  }
  function clearHistory(message = 'Start fitting to see the reconstruction evolve.') {
    reconstructed = null; reconstructedRaster = null; checkpoints = []; selected = -1;
    following = true; trainingState = 'Fitting'; pending = message; updateControls(); draw();
  }
  function addSpectrum(spectrum, {iteration, iterations} = {}) {
    if (!target || spectrum.sampleRate !== target.sampleRate) throw new Error('Spectrogram sample rates must match the target.');
    if (spectrum.power.length !== spectrum.frames * spectrum.bins) throw new Error('Invalid spectrogram dimensions.');
    const pixels = quantizeSpectrogram(spectrum, scale), info = metadata(spectrum);
    if (!Number.isInteger(iteration)) {
      reconstructed = info; reconstructedRaster = raster(info, pixels, reconstructedRaster); updateControls(); draw(); return;
    }
    const checkpoint = {iteration, iterations, spectrum: info, pixels};
    let selectionEvicted = false;
    const existing = checkpoints.findIndex(entry => entry.iteration === iteration);
    if (existing >= 0) checkpoints[existing] = checkpoint;
    else {
      checkpoints.push(checkpoint);
      if (checkpoints.length > MAX_CHECKPOINTS) {
        selectionEvicted = selected === 1;
        checkpoints.splice(1, 1); selected = selected > 1 ? selected - 1 : selected;
      }
    }
    if (following) select(checkpoints.length - 1);
    else if (selected === existing || selectionEvicted) select(selected);
    else updateControls();
  }
  const scrub = () => { following = false; select(Math.round(Number(stepSlider.value))); };
  const follow = () => { following = followLatest.checked; if (following) select(checkpoints.length - 1); else updateControls(); };
  stepSlider?.addEventListener('input', scrub); followLatest?.addEventListener('change', follow);
  const observer = typeof ResizeObserver === 'function' ? new ResizeObserver(draw) : null;
  observer?.observe(targetCanvas); observer?.observe(reconstructedCanvas);
  return {
    reset() { target = null; targetRaster = null; scale = null; clearHistory(); },
    setTarget(samples, sampleRate) {
      const spectrum = computeSpectrogram(samples, {sampleRate});
      scale = {...sharedSpectrogramScale(spectrum), maximumDb: SPECTROGRAM_MAX_DB};
      target = metadata(spectrum); targetRaster = raster(spectrum, quantizeSpectrogram(spectrum, scale)); clearHistory();
    },
    addCheckpoint(spectrum, options) { addSpectrum(spectrum, options); },
    setReconstruction(samples, {sampleRate, normalizationGain = 1}) {
      addSpectrum(computeSpectrogram(samples, {sampleRate, amplitudeScale: 1 / normalizationGain}));
    },
    completeTraining() { trainingState = 'Fit complete'; updateControls(); },
    stopTraining() { trainingState = 'Fitting stopped'; if (!reconstructed) pending = 'Fitting stopped before the first preview.'; updateControls(); draw(); },
    clearReconstruction(message) { clearHistory(message); },
    dispose() { observer?.disconnect(); stepSlider?.removeEventListener('input', scrub); followLatest?.removeEventListener('change', follow); checkpoints = []; },
  };
}
