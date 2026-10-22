import {estimatePitch} from './pitch/swift-f0.js';

const $ = id => document.getElementById(id);
const MAX_BYTES = 20 * 1024 * 1024;
const MAX_SECONDS = 30;
const NS = 'http://www.w3.org/2000/svg';
function svgNode(tag, attributes, text) {
  const node = document.createElementNS(NS, tag);
  for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, value);
  if (text !== undefined) node.textContent = text;
  return node;
}
function median(values) {
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}
function drawPitch(result, duration) {
  const svg = $('pitch-plot');
  // Keep the accessible title and description while replacing plot marks.
  for (const node of [...svg.children]) if (!['title', 'desc'].includes(node.tagName)) node.remove();
  const pitches = Array.from(result.pitchHz).filter((value, i) => result.voiced[i] && Number.isFinite(value));
  const low = pitches.length ? Math.max(0, Math.floor((Math.min(...pitches) - 15) / 25) * 25) : 50;
  const high = pitches.length ? Math.max(low + 100, Math.ceil((Math.max(...pitches) + 15) / 25) * 25) : 300;
  const x = time => 53 + time / duration * 680;
  const y = pitch => 142 - (pitch - low) / (high - low) * 122;
  for (let i = 0; i <= 2; i++) {
    const pitch = low + (high - low) * i / 2;
    svg.append(svgNode('line', {x1: 53, x2: 733, y1: y(pitch), y2: y(pitch), class: 'plot-grid'}));
    svg.append(svgNode('text', {x: 43, y: y(pitch) + 4, 'text-anchor': 'end', class: 'plot-axis'}, Math.round(pitch)));
  }
  let path = '', previous = false;
  for (let i = 0; i < result.pitchHz.length; i++) {
    const voiced = result.voiced[i] && Number.isFinite(result.pitchHz[i]);
    if (voiced) {
      path += `${previous ? 'L' : 'M'}${x(result.times[i]).toFixed(2)},${y(result.pitchHz[i]).toFixed(2)} `;
      svg.append(svgNode('circle', {cx: x(result.times[i]), cy: y(result.pitchHz[i]), r: 2.2, fill: '#146f68', opacity: .35 + .65 * result.confidence[i]}));
    }
    previous = voiced;
  }
  svg.append(svgNode('path', {d: path, class: 'plot-current', 'stroke-width': 1.7}));
  svg.append(svgNode('text', {x: 53, y: 177, class: 'plot-axis'}, 'F₀ (Hz) · 0 s'));
  svg.append(svgNode('text', {x: 733, y: 177, class: 'plot-axis', 'text-anchor': 'end'}, `${duration.toFixed(1)} s`));
  if (!pitches.length) svg.append(svgNode('text', {x: 390, y: 88, class: 'plot-empty', 'text-anchor': 'middle'}, 'No confident voiced frames'));
}

async function decodeMono(blob) {
  if (blob.size > MAX_BYTES) throw new Error('Choose an audio file smaller than 20 MB.');
  const OfflineContext = window.OfflineAudioContext || window.webkitOfflineAudioContext;
  if (!OfflineContext) throw new Error('This browser does not support audio decoding. Try a recent browser.');
  // decodeAudioData resamples to its context rate; averaging channels preserves level.
  const decoder = new OfflineContext(1, 1, 16000);
  let audio;
  try { audio = await decoder.decodeAudioData(await blob.arrayBuffer()); }
  catch { throw new Error('This audio format could not be decoded. Try a WAV, MP3, or M4A file.'); }
  if (audio.duration > MAX_SECONDS) throw new Error('Choose a clip of 30 seconds or less.');
  if (audio.duration < .1) throw new Error('Choose a recording at least 0.1 seconds long.');
  const samples = new Float32Array(audio.length);
  for (let channel = 0; channel < audio.numberOfChannels; channel++) {
    const source = audio.getChannelData(channel);
    for (let i = 0; i < samples.length; i++) samples[i] += source[i] / audio.numberOfChannels;
  }
  return {samples, duration: audio.duration};
}

export function setupPitchEstimator({onPitch}) {
  let active = null, previewUrl = null;
  const status = message => { $('pitch-status').textContent = message; };
  function setBusy(busy) {
    $('pitch-example').disabled = busy;
    $('pitch-file').disabled = busy;
    $('pitch-cancel').hidden = !busy;
    $('pitch-estimator').setAttribute('aria-busy', String(busy));
  }
  function cancel(message = 'Pitch analysis canceled. Your source pitch is unchanged.') {
    if (!active) return;
    const controller = active;
    active = null; controller.abort(); setBusy(false); status(message);
  }
  async function analyze(file) {
    cancel();
    const controller = new AbortController();
    active = controller;
    const current = () => active === controller && !controller.signal.aborted;
    const name = file ? file.name : 'Example speech';
    setBusy(true); $('pitch-result').hidden = true;
    $('pitch-preview').pause();
    status(`Reading ${name}…`);
    try {
      let blob = file;
      if (!blob) {
        const response = await fetch(new URL('./assets/speech-original.wav', import.meta.url), {signal: controller.signal});
        if (!response.ok) throw new Error('The example recording could not be loaded. Try choosing a local audio file.');
        blob = await response.blob();
      }
      if (!current()) return;
      const {samples, duration} = await decodeMono(blob);
      if (!current()) return;
      const result = await estimatePitch(samples, {
        signal: controller.signal,
        onStage: stage => {
          if (!current()) return;
          status(stage === 'loading' ? 'Loading SwiftF0 and its WASM runtime…' : `Estimating pitch for ${name}…`);
        },
        onProgress: fraction => { if (current()) status(`Estimating pitch for ${name}… ${Math.round(fraction * 100)}%`); },
      });
      if (!current()) return;
      const voiced = Array.from(result.pitchHz).filter((value, i) => result.voiced[i] && Number.isFinite(value));
      const estimate = voiced.length >= 7 ? Math.round(median(voiced)) : null;
      $('estimated-pitch').textContent = estimate === null ? '—' : `${estimate} Hz`;
      $('voiced-fraction').textContent = `${Math.round(100 * voiced.length / result.pitchHz.length)}%`;
      drawPitch(result, duration);
      if (previewUrl) URL.revokeObjectURL(previewUrl);
      previewUrl = URL.createObjectURL(blob);
      $('pitch-preview').src = previewUrl;
      $('pitch-result').hidden = false;
      if (estimate === null) {
        status(`${name}: too little confident voiced audio to estimate a median. Try a clear spoken sentence or a sustained vowel. Source pitch is unchanged.`);
      } else if (estimate < 65 || estimate > 500) {
        status(`${name}: median ${estimate} Hz is outside this synthesizer’s 65–500 Hz range. The contour is shown; source pitch is unchanged.`);
      } else {
        onPitch(estimate);
        status(`${name}: source pitch set to ${estimate} Hz. Listen to the current or target vowel above to hear it.`);
      }
    } catch (error) {
      if (current()) status(error.message || 'Pitch estimation failed. Try another recording.');
    } finally {
      if (current()) { active = null; setBusy(false); }
    }
  }
  $('pitch-example').addEventListener('click', () => analyze(null));
  $('pitch-file').addEventListener('change', event => {
    const file = event.target.files?.[0];
    event.target.value = '';
    if (file) analyze(file);
  });
  $('pitch-cancel').addEventListener('click', () => cancel());
  return {cancel};
}
