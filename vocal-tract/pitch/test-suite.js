import { estimatePitch, disposePitchEstimator } from './swift-f0.js';

export async function runPitchTests() {
  const check = (value, message) => { if (!value) throw new Error(message); };
  const median = (values) => { const sorted = values.toSorted((a, b) => a - b); return sorted[Math.floor(sorted.length / 2)]; };
  const summaries = [];
  const begun = performance.now();
  for (const hz of [120, 220, 400]) {
    const samples = Float32Array.from({ length: 32000 }, (_, i) => {
      const phase = 2 * Math.PI * hz * i / 16000;
      return 0.35 * Math.sin(phase) + 0.1 * Math.sin(2 * phase) + 0.05 * Math.sin(3 * phase);
    });
    const result = await estimatePitch(samples);
    const voicedPitch = [...result.pitchHz].filter((_, i) => result.voiced[i]);
    const measured = median(voicedPitch);
    check(result.pitchHz.length === 125, 'frame count');
    check(Math.abs(result.times[10] - 0.16) < 1e-12, 'timestamps');
    check(result.voiced.every((v, i) => Boolean(v) === (result.confidence[i] >= 0.5)), 'confidence threshold');
    check(voicedPitch.length > 100, `${hz} Hz voiced detection`);
    check(Math.abs(measured / hz - 1) < 0.03, `${hz} Hz pitch accuracy: ${measured}`);
    check(samples.length === 32000, 'caller audio must remain usable');
    summaries.push({ targetHz: hz, measuredHz: measured, voicedFrames: voicedPitch.length });
  }
  const silent = await estimatePitch(new Float32Array(16000));
  check(silent.voiced.every((v) => !v), 'silence must be unvoiced');
  check(silent.confidence.every((v) => v === 0), 'silence gate');
  check(silent.loudnessDb.every((v) => v === -140), 'silence loudness');
  const short = await estimatePitch(new Float32Array(32));
  check(short.pitchHz.length === 1 && !short.voiced[0], 'short clips yield one unvoiced frame');
  
  const response = await fetch('../assets/speech-original.wav');
  const context = new OfflineAudioContext(1, 1, 16000);
  const audio = await context.decodeAudioData(await response.arrayBuffer());
  const mono = new Float32Array(audio.length);
  for (let c = 0; c < audio.numberOfChannels; c++) {
    const channel = audio.getChannelData(c);
    for (let i = 0; i < mono.length; i++) mono[i] += channel[i] / audio.numberOfChannels;
  }
  const speech = await estimatePitch(mono);
  const voicedSpeech = [...speech.pitchHz].filter((_, i) => speech.voiced[i]);
  check(voicedSpeech.length > 30, 'bundled speech has voiced frames');
  check(voicedSpeech.length < speech.pitchHz.length, 'speech includes unvoiced frames');
  check(median(voicedSpeech) > 60 && median(voicedSpeech) < 500, 'plausible speech F0');
  const aborter = new AbortController();
  aborter.abort();
  try { await estimatePitch(mono, { signal: aborter.signal }); throw new Error('cancellation accepted'); }
  catch (error) { check(error.name === 'AbortError', 'cancellation is reported'); }
  try { await estimatePitch(Float32Array.of(NaN)); throw new Error('invalid audio accepted'); }
  catch (error) { check(error.name === 'TypeError', 'invalid samples rejected'); }
  disposePitchEstimator();
  return { model: speech.modelVersion, synthetic: summaries, silenceFrames: silent.pitchHz.length, speechFrames: speech.pitchHz.length, voicedSpeechFrames: voicedSpeech.length, speechMedianHz: median(voicedSpeech), durationMs: performance.now() - begun };
}
