/** Liljencrants–Fant glottal volume-flow source, matching diffTLM glottis.py.
 * The tract transfer function consumes flow, so this is the LF flow pulse,
 * not its derivative. Both vowel playback and speech fitting use these equations.
 */
function epsilonFor(ta, te) {
  const c = Math.max(1 - te, .001);
  ta = Math.min(Math.max(ta, .0001), c - .00001);
  let epsilon = 1 / ta;
  for (let step = 0; step < 40; step++) {
    const h = 1 - Math.exp(-epsilon * c) - epsilon * ta;
    if (Math.abs(h) < .00001) break;
    epsilon -= h / (c * Math.exp(-epsilon * c) - ta);
  }
  return epsilon;
}

function alphaFor(tp, te, ta, epsilon) {
  const w = Math.PI / tp, w2 = w * w, sin = Math.sin(w * te), cos = Math.cos(w * te);
  const X = sin * Math.exp(epsilon * te) * (-Math.exp(-epsilon) / epsilon - Math.exp(-epsilon)) / (epsilon * ta);
  const Y = sin * Math.exp(epsilon * te) * (-Math.exp(-epsilon * te) / epsilon - te * Math.exp(-epsilon)) / (epsilon * ta);
  const h = a => w / (w2 + a * a) + Math.exp(a * te) * ((a * sin - w * cos) / (w2 + a * a) + X - Y);
  let low = 0, high = 0, alpha = 0;
  for (let step = 0; step < 20; step++) { high++; if (h(high) < 0) break; }
  if (h(high) >= 0) return 0;
  for (let step = 0; step < 20; step++) {
    const lower = h(low), upper = h(high);
    alpha = low - lower * (high - low) / (upper - lower);
    const value = h(alpha);
    if (Math.abs(value) <= .00001) break;
    if (value < 0) high = alpha; else low = alpha;
  }
  return alpha;
}

export function createLfFlowPulse({
  openQuotient = .5, shapeQuotient = 3, spectralTilt = .02,
} = {}) {
  const te = openQuotient, tp = te * shapeQuotient / (1 + shapeQuotient), ta = Math.min(Math.max(spectralTilt, .01), 1 - te);
  const epsilon = epsilonFor(ta, te), alpha = alphaFor(tp, te, ta, epsilon), w = Math.PI / tp;
  const B = 1 / ((Math.exp(alpha * tp) * (alpha * Math.sin(w * tp) - w * Math.cos(w * tp)) + w) / (w * w + alpha * alpha));
  const endFlow = B * (Math.exp(alpha * te) * (alpha * Math.sin(w * te) - w * Math.cos(w * te)) + w) / (w * w + alpha * alpha);
  const factor = B * Math.exp(alpha * te) * Math.sin(w * te) * Math.exp(epsilon * te) / (epsilon * ta);
  const endIntegral = factor * (-Math.exp(-epsilon * te) / epsilon - te * Math.exp(-epsilon));
  return t => t < te
    ? B * (Math.exp(alpha * t) * (alpha * Math.sin(w * t) - w * Math.cos(w * t)) + w) / (w * w + alpha * alpha)
    : endFlow + factor * (-Math.exp(-epsilon * t) / epsilon - t * Math.exp(-epsilon)) - endIntegral;
}

/** One periodic, noise-free LF cycle, sampled on [0, 1). */
export function lfPulseCycle(count = 2048, options) {
  if (!Number.isInteger(count) || count < 2) throw new Error('Invalid LF cycle sample count.');
  const pulseAt = createLfFlowPulse(options);
  return Float64Array.from({length: count}, (_, i) => pulseAt(i / count));
}
