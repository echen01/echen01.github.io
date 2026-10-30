// Shared float64 FFT for the direct port of diffTLM. Forward is unnormalized;
// inverse is normalized by N, matching jax.numpy.fft's default convention.
const plans = new Map();

function planFor(n) {
  if (!Number.isInteger(n) || n < 2 || (n & (n - 1))) throw new Error('FFT length must be a power of two.');
  if (plans.has(n)) return plans.get(n);
  const reverse = new Uint32Array(n), cos = new Float64Array(n / 2), sin = new Float64Array(n / 2);
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    while (j & bit) { j ^= bit; bit >>= 1; }
    j ^= bit; reverse[i] = j;
  }
  for (let k = 0; k < n / 2; k++) { cos[k] = Math.cos(2 * Math.PI * k / n); sin[k] = Math.sin(2 * Math.PI * k / n); }
  const plan = {reverse, cos, sin}; plans.set(n, plan); return plan;
}

export function fft(real, imag, inverse = false) {
  const n = real.length, p = planFor(n);
  if (imag.length !== n) throw new Error('FFT real and imaginary lengths differ.');
  for (let i = 1; i < n; i++) {
    const j = p.reverse[i];
    if (i < j) { let t = real[i]; real[i] = real[j]; real[j] = t; t = imag[i]; imag[i] = imag[j]; imag[j] = t; }
  }
  for (let size = 2; size <= n; size *= 2) {
    const half = size / 2, stride = n / size;
    for (let start = 0; start < n; start += size) for (let k = 0; k < half; k++) {
      const a = start + k, b = a + half, wr = p.cos[k * stride], wi = (inverse ? 1 : -1) * p.sin[k * stride];
      const br = real[b] * wr - imag[b] * wi, bi = real[b] * wi + imag[b] * wr;
      real[b] = real[a] - br; imag[b] = imag[a] - bi; real[a] += br; imag[a] += bi;
    }
  }
  if (inverse) for (let i = 0; i < n; i++) { real[i] /= n; imag[i] /= n; }
  return {real, imag};
}

export function rfft(input) {
  const real = Float64Array.from(input), imag = new Float64Array(input.length);
  fft(real, imag);
  return {real: real.slice(0, input.length / 2 + 1), imag: imag.slice(0, input.length / 2 + 1)};
}

export function irfft(halfReal, halfImag, n = (halfReal.length - 1) * 2) {
  if (halfReal.length !== n / 2 + 1 || halfImag.length !== halfReal.length) throw new Error('Invalid real inverse FFT shape.');
  const real = new Float64Array(n), imag = new Float64Array(n);
  real.set(halfReal); imag.set(halfImag); imag[0] = 0; imag[n / 2] = 0;
  for (let k = 1; k < n / 2; k++) { real[n - k] = real[k]; imag[n - k] = -imag[k]; }
  fft(real, imag, true); return real;
}

// Cotangents represent dL/d(real) + i*dL/d(imag), not JAX's conjugated
// complex-gradient convention. Both helpers are adjoints for real scalar L.
export function irfftVjp(cotangent) {
  const n = cotangent.length, result = rfft(cotangent);
  for (let k = 0; k <= n / 2; k++) {
    const scale = (k === 0 || k === n / 2 ? 1 : 2) / n;
    result.real[k] *= scale; result.imag[k] *= scale;
  }
  result.imag[0] = 0; result.imag[n / 2] = 0;
  return result;
}

export function rfftVjp(halfReal, halfImag, n = (halfReal.length - 1) * 2) {
  const real = new Float64Array(n), imag = new Float64Array(n);
  real.set(halfReal); imag.set(halfImag);
  fft(real, imag, true);
  for (let k = 0; k < n; k++) real[k] *= n;
  return real;
}
