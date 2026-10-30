import {rfft, irfft, irfftVjp, rfftVjp} from './faithful-fft.js';

// Direct float64 port of difftlm/tract.py::step_frame. The 31 distributed
// turbulence sources, source-to-constriction transfer functions, Reynolds
// gate, and complex phase are all part of this model. The last of 32 areas is
// the radiation boundary, as in the upstream implementation.
const DEFAULTS = Object.freeze({
  sampleRate: 32000, fftSize: 1024, length: 17.5, sections: 32,
  rho: 1.14e-3, c: 3.43e4, mu: 1.86e-4,
  wallMass: 2.1, wallResistance: 800, wallStiffness: 84500,
  reynoldsCritical: 3500, inputAmplitude: 400,
});

/** Smallest power-of-two FFT containing the original 25 ms frame. */
export function acousticFftSize(sampleRate = 32000) {
  if (!Number.isFinite(sampleRate) || sampleRate < 80) throw new Error('Sample rate must support a 25 ms frame of at least two samples.');
  return 2 ** Math.ceil(Math.log2(Math.floor(sampleRate * .025)));
}

function softplus(x) { return Math.max(x, 0) + Math.log1p(Math.exp(-Math.abs(x))); }
function sigmoid(x) { const e = Math.exp(-Math.abs(x)); return x >= 0 ? 1 / (1 + e) : e / (1 + e); }

// Complex scalars use adjacent real/imaginary entries. The reverse helpers
// implement the real-valued cotangent convention g = dL/dRe + i*dL/dIm.
function mul(v, y, a, b) {
  y *= 2; a *= 2; b *= 2;
  const ar = v[a], ai = v[a + 1], br = v[b], bi = v[b + 1];
  v[y] = ar * br - ai * bi; v[y + 1] = ar * bi + ai * br;
}
function div(v, y, a, b) {
  y *= 2; a *= 2; b *= 2;
  const ar = v[a], ai = v[a + 1], br = v[b], bi = v[b + 1], d = br * br + bi * bi;
  v[y] = (ar * br + ai * bi) / d; v[y + 1] = (ai * br - ar * bi) / d;
}
function add(v, y, a, b, sign = 1) { v[2 * y] = v[2 * a] + sign * v[2 * b]; v[2 * y + 1] = v[2 * a + 1] + sign * v[2 * b + 1]; }
function addAdj(g, y, a, b, sign = 1) {
  g[2 * a] += g[2 * y]; g[2 * a + 1] += g[2 * y + 1];
  g[2 * b] += sign * g[2 * y]; g[2 * b + 1] += sign * g[2 * y + 1];
}
function mulAdj(v, g, y, a, b) {
  y *= 2; a *= 2; b *= 2;
  const gr = g[y], gi = g[y + 1];
  g[a] += gr * v[b] + gi * v[b + 1]; g[a + 1] += gi * v[b] - gr * v[b + 1];
  g[b] += gr * v[a] + gi * v[a + 1]; g[b + 1] += gi * v[a] - gr * v[a + 1];
}
function divAdj(v, g, y, a, b) {
  y *= 2; a *= 2; b *= 2;
  const br = v[b], bi = v[b + 1], d = br * br + bi * bi;
  const gr = g[y], gi = g[y + 1];
  const gar = (gr * br - gi * bi) / d, gai = (gi * br + gr * bi) / d;
  g[a] += gar; g[a + 1] += gai;
  g[b] -= gar * v[y] + gai * v[y + 1]; g[b + 1] -= gai * v[y] - gar * v[y + 1];
}

function matrixMultiply(out, o, a, ao, b, bo) {
  for (let row = 0; row < 2; row++) for (let col = 0; col < 2; col++) {
    let re = 0, im = 0;
    for (let k = 0; k < 2; k++) {
      const ai = ao + 2 * (2 * row + k), bi = bo + 2 * (2 * k + col);
      re += a[ai] * b[bi] - a[ai + 1] * b[bi + 1]; im += a[ai] * b[bi + 1] + a[ai + 1] * b[bi];
    }
    const oi = o + 2 * (2 * row + col); out[oi] = re; out[oi + 1] = im;
  }
}
function matrixMultiplyAdj(g, go, a, ao, b, bo, ga, gao, gb, gbo) {
  for (let row = 0; row < 2; row++) for (let col = 0; col < 2; col++) {
    const yi = go + 2 * (2 * row + col), gr = g[yi], gi = g[yi + 1];
    for (let k = 0; k < 2; k++) {
      const ai = ao + 2 * (2 * row + k), bi = bo + 2 * (2 * k + col);
      const gai = gao + 2 * (2 * row + k), gbi = gbo + 2 * (2 * k + col);
      ga[gai] += gr * b[bi] + gi * b[bi + 1]; ga[gai + 1] += gi * b[bi] - gr * b[bi + 1];
      gb[gbi] += gr * a[ai] + gi * a[ai + 1]; gb[gbi + 1] += gi * a[ai] - gr * a[ai + 1];
    }
  }
}

function geometry(areas, opt) {
  const count = opt.sections - 1, li = opt.length / opt.sections;
  const values = new Float64Array(count * 6);
  for (let s = 0; s < count; s++) {
    const a = areas[s], surface = 2 * Math.PI * Math.sqrt(a / Math.PI) * li, i = 6 * s;
    values[i] = opt.rho * li / (2 * a);
    values[i + 1] = surface / (2 * a * a) * Math.sqrt(opt.rho * opt.mu * .5);
    values[i + 2] = a * li / (opt.rho * opt.c * opt.c);
    values[i + 3] = opt.wallMass / surface;
    values[i + 4] = opt.wallResistance / surface;
    values[i + 5] = surface / opt.wallStiffness;
  }
  const last = areas[count];
  return {values, Lr: 8 * opt.rho / (3 * Math.PI * Math.sqrt(Math.PI * last)), Rr: 128 * opt.rho * opt.c / (9 * Math.PI ** 2 * last)};
}

// Produces the section matrices and their derivatives with respect to each
// section's own log area. Y=1/Zb=1/Zb1+1/Zb2 is the parallel impedance identity;
// no acoustic term has been omitted. dY follows the exact geometric powers.
function frequencyMatrices(g, omega, count, K, dK, F, B, radiation) {
  const x = omega * g.Lr, r = g.Rr, d = r * r + x * x;
  radiation[0] = r * x * x / d; radiation[1] = r * r * x / d;
  // dZr/dlog(A_lip): differentiate the same parallel radiation impedance.
  const yr = 1 / r, yi = -.5 / x;
  const zr = radiation[0], zi = radiation[1], z2r = zr * zr - zi * zi, z2i = 2 * zr * zi;
  radiation[2] = -(z2r * yr - z2i * yi); radiation[3] = -(z2r * yi + z2i * yr);
  const rootOmega = Math.sqrt(omega);
  for (let s = 0; s < count; s++) {
    const j = 6 * s, i = 8 * s, p = g.values;
    const ar = p[j + 1] * rootOmega, ai = p[j] * omega;
    const b1r = p[j + 4], b1i = p[j + 3] * omega - 1 / (omega * p[j + 5]);
    const b1d = b1r * b1r + b1i * b1i, wallYr = b1r / b1d, wallYi = -b1i / b1d;
    const yR = wallYr, yI = wallYi + omega * p[j + 2];
    const dyR = .5 * wallYr, dyI = .5 * wallYi + omega * p[j + 2];
    const ayR = ar * yR - ai * yI, ayI = ar * yI + ai * yR;
    const a2R = ar * ar - ai * ai, a2I = 2 * ar * ai;
    K[i] = K[i + 6] = 1 + ayR; K[i + 1] = K[i + 7] = ayI;
    K[i + 2] = -2 * ar - (a2R * yR - a2I * yI);
    K[i + 3] = -2 * ai - (a2R * yI + a2I * yR);
    K[i + 4] = -yR; K[i + 5] = -yI;
    if (dK) {
      const dar = -1.5 * ar, dai = -ai;
      const dayR = dar * yR - dai * yI + ar * dyR - ai * dyI;
      const dayI = dar * yI + dai * yR + ar * dyI + ai * dyR;
      const da2R = 2 * (ar * dar - ai * dai), da2I = 2 * (ar * dai + ai * dar);
      dK[i] = dK[i + 6] = dayR; dK[i + 1] = dK[i + 7] = dayI;
      dK[i + 2] = -2 * dar - (da2R * yR - da2I * yI + a2R * dyR - a2I * dyI);
      dK[i + 3] = -2 * dai - (da2R * yI + da2I * yR + a2R * dyI + a2I * dyR);
      dK[i + 4] = -dyR; dK[i + 5] = -dyI;
    }
  }
  F.set(K.subarray(0, 8), 0); B.set(K.subarray((count - 1) * 8, count * 8), 0);
  for (let s = 1; s < count; s++) {
    matrixMultiply(F, s * 8, K, s * 8, F, (s - 1) * 8);
    // Preserve upstream's Ks_bkwd order exactly, including its use alongside
    // Ks_fwd without reversing the section axis in H_fric / H_c.
    matrixMultiply(B, s * 8, B, (s - 1) * 8, K, (count - 1 - s) * 8);
  }
}

// Static scalar expression graph for one section's H_fric and H_c.
function sectionTransfers(v, F, B, offset, radiation) {
  for (let j = 0; j < 8; j++) { v[j] = F[offset + j]; v[8 + j] = B[offset + j]; }
  v[16] = radiation[0]; v[17] = radiation[1];
  v[18] = -v[2]; v[19] = -v[3]; // 9 = -F01
  div(v, 10, 9, 3);            // Z1
  mul(v, 11, 6, 8); add(v, 12, 4, 11, -1); // Bd
  mul(v, 13, 7, 8); add(v, 14, 13, 5, -1); // Bn
  div(v, 15, 14, 12);          // Z2
  add(v, 16, 10, 15);
  mul(v, 17, 12, 16); mul(v, 18, 8, 10); div(v, 19, 18, 17); // H_fric
  mul(v, 20, 2, 15); add(v, 21, 0, 20, -1);
  const d = v[42] * v[42] + v[43] * v[43]; v[44] = v[42] / d; v[45] = -v[43] / d; // H_c
}

function sectionTransfersAdj(v, g, gF, gB, offset, gRadiation, hfr, hfi, hcr, hci) {
  g.fill(0); g[38] = hfr; g[39] = hfi;
  // H_c=1/Cden, hence derivative -H_c^2.
  const hr = v[44], hi = v[45], dr = -(hr * hr - hi * hi), di = -2 * hr * hi;
  g[42] = hcr * dr + hci * di; g[43] = hci * dr - hcr * di;
  addAdj(g, 21, 0, 20, -1); mulAdj(v, g, 20, 2, 15);
  divAdj(v, g, 19, 18, 17); mulAdj(v, g, 18, 8, 10); mulAdj(v, g, 17, 12, 16);
  addAdj(g, 16, 10, 15); divAdj(v, g, 15, 14, 12);
  addAdj(g, 14, 13, 5, -1); mulAdj(v, g, 13, 7, 8);
  addAdj(g, 12, 4, 11, -1); mulAdj(v, g, 11, 6, 8);
  divAdj(v, g, 10, 9, 3); g[2] -= g[18]; g[3] -= g[19];
  for (let j = 0; j < 8; j++) { gF[offset + j] += g[j]; gB[offset + j] += g[8 + j]; }
  gRadiation[0] += g[16]; gRadiation[1] += g[17];
}

function packedNoise(input, count, n) {
  if (input.length === count * n && typeof input[0] === 'number') return Float64Array.from(input);
  if (input.length !== count || input.some(row => row.length !== n)) throw new Error(`Noise must have shape [${count}, ${n}].`);
  const flat = new Float64Array(count * n); for (let s = 0; s < count; s++) flat.set(input[s], s * n); return flat;
}

export function createFaithfulAcoustic(options = {}) {
  const opt = {...DEFAULTS, ...options};
  opt.fftSize = options.fftSize ?? acousticFftSize(opt.sampleRate);
  if (opt.sections !== 32) throw new Error('The faithful browser model currently uses exactly 32 areas.');
  const n = opt.fftSize, count = opt.sections - 1, bins = n / 2 + 1;
  if (!Number.isInteger(n) || n < 2 || (n & (n - 1))) throw new Error('FFT size must be a power of two.');
  if (!(opt.sampleRate > 0) || !(opt.length > 0)) throw new Error('Sample rate and tract length must be positive.');
  const reynoldsScale = 4 * opt.rho ** 2 / (Math.PI * opt.mu ** 2);
  const threshold = (opt.reynoldsCritical / opt.inputAmplitude) ** 2;
  const alpha = 2 * Math.PI * 2000 / opt.sampleRate;

  function forward(uinInput, areaInput, noiseInput) {
    if (uinInput.length !== n || areaInput.length !== opt.sections) throw new Error('Invalid faithful acoustic input shape.');
    const areas = Float64Array.from(areaInput), noise = packedNoise(noiseInput, count, n);
    if (areas.some(a => !(a > 0) || !Number.isFinite(a))) throw new Error('Areas must be finite and strictly positive.');
    const geo = geometry(areas, opt), U = rfft(uinInput);
    const Hc = new Float64Array(count * bins * 2), Hf = new Float64Array(count * bins * 2), Ht = new Float64Array(bins * 2);
    const K = new Float64Array(count * 8), F = new Float64Array(count * 8), B = new Float64Array(count * 8), radiation = new Float64Array(4), v = new Float64Array(46);
    for (let f = 0; f < bins; f++) {
      const omega = 2 * Math.PI * Math.max(1e-8, f * opt.sampleRate / n);
      frequencyMatrices(geo, omega, count, K, null, F, B, radiation);
      for (let s = 0; s < count; s++) {
        sectionTransfers(v, F, B, s * 8, radiation);
        const j = 2 * (s * bins + f); Hf[j] = v[38]; Hf[j + 1] = v[39]; Hc[j] = v[44]; Hc[j + 1] = v[45];
      }
      const j = (count - 1) * 8, zr = radiation[0], zi = radiation[1];
      const dr = F[j] - (F[j + 4] * zr - F[j + 5] * zi), di = F[j + 1] - (F[j + 4] * zi + F[j + 5] * zr), d = dr * dr + di * di;
      Ht[2 * f] = (zr * dr + zi * di) / d; Ht[2 * f + 1] = (zi * dr - zr * di) / d;
    }
    const attention = new Float64Array(count), minArea = Math.min(...areas.subarray(0, count)); let attentionSum = 0;
    for (let s = 0; s < count; s++) attentionSum += (attention[s] = Math.exp(minArea - areas[s]));
    for (let s = 0; s < count; s++) attention[s] /= attentionSum;
    const lowpassed = new Float64Array(count * n), noiseSpectrum = new Float64Array(count * bins * 2);
    const outputReal = new Float64Array(bins), outputImag = new Float64Array(bins), sectionReal = new Float64Array(bins), sectionImag = new Float64Array(bins);
    for (let f = 0; f < bins; f++) {
      outputReal[f] = U.real[f] * Ht[2 * f] - U.imag[f] * Ht[2 * f + 1];
      outputImag[f] = U.real[f] * Ht[2 * f + 1] + U.imag[f] * Ht[2 * f];
    }
    for (let s = 0; s < count; s++) {
      const base = s * n;
      for (let f = 0; f < bins; f++) {
        const j = 2 * (s * bins + f);
        sectionReal[f] = Hc[j] * U.real[f] - Hc[j + 1] * U.imag[f];
        sectionImag[f] = Hc[j] * U.imag[f] + Hc[j + 1] * U.real[f];
      }
      const uc = irfft(sectionReal, sectionImag, n), un = new Float64Array(n); let previous = 0;
      for (let t = 0; t < n; t++) {
        previous += (uc[t] - previous) * alpha; lowpassed[base + t] = previous;
        const re2 = reynoldsScale * previous * previous / areas[s];
        un[t] = 1e-4 * noise[base + t] * softplus(re2 - threshold) * attention[s];
      }
      const unFFT = rfft(un);
      for (let f = 0; f < bins; f++) {
        const j = 2 * (s * bins + f), re = unFFT.real[f], im = unFFT.imag[f];
        noiseSpectrum[j] = re; noiseSpectrum[j + 1] = im;
        outputReal[f] += re * Hf[j] - im * Hf[j + 1]; outputImag[f] += re * Hf[j + 1] + im * Hf[j];
      }
    }
    const samples = irfft(outputReal, outputImag, n);

    function backward(cotangent) {
      if (cotangent.length !== n) throw new Error('Acoustic cotangent has the wrong length.');
      const G = irfftVjp(cotangent), gHc = new Float64Array(Hc.length), gHf = new Float64Array(Hf.length), gHt = new Float64Array(Ht.length);
      const result = new Float64Array(opt.sections), gAttention = new Float64Array(count);
      for (let f = 0; f < bins; f++) {
        gHt[2 * f] = G.real[f] * U.real[f] + G.imag[f] * U.imag[f];
        gHt[2 * f + 1] = G.imag[f] * U.real[f] - G.real[f] * U.imag[f];
      }
      for (let s = 0; s < count; s++) {
        const base = s * n;
        for (let f = 0; f < bins; f++) {
          const j = 2 * (s * bins + f), gr = G.real[f], gi = G.imag[f];
          gHf[j] = gr * noiseSpectrum[j] + gi * noiseSpectrum[j + 1];
          gHf[j + 1] = gi * noiseSpectrum[j] - gr * noiseSpectrum[j + 1];
          sectionReal[f] = gr * Hf[j] + gi * Hf[j + 1]; sectionImag[f] = gi * Hf[j] - gr * Hf[j + 1];
        }
        const gun = rfftVjp(sectionReal, sectionImag, n), guc = new Float64Array(n); let carry = 0;
        for (let t = n - 1; t >= 0; t--) {
          const lp = lowpassed[base + t], re2 = reynoldsScale * lp * lp / areas[s], x = re2 - threshold;
          const baseGradient = gun[t] * 1e-4 * noise[base + t];
          gAttention[s] += baseGradient * softplus(x);
          const gx = baseGradient * attention[s] * sigmoid(x);
          result[s] -= gx * re2;
          carry += gx * 2 * reynoldsScale * lp / areas[s];
          guc[t] = alpha * carry; carry *= 1 - alpha;
        }
        const gucFFT = irfftVjp(guc);
        for (let f = 0; f < bins; f++) {
          const j = 2 * (s * bins + f);
          gHc[j] = gucFFT.real[f] * U.real[f] + gucFFT.imag[f] * U.imag[f];
          gHc[j + 1] = gucFFT.imag[f] * U.real[f] - gucFFT.real[f] * U.imag[f];
        }
      }
      let attentionDot = 0; for (let s = 0; s < count; s++) attentionDot += attention[s] * gAttention[s];
      for (let s = 0; s < count; s++) result[s] -= areas[s] * attention[s] * (gAttention[s] - attentionDot);
      const dK = new Float64Array(K.length), gK = new Float64Array(K.length), gF = new Float64Array(F.length), gB = new Float64Array(B.length), g = new Float64Array(v.length), gRadiation = new Float64Array(2);
      for (let f = 0; f < bins; f++) {
        frequencyMatrices(geo, 2 * Math.PI * Math.max(1e-8, f * opt.sampleRate / n), count, K, dK, F, B, radiation);
        gK.fill(0); gF.fill(0); gB.fill(0); gRadiation.fill(0);
        for (let s = 0; s < count; s++) {
          sectionTransfers(v, F, B, s * 8, radiation);
          const j = 2 * (s * bins + f);
          sectionTransfersAdj(v, g, gF, gB, s * 8, gRadiation, gHf[j], gHf[j + 1], gHc[j], gHc[j + 1]);
        }
        // H_tract = Zr / (F00 - F10*Zr).
        const j = (count - 1) * 8, zr = radiation[0], zi = radiation[1];
        const dr = F[j] - (F[j + 4] * zr - F[j + 5] * zi), di = F[j + 1] - (F[j + 4] * zi + F[j + 5] * zr), d = dr * dr + di * di;
        const gr = gHt[2 * f], gi = gHt[2 * f + 1], nr = (gr * dr - gi * di) / d, ni = (gi * dr + gr * di) / d;
        gRadiation[0] += nr; gRadiation[1] += ni;
        const dgr = -(nr * Ht[2 * f] + ni * Ht[2 * f + 1]), dgi = -(ni * Ht[2 * f] - nr * Ht[2 * f + 1]);
        gF[j] += dgr; gF[j + 1] += dgi;
        gF[j + 4] -= dgr * zr + dgi * zi; gF[j + 5] -= dgi * zr - dgr * zi;
        gRadiation[0] -= dgr * F[j + 4] + dgi * F[j + 5]; gRadiation[1] -= dgi * F[j + 4] - dgr * F[j + 5];
        for (let s = count - 1; s >= 1; s--) {
          matrixMultiplyAdj(gF, s * 8, K, s * 8, F, (s - 1) * 8, gK, s * 8, gF, (s - 1) * 8);
          const section = count - 1 - s;
          matrixMultiplyAdj(gB, s * 8, B, (s - 1) * 8, K, section * 8, gB, (s - 1) * 8, gK, section * 8);
        }
        for (let q = 0; q < 8; q++) { gK[q] += gF[q]; gK[(count - 1) * 8 + q] += gB[q]; }
        for (let s = 0; s < count; s++) for (let q = 0; q < 8; q++) result[s] += gK[s * 8 + q] * dK[s * 8 + q];
        result[count] += gRadiation[0] * radiation[2] + gRadiation[1] * radiation[3];
      }
      return result;
    }
    return {samples, backward};
  }
  return {options: Object.freeze({...opt}), forward};
}
