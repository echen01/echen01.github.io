/** JAX's partitionable Threefry2x32 PRNG and diffTLM/perlin.py.
 * Random uniforms match JAX's mantissa construction (not Math.random()).
 * Float64 is useful for JAX x64 validation; the application uses float32
 * random draws, matching upstream's default JAX configuration.
 */
export const jaxKey = seed => [Math.floor(seed / 4294967296) >>> 0, seed >>> 0];
export function threefry(key, high, low) {
  const keys = [key[0] >>> 0, key[1] >>> 0, (key[0] ^ key[1] ^ 0x1bd11bda) >>> 0];
  const rotations = [13, 15, 26, 6, 17, 29, 16, 24];
  let x = (high + keys[0]) >>> 0, y = (low + keys[1]) >>> 0;
  for (let round = 0; round < 20; round++) {
    const r = rotations[round % 8];
    x = (x + y) >>> 0;
    y = ((y << r | y >>> (32 - r)) ^ x) >>> 0;
    if (round % 4 === 3) {
      const injection = (round + 1) / 4;
      x = (x + keys[injection % 3]) >>> 0;
      y = (y + keys[(injection + 1) % 3] + injection) >>> 0;
    }
  }
  return [x, y];
}
export function splitKey(key, count = 2) {
  return Array.from({length: count}, (_, i) => threefry(key, Math.floor(i / 4294967296), i >>> 0));
}
export function jaxUniform(key, count, {dtype = 32, min = 0, max = 1} = {}) {
  const values = new Float64Array(count);
  for (let i = 0; i < count; i++) {
    const [high, low] = threefry(key, Math.floor(i / 4294967296), i >>> 0);
    const unit = dtype === 64 ? high / 4294967296 + (low >>> 12) / 4503599627370496 : ((high ^ low) >>> 9) / 8388608;
    values[i] = dtype === 64 ? Math.max(min, unit * (max - min) + min) : Math.max(min, Math.fround(Math.fround(unit * Math.fround(max - min)) + min));
  }
  return values;
}
export function fractalNoise(shape, {res = Math.floor(shape / 64), octaves = 5, persistence = .5, lacunarity = 2, key = jaxKey(0), dtype = 32} = {}) {
  const resolutions = Array.from({length: octaves}, (_, i) => res * lacunarity ** i);
  const counts = resolutions.map(r => r + 2);
  const [gradientKey, phaseKey] = splitKey(key);
  const gradients = jaxUniform(gradientKey, counts.reduce((a, b) => a + b, 0), {dtype, min: -1, max: 1});
  const phases = jaxUniform(phaseKey, octaves, {dtype});
  const result = new Float64Array(shape), f = dtype === 32 ? Math.fround : x => x;
  let offset = 0;
  for (let octave = 0; octave < octaves; octave++) {
    const step = f(resolutions[octave] / shape), amplitude = f(persistence ** octave);
    for (let n = 0; n < shape; n++) {
      const position = f(f(n * step) + phases[octave]), cell = Math.floor(position), t = f(position - cell);
      const d0 = f(gradients[offset + cell] * t), d1 = f(gradients[offset + cell + 1] * f(t - 1));
      const interpolant = f(f(f(t * t) * t) * f(f(t * f(f(t * 6) - 15)) + 10));
      result[n] = f(result[n] + f(amplitude * f(d0 + f(interpolant * f(d1 - d0)))));
    }
    offset += counts[octave];
  }
  return result;
}
export function rolloutNoise(sampleCount, iteration, {numAreas = 32, dtype = 32} = {}) {
  return splitKey(jaxKey(iteration), numAreas - 1).map(key => fractalNoise(sampleCount, {key, dtype}));
}
