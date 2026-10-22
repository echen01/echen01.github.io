// Usage: node test-browser.mjs [path/to/playwright/index.mjs]
// Requires Playwright and Chrome (or CHROMIUM_PATH); no test dependencies ship
// to page visitors. Uses the vendored model in an actual browser Web Worker.
import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import { resolve, sep } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { createHash } from 'node:crypto';

const root = resolve(fileURLToPath(new URL('../../', import.meta.url)));
const manifest = JSON.parse(await readFile(new URL('./vendor-manifest.json', import.meta.url)));
for (const [name, entry] of Object.entries(manifest.files)) {
  const data = await readFile(new URL(name, import.meta.url));
  assert.equal(data.length, entry.bytes, `${name}: length`);
  assert.equal(createHash('sha256').update(data).digest('hex'), entry.sha256, `${name}: SHA-256`);
}
const mime = { html: 'text/html', js: 'text/javascript', mjs: 'text/javascript', wasm: 'application/wasm', wav: 'audio/wav' };
const server = createServer(async (request, response) => {
  const path = resolve(root, `.${decodeURIComponent(new URL(request.url, 'http://localhost').pathname)}`);
  if (!path.startsWith(root + sep) && path !== root) { response.writeHead(403).end(); return; }
  try {
    const data = await readFile(path);
    response.writeHead(200, { 'Content-Type': mime[path.split('.').pop()] || 'application/octet-stream' });
    response.end(data);
  } catch { response.writeHead(404).end(); }
});
await new Promise((done) => server.listen(0, '127.0.0.1', done));
const base = `http://127.0.0.1:${server.address().port}`;
let browser;
try {
  const playwright = await import(process.argv[2] ? pathToFileURL(resolve(process.argv[2])).href : 'playwright');
  browser = await playwright.chromium.launch(process.env.CHROMIUM_PATH
    ? { executablePath: process.env.CHROMIUM_PATH, headless: true }
    : { channel: 'chrome', headless: true });
  const context = await browser.newContext();
  const requests = [];
  const external = [];
  context.on('request', (request) => {
    requests.push(request.url());
    if (!request.url().startsWith(base)) external.push(request.url());
  });
  const page = await context.newPage();
  const errors = [];
  page.on('pageerror', (error) => errors.push(error.message));
  await page.goto(`${base}/vocal-tract/pitch/test.html`);
  await page.evaluate(async () => { window.pitch = await import('./swift-f0.js'); });
  assert(!requests.some((url) => /\.onnx$|\.wasm$/.test(url)), 'model/runtime are lazy');
  assert.equal(await page.evaluate(() => crossOriginIsolated), false, 'test ordinary static hosting');
  const result = await page.evaluate(async () => (await import('./test-suite.js')).runPitchTests());
  assert.equal(requests.filter((url) => url.endsWith('/model.onnx')).length, 1, 'one cached model');
  assert.deepEqual(external, [], 'all inference requests must remain local');
  assert.deepEqual(errors, [], 'no browser exceptions');
  console.log(JSON.stringify({ passed: true, ...result }, null, 2));
} finally {
  await browser?.close();
  await new Promise((done) => server.close(done));
}
