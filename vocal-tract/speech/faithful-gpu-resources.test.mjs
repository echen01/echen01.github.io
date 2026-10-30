import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {loadFaithfulInitialization} from './faithful-field.js';
import {loadFaithfulMfnInitialization} from './faithful-mfn-field.js';
import {createFaithfulGpuFitter} from './faithful-gpu-engine.js';

// Exercise the real fitter's resource ownership and failure paths. This API
// double records GPU resources and readbacks; it does not execute or emulate
// shader arithmetic. Numerical fidelity is tested by the browser/JAX harnesses.
const originalGlobals = new Map(['navigator', 'GPUBufferUsage', 'GPUMapMode'].map(name => [name, Object.getOwnPropertyDescriptor(globalThis, name)]));
const usage = {MAP_READ: 1, COPY_SRC: 4, COPY_DST: 8, UNIFORM: 64, STORAGE: 128};
Object.defineProperty(globalThis, 'GPUBufferUsage', {configurable: true, value: usage});
Object.defineProperty(globalThis, 'GPUMapMode', {configurable: true, value: {READ: 1}});

function fakeDevice({failPipeline} = {}) {
  const buffers = [], listeners = new Map(), hostWrites = new Map(), trace = [];
  let resolveLost;
  const device = {
    buffers, trace, destroyCalls: 0, nextMapFault: null, nextRangeFault: null, onWorkDone: null,
    lost: new Promise(resolve => { resolveLost = resolve; }),
    limits: {maxComputeWorkgroupsPerDimension: 65535, maxComputeWorkgroupStorageSize: 16384,
      maxComputeInvocationsPerWorkgroup: 256, maxComputeWorkgroupSizeX: 256},
    queue: {
      writeBuffer(buffer, offset, data) {
        const bytes = ArrayBuffer.isView(data)
          ? new Uint8Array(data.buffer, data.byteOffset, data.byteLength) : new Uint8Array(data);
        // Record CPU-supplied values at dispatch time without emulating GPU
        // writes. This catches an incorrect RNG seed or Adam iteration upload.
        hostWrites.set(buffer.id, {offset, hash: createHash('sha256').update(bytes).digest('hex'),
          words: bytes.byteLength <= 16 ? Array.from(new Uint32Array(bytes.slice().buffer)) : undefined});
      },
      submit(commandBuffers) {
        for (const commandBuffer of commandBuffers) {
          trace.push({type: 'submit', label: commandBuffer.label, commands: commandBuffer.commands.map(command =>
            command.type !== 'dispatch' ? command : {...command, bindings: command.bindings.map(binding =>
              ({...binding, hostWrite: hostWrites.get(binding.buffer)}))})});
        }
      },
      async onSubmittedWorkDone() { trace.push({type: 'wait'}); await device.onWorkDone?.(); },
    },
    lose(info) { resolveLost(info); },
    emitError(error) { listeners.get('uncapturederror')?.({error}); },
    addEventListener(type, listener) { listeners.set(type, listener); },
    removeEventListener(type, listener) { if (listeners.get(type) === listener) listeners.delete(type); },
    listenerCount() { return listeners.size; },
    createBuffer(descriptor) {
      const buffer = {...descriptor, id: buffers.length, destroyCalls: 0, unmapCalls: 0, mapState: 'unmapped',
        async mapAsync() {
          trace.push({type: 'map', buffer: this.id});
          assert.equal(this.destroyCalls, 0, 'Destroyed buffer must not be mapped.');
          this.mapState = 'pending';
          const fault = device.nextMapFault; device.nextMapFault = null;
          if (fault) {
            this.mapState = 'unmapped';
            await fault();
          }
          this.mapState = 'mapped';
        },
        getMappedRange() {
          assert.equal(this.mapState, 'mapped');
          const fault = device.nextRangeFault; device.nextRangeFault = null;
          if (fault) throw fault;
          return new ArrayBuffer(this.size);
        },
        unmap() { this.unmapCalls++; this.mapState = 'unmapped'; },
        destroy() { this.destroyCalls++; this.mapState = 'unmapped'; },
      };
      buffers.push(buffer); return buffer;
    },
    createShaderModule() { return {async getCompilationInfo() { return {messages: []}; }}; },
    async createComputePipelineAsync(descriptor) {
      if (descriptor.label === failPipeline) throw new Error('Injected pipeline construction failure.');
      return {label: descriptor.label, entryPoint: descriptor.compute.entryPoint, getBindGroupLayout() { return {}; }};
    },
    createBindGroup(descriptor) { return descriptor; },
    createCommandEncoder({label} = {}) {
      const commands = [];
      return {
        copyBufferToBuffer(source, sourceOffset, destination, destinationOffset, size) {
          commands.push({type: 'copy', source: source.id, sourceOffset, destination: destination.id, destinationOffset, size});
        },
        clearBuffer(buffer, offset = 0, size = buffer.size - offset) { commands.push({type: 'clear', buffer: buffer.id, offset, size}); },
        finish() { return {label, commands}; },
        beginComputePass() {
          let pipeline;
          const groups = new Map();
          return {
            setPipeline(value) { pipeline = value; },
            setBindGroup(index, value) { groups.set(index, value); },
            dispatchWorkgroups(x, y = 1, z = 1) {
              const bindings = [...groups].flatMap(([group, value]) => value.entries.map(entry =>
                ({group, binding: entry.binding, buffer: entry.resource.buffer.id,
                  offset: entry.resource.offset ?? 0, size: entry.resource.size ?? entry.resource.buffer.size})));
              commands.push({type: 'dispatch', pipeline: pipeline.label, entryPoint: pipeline.entryPoint, groups: [x, y, z], bindings});
            },
            end() {},
          };
        },
      };
    },
    destroy() { this.destroyCalls++; resolveLost({reason: 'destroyed', message: 'Owned device disposed.'}); },
  };
  return device;
}

const base = new URL('./upstream/reference-small/', import.meta.url);
const initialization = await loadFaithfulInitialization(name => fs.readFile(new URL(name, base)), {hiddenDim: 128});
const input = {sampleRate: 16000, target: Float32Array.from({length: 400}, (_, i) => .1 * Math.sin(i * .13)), uin: new Float32Array(400)};
// float32 intentionally bypasses the compensated-arithmetic device self-check:
// a fake device cannot establish compiler or numerical correctness.
const options = {hiddenDim: 128, initialization, iterations: 2, fftOptions: {arithmetic: 'float32'}};
const cases = [];

function installAdapter(device) {
  Object.defineProperty(globalThis, 'navigator', {configurable: true,
    value: {gpu: {async requestAdapter() { return {async requestDevice() { return device; }}; }}}});
}
function assertResourcesReleased(device) {
  assert(device.buffers.length > 0, 'The case must allocate real fitter resources.');
  assert(device.buffers.every(buffer => buffer.destroyCalls > 0), 'Every allocated GPU buffer must be destroyed.');
  assert.equal(device.listenerCount(), 0, 'Device error listeners must be removed.');
}
async function withFitter(device, operation, {owned = false, engineOptions = {}, sample = input} = {}) {
  if (owned) installAdapter(device);
  const fitter = await createFaithfulGpuFitter(sample, {...options, ...engineOptions, ...(owned ? {} : {device})});
  try { await operation(fitter); }
  finally { fitter.dispose(); }
  assertResourcesReleased(device);
  assert.equal(device.destroyCalls, owned ? 1 : 0, 'Only an internally requested device may be destroyed.');
}

try {
  {
    const mfnBase = new URL('./upstream/reference-mfn/', import.meta.url);
    const mfnInitialization = await loadFaithfulMfnInitialization(name => fs.readFile(new URL(name, mfnBase)));
    const engineOptions = {fieldType: 'mfn', hiddenDim: 256, initialization: mfnInitialization};
    const device = fakeDevice();
    await withFitter(device, async fitter => {
      await fitter.evaluate(); await fitter.step(); await fitter.synthesize(); await fitter.step();
      const snapshot = await fitter.snapshot();
      assert.equal(snapshot.iteration, 2); assert.equal(snapshot.fieldMetadata.family, 'mfn');
      const dispatches = device.trace.filter(event => event.type === 'submit').flatMap(event => event.commands).filter(command => command.type === 'dispatch');
      const updates = dispatches.filter(command => command.pipeline === 'Faithful MFN Optax Adam');
      assert.deepEqual(updates.map(command => command.bindings.find(binding => binding.binding === 4).hostWrite.words[0]), [0, 1]);
      assert(!dispatches.some(command => command.pipeline === 'Faithful Optax Adam'), 'MFN must use its own GPU field and optimizer schedule.');
    }, {engineOptions});
    const failing = fakeDevice({failPipeline: 'Faithful MFN product VJP 2'});
    await assert.rejects(createFaithfulGpuFitter(input, {...options, ...engineOptions, device: failing}), /Injected pipeline construction failure/);
    assertResourcesReleased(failing);
    cases.push('MFN selection, two updates with previews, and partial-construction cleanup');
  }
  for (const owned of [false, true]) {
    const device = fakeDevice();
    await withFitter(device, async fitter => {
      await fitter.snapshot();
      fitter.dispose();
      const calls = device.buffers.map(buffer => buffer.destroyCalls);
      fitter.dispose();
      assert.deepEqual(device.buffers.map(buffer => buffer.destroyCalls), calls, 'Repeated dispose must have no effects.');
    }, {owned});
    cases.push(`${owned ? 'owned' : 'supplied'} device disposal is idempotent`);
  }

  for (const owned of [false, true]) {
    // This fails after the field has been built and rollout buffers allocated,
    // checking cleanup both within the failed constructor and by the fitter.
    const device = fakeDevice({failPipeline: 'Faithful Threefry uniforms'});
    if (owned) installAdapter(device);
    await assert.rejects(createFaithfulGpuFitter(input, {...options, ...(owned ? {} : {device})}), /Injected pipeline construction failure/);
    assertResourcesReleased(device);
    assert.equal(device.destroyCalls, owned ? 1 : 0);
    cases.push(`initialization failure cleans ${owned ? 'owned' : 'supplied'} device resources`);
  }

  {
    const device = fakeDevice();
    await withFitter(device, async fitter => {
      device.nextMapFault = async () => {
        device.lose({reason: 'destroyed', message: 'Injected GPU reset.'});
        throw new Error('Mapping WebGPU buffer failed: Context lost');
      };
      await assert.rejects(fitter.step(), error => {
        assert.equal(error.code, 'WEBGPU_DEVICE_LOST');
        assert.equal(error.reason, 'destroyed');
        assert.match(error.message, /Injected GPU reset/);
        return true;
      });
    });
    cases.push('loss readback preserves device-loss diagnostics');
  }

  {
    const device = fakeDevice();
    await withFitter(device, async fitter => {
      const interrupted = new DOMException('Buffer was unmapped before mapping was resolved.', 'AbortError');
      let lossDelivered = false;
      device.nextMapFault = async () => {
        // Match the browser ordering: mapAsync rejects before the device-loss
        // notification arrives in a later task, not merely a microtask.
        setTimeout(() => {
          lossDelivered = true;
          device.lose({reason: 'destroyed', message: 'Injected delayed GPU reset.'});
        }, 0);
        throw interrupted;
      };
      await assert.rejects(fitter.step(), error => {
        assert.equal(lossDelivered, true);
        assert.equal(error.code, 'WEBGPU_DEVICE_LOST');
        assert.equal(error.reason, 'destroyed');
        assert.match(error.message, /Injected delayed GPU reset/);
        assert.equal(error.cause, interrupted);
        assert.notEqual(error.name, 'AbortError', 'A GPU reset must not be mistaken for user cancellation.');
        return true;
      });
    });
    cases.push('delayed device loss identifies an earlier AbortError mapping rejection');
  }

  {
    const device = fakeDevice();
    await withFitter(device, async fitter => {
      const interrupted = new DOMException('Buffer was unmapped before mapping was resolved.', 'AbortError');
      device.nextMapFault = async () => { throw interrupted; };
      await assert.rejects(fitter.step(), error => {
        assert.equal(error.name, 'Error', 'An unexplained mapping interruption is not user cancellation.');
        assert.notEqual(error.code, 'WEBGPU_DEVICE_LOST', 'No loss notification means no automatic loss recovery.');
        assert.match(error.message, /GPU operation was interrupted/);
        assert.equal(error.cause, interrupted);
        return true;
      });
      await fitter.snapshot();
    });
    cases.push('mapping AbortError without device loss remains an ordinary recoverable error');
  }

  {
    const device = fakeDevice();
    await withFitter(device, async fitter => {
      device.nextMapFault = async () => { throw new Error('Mapping WebGPU buffer failed: Context lost'); };
      await assert.rejects(fitter.snapshot(), error => error.code === 'WEBGPU_DEVICE_LOST');
    });
    cases.push('context-loss mapping rejection is classified without a loss event');
  }

  {
    const device = fakeDevice();
    await withFitter(device, async fitter => {
      device.emitError(new Error('Injected GPU validation error.'));
      await assert.rejects(fitter.step(), error => {
        assert.match(error.message, /Injected GPU validation error/);
        assert.notEqual(error.code, 'WEBGPU_DEVICE_LOST', 'Validation errors must not trigger loss recovery.');
        return true;
      });
    });
    cases.push('ordinary GPU errors are not classified as device loss');
  }

  {
    const device = fakeDevice();
    await withFitter(device, async fitter => {
      device.nextRangeFault = new Error('Injected mapped-range failure.');
      await assert.rejects(fitter.snapshot(), /Injected mapped-range failure/);
      assert(device.buffers.every(buffer => buffer.mapState !== 'mapped'), 'Thrown readback must still unmap.');
      // A non-loss readback error must not poison an otherwise usable device.
      await fitter.snapshot();
    });
    cases.push('mapped-range exception unmaps the staging buffer');
  }

  for (const method of ['readParameters', 'readGradient']) {
    const device = fakeDevice();
    await withFitter(device, async fitter => {
      const before = device.buffers.length;
      device.nextMapFault = async () => { throw new Error('Injected transient mapping failure.'); };
      await assert.rejects(fitter[method](), /Injected transient mapping failure/);
      const temporary = device.buffers.slice(before);
      assert.equal(temporary.length, 1);
      assert.equal(temporary[0].destroyCalls, 1, 'Temporary readback storage must be destroyed before fitter disposal.');
      const values = await fitter[method]();
      assert.equal(values.length, initialization.parameters.length);
    });
    cases.push(`${method} releases temporary staging after rejection`);
  }

  {
    // Eleven frames exercise both a complete eight-frame acoustic batch and
    // the remainder. Host writes are attached to dispatch records at queue
    // submission time, so equality also checks seed/Adam upload timing.
    const sample = {sampleRate: 16000,
      target: Float32Array.from({length: 1400}, (_, i) => .1 * Math.sin(i * .13)), uin: new Float32Array(1400)};
    async function commandTraces(submissionMode) {
      const device = fakeDevice(), result = [];
      await withFitter(device, async fitter => {
        for (const [method, argument] of [['evaluate', 17], ['step'], ['step']]) {
          device.trace.length = 0;
          await fitter[method](argument);
          const submissions = device.trace.filter(event => event.type === 'submit');
          assert.equal(submissions.length, submissionMode === 'staged' ? 5 : 1);
          assert.deepEqual(device.trace.map(event => event.type), submissionMode === 'staged'
            ? ['submit', 'wait', 'submit', 'wait', 'submit', 'wait', 'submit', 'wait', 'submit', 'map']
            : ['submit', 'map'], 'Every stage must complete before submission of the next stage.');
          result.push(submissions.flatMap(event => event.commands));
        }
      }, {sample, engineOptions: {submissionMode}});
      return result;
    }
    const combined = await commandTraces('combined'), staged = await commandTraces('staged');
    for (let operation = 0; operation < combined.length; operation++) {
      assert.equal(staged[operation].length, combined[operation].length);
      for (let command = 0; command < combined[operation].length; command++) {
        assert.deepEqual(staged[operation][command], combined[operation][command],
          `Operation ${operation}, command ${command}: staging must preserve GPU order, bindings and host uploads.`);
      }
      const dispatches = staged[operation].filter(command => command.type === 'dispatch');
      const random = dispatches.filter(command => command.pipeline === 'Faithful Threefry uniforms');
      assert.equal(random.length, 1, 'Generate Perlin randomness once for each objective.');
      assert.equal(random[0].bindings.find(binding => binding.binding === 0).hostWrite.words[0], [17, 0, 1][operation]);
      const adam = dispatches.filter(command => command.pipeline === 'Faithful Optax Adam');
      assert.equal(adam.length, operation === 0 ? 0 : 1, 'Evaluation must not update; a step must update exactly once.');
      if (adam.length) {
        assert.equal(adam[0].bindings.find(binding => binding.binding === 4).hostWrite.words[0], operation - 1);
        assert.equal(dispatches.at(-1), adam[0], 'Adam must follow the complete backward pass.');
      }
    }
    cases.push('staged and combined evaluation/two steps preserve command order, bindings and RNG/Adam uploads');
  }

  {
    async function trainingTraces(withPreviews) {
      const device = fakeDevice(), steps = [];
      await withFitter(device, async fitter => {
        async function preview(iteration) {
          const allocated = device.buffers.length;
          device.trace.length = 0;
          const raw = await fitter.synthesize();
          assert.equal(raw.length, input.target.length);
          assert.equal(device.buffers.length, allocated, 'Preview readback reuses the waveform staging buffer.');
          assert.equal(device.trace.filter(event => event.type === 'map').length, 1,
            'Each preview reads back the waveform once.');
          const commands = device.trace.filter(event => event.type === 'submit').flatMap(event => event.commands);
          const dispatches = commands.filter(command => command.type === 'dispatch');
          assert(!dispatches.some(command => /Adam|backward/i.test(command.pipeline)),
            'Preview synthesis must not apply a backward pass or optimizer update.');
          const noise = dispatches.filter(command => command.pipeline === 'Faithful Threefry uniforms');
          assert.equal(noise.length, 1);
          assert.equal(noise[0].bindings.find(binding => binding.binding === 0).hostWrite.words[0], 0,
            'Every preview uses key(0), independent of the next training seed.');
          assert.equal((await fitter.snapshot()).iteration, iteration, 'Synthesis does not advance the optimizer.');
        }
        if (withPreviews) await preview(0);
        for (let iteration = 1; iteration <= 2; iteration++) {
          device.trace.length = 0;
          await fitter.step();
          steps.push(device.trace.filter(event => event.type === 'submit').flatMap(event => event.commands));
          if (withPreviews) { await preview(iteration); await preview(iteration); }
        }
      });
      return steps;
    }
    assert.deepEqual(await trainingTraces(true), await trainingTraces(false),
      'Interleaved fixed-key previews must preserve subsequent dispatches, bindings, RNG seeds and Adam iterations.');
    cases.push('interleaved synthesis previews reuse readback storage and preserve training RNG/Adam command state');
  }

  for (let boundary = 1; boundary <= 4; boundary++) {
    const controller = new AbortController(), device = fakeDevice();
    await withFitter(device, async fitter => {
      device.trace.length = 0;
      let waits = 0;
      device.onWorkDone = () => { if (++waits === boundary) controller.abort(); };
      await assert.rejects(fitter.step(), error => error.name === 'AbortError' && error.code !== 'WEBGPU_DEVICE_LOST');
      const submissions = device.trace.filter(event => event.type === 'submit');
      assert.equal(submissions.length, boundary, 'Cancellation must prevent submission of later stages.');
      assert.equal(device.trace.filter(event => event.type === 'wait').length, boundary);
      assert(!device.trace.some(event => event.type === 'map'), 'Canceled partial objectives must not publish scalar progress.');
      assert(!submissions.flatMap(event => event.commands).some(command => command.pipeline === 'Faithful Optax Adam'),
        'Cancellation before the final stage must not apply an optimizer update.');
    }, {engineOptions: {signal: controller.signal}});
    cases.push(`cancellation after stage ${boundary} prevents subsequent GPU work and Adam`);
  }

  {
    const device = fakeDevice();
    await withFitter(device, async fitter => {
      device.trace.length = 0;
      let waits = 0;
      device.onWorkDone = async () => {
        if (++waits !== 2) return;
        setTimeout(() => device.lose({reason: 'unknown', message: 'Injected staged GPU reset.'}), 0);
        throw new DOMException('GPU completion was interrupted.', 'AbortError');
      };
      await assert.rejects(fitter.step(), error => {
        assert.equal(error.code, 'WEBGPU_DEVICE_LOST');
        assert.match(error.message, /Injected staged GPU reset/);
        return true;
      });
      assert.equal(device.trace.filter(event => event.type === 'submit').length, 2,
        'A lost device must not receive further objective stages.');
      assert(!device.trace.some(event => event.type === 'map'));
    });
    cases.push('device loss during a stage-completion wait is classified and stops later submissions');
  }
} finally {
  for (const [name, descriptor] of originalGlobals) {
    if (descriptor) Object.defineProperty(globalThis, name, descriptor);
    else delete globalThis[name];
  }
}

console.log(JSON.stringify({passed: true, scope: 'GPU resource ownership, scheduling and error handling only; no numerical GPU claims', cases}, null, 2));
