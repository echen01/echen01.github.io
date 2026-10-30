import assert from 'node:assert/strict';
import {createMicrophoneRecorder} from './microphone-recorder.js';

const deferred = () => {
  let resolve, reject;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  return {promise, resolve, reject};
};

function clock() {
  let time = 0, sequence = 0;
  const tasks = new Map();
  const schedule = (callback, delay, repeat = 0) => {
    const id = ++sequence;
    tasks.set(id, {callback, at: time + delay, repeat});
    return id;
  };
  return {
    now: () => time,
    setTimeout: (fn, delay) => schedule(fn, delay),
    clearTimeout: id => tasks.delete(id),
    setInterval: (fn, delay) => schedule(fn, delay, delay),
    clearInterval: id => tasks.delete(id),
    get pending() { return tasks.size; },
    advance(ms) {
      const until = time + ms;
      for (;;) {
        const next = [...tasks].filter(([, task]) => task.at <= until).sort((a, b) => a[1].at - b[1].at)[0];
        if (!next) break;
        const [id, task] = next;
        time = task.at;
        if (task.repeat) task.at += task.repeat;
        else tasks.delete(id);
        task.callback();
      }
      time = until;
    },
  };
}

function stream() {
  const track = new EventTarget();
  track.readyState = 'live';
  track.stops = 0;
  track.stop = () => {
    track.stops++;
    track.readyState = 'ended';
    // Native stop() does not emit ended. Emitting it here also proves our own
    // cleanup cannot be mistaken for the microphone unexpectedly disconnecting.
    track.dispatchEvent(new Event('ended'));
  };
  return {track, getTracks: () => [track], getAudioTracks: () => [track]};
}

function rig(options = {}) {
  const time = clock(), requests = [], recorders = [], states = [], ticks = [], clips = [], errors = [];
  const pageTarget = new EventTarget();
  class Recorder {
    static isTypeSupported(type) { return (options.types ?? ['audio/webm;codecs=opus']).includes(type); }
    constructor(input, config) {
      if (options.constructorThrows) throw new Error('unsupported');
      this.stream = input;
      this.config = config;
      this.mimeType = config?.mimeType ?? 'audio/fallback';
      this.state = 'inactive';
      this.stops = 0;
      recorders.push(this);
    }
    start(timeslice) {
      if (options.startThrows) throw new Error('start failed');
      this.timeslice = timeslice;
      this.state = 'recording';
      if (options.startError) this.onerror({error: new Error('sync recording error')});
    }
    stop() {
      this.stops++;
      if (options.stopThrows) throw new Error('stop failed');
      this.state = 'inactive';
      // Keep captured handlers to simulate events already queued before cancel.
      const data = this.ondataavailable, done = this.onstop;
      this.flushStop = (bytes = 'end') => {
        data?.({data: new Blob([bytes], {type: this.mimeType})});
        done?.();
      };
    }
    data(bytes) { this.ondataavailable?.({data: new Blob([bytes], {type: this.mimeType})}); }
  }
  const recorder = createMicrophoneRecorder({
    ...time,
    MediaRecorder: options.unsupported ? undefined : Recorder,
    getUserMedia: constraints => {
      const request = {...deferred(), constraints};
      requests.push(request);
      return request.promise;
    },
    pageTarget,
    maxSeconds: options.maxSeconds ?? 10,
    maxBytes: options.maxBytes ?? 20 * 1024 * 1024,
    onState: state => states.push(state), onTick: seconds => ticks.push(seconds),
    onComplete: clip => clips.push(clip), onError: message => errors.push(message),
  });
  return {recorder, requests, recorders, states, ticks, clips, errors, time, pageTarget,
    async begin(input = stream()) {
      const pending = recorder.start();
      requests.at(-1).resolve(input);
      await pending;
      return {input, capture: recorders.at(-1)};
    },
  };
}

// No permission request at construction; duplicate starts share no new request.
{
  const r = rig();
  assert.equal(r.recorder.supported, true);
  assert.equal(r.requests.length, 0);
  const pending = r.recorder.start();
  await r.recorder.start();
  assert.equal(r.requests.length, 1);
  assert.deepEqual(r.requests[0].constraints, {audio: {channelCount: 1,
    echoCancellation: false, noiseSuppression: false, autoGainControl: false}, video: false});
  const input = stream(); r.requests[0].resolve(input); await pending;
  const capture = r.recorders[0];
  assert.equal(capture.timeslice, 250);
  assert.equal(capture.config.mimeType, 'audio/webm;codecs=opus');
  capture.data('hello');
  r.time.advance(3250);
  r.recorder.stop(); r.recorder.stop();
  assert.equal(capture.stops, 1);
  assert.equal(input.track.stops, 1, 'Hardware is released before codec finalization.');
  assert.equal(r.clips.length, 0);
  assert.equal(r.ticks.at(-1), 3.25);
  assert.equal(r.time.pending, 0);
  capture.flushStop(' there');
  assert.equal(await r.clips[0].text(), 'hello there');
  assert.equal(r.clips[0].type, 'audio/webm;codecs=opus');
  assert.deepEqual(r.states, ['requesting', 'recording', 'stopping', 'idle']);
  assert.deepEqual(r.errors, []);
  r.recorder.dispose();
}

// Automatic duration limit, then a fresh recording without stale chunks.
{
  const r = rig();
  const first = await r.begin();
  r.time.advance(9999); assert.equal(first.capture.stops, 0);
  r.time.advance(1); assert.equal(first.capture.stops, 1);
  assert.equal(first.input.track.stops, 1);
  assert.equal(r.ticks.at(-1), 10);
  first.capture.flushStop('first');
  const second = await r.begin();
  r.recorder.stop(); second.capture.flushStop('second');
  assert.deepEqual(await Promise.all(r.clips.map(clip => clip.text())), ['first', 'second']);
  assert.equal(r.time.pending, 0);
  r.recorder.dispose();
}

// Cancel or page exit while a permission prompt is open: late streams close,
// and a newer recording is not affected by the older request resolving/rejecting.
for (const action of ['cancel', 'stop', 'dispose', 'pagehide']) {
  const r = rig(), pending = r.recorder.start();
  if (action === 'pagehide') r.pageTarget.dispatchEvent(new Event('pagehide'));
  else r.recorder[action]();
  const input = stream(); r.requests[0].resolve(input); await pending;
  assert.equal(input.track.stops, 1, action);
  assert.equal(r.recorders.length, 0);
  assert.equal(r.time.pending, 0);
  assert.deepEqual(r.errors, []); assert.deepEqual(r.clips, []);
  if (action === 'dispose') { await r.recorder.start(); assert.equal(r.requests.length, 1); }
  r.recorder.dispose();
}
for (const staleResult of ['resolve', 'reject']) {
  const r = rig(), oldStart = r.recorder.start();
  r.recorder.cancel();
  const current = await r.begin();
  const oldInput = stream();
  r.requests[0][staleResult](staleResult === 'resolve' ? oldInput : {name: 'NotAllowedError'});
  await oldStart;
  assert.equal(oldInput.track.stops, staleResult === 'resolve' ? 1 : 0);
  assert.equal(current.input.track.stops, 0);
  assert.equal(r.states.at(-1), 'recording');
  assert.deepEqual(r.errors, []);
  r.recorder.stop(); current.capture.flushStop();
  assert.equal(r.clips.length, 1);
  r.recorder.dispose();
}

// Cancellation discards queued final data and codec errors, including after a
// manual Stop; page exit releases the microphone and all recording timers.
for (const action of ['cancel', 'dispose', 'pagehide']) {
  const r = rig(), {input, capture} = await r.begin();
  const error = capture.onerror;
  r.recorder.stop();
  if (action === 'pagehide') r.pageTarget.dispatchEvent(new Event('pagehide'));
  else r.recorder[action]();
  capture.flushStop('discard me'); error({error: new Error('late error')});
  assert.equal(input.track.stops, 1);
  assert.equal(r.time.pending, 0);
  assert.deepEqual(r.errors, []); assert.deepEqual(r.clips, []);
  r.recorder.dispose();
}
{
  const r = rig(), {input, capture} = await r.begin();
  r.pageTarget.dispatchEvent(new Event('pagehide'));
  assert.equal(input.track.stops, 1); assert.equal(capture.stops, 1);
  assert.equal(r.time.pending, 0); assert.deepEqual(r.clips, []);
  r.recorder.dispose();
}

for (const [name, match] of [['NotAllowedError', /denied/], ['NotFoundError', /No microphone/],
  ['NotReadableError', /could not be opened/], ['UnknownError', /could not be started/]]) {
  const r = rig(), pending = r.recorder.start();
  r.requests[0].reject({name}); await pending;
  assert.match(r.errors[0], match); assert.equal(r.states.at(-1), 'idle');
  assert.equal(r.time.pending, 0); assert.deepEqual(r.clips, []);
  r.recorder.dispose();
}

// Negotiate the next supported MIME or let the browser select its own format.
for (const [types, expected] of [[['audio/ogg;codecs=opus'], 'audio/ogg;codecs=opus'],
  [['audio/mp4'], 'audio/mp4'], [[], 'audio/fallback']]) {
  const r = rig({types}), {capture} = await r.begin();
  assert.equal(capture.mimeType, expected);
  r.recorder.dispose();
}
{
  const r = rig({unsupported: true});
  assert.equal(r.recorder.supported, false); await r.recorder.start();
  assert.equal(r.requests.length, 0); assert.match(r.errors[0], /unavailable/);
  r.recorder.dispose();
}
for (const option of ['constructorThrows', 'startThrows', 'startError', 'stopThrows']) {
  const r = rig({[option]: true}), {input} = await r.begin();
  if (option === 'stopThrows') r.recorder.stop();
  assert.equal(input.track.stops, 1, option);
  assert.equal(r.errors.length, 1, option);
  assert.equal(r.time.pending, 0); assert.equal(r.states.at(-1), 'idle');
  assert.deepEqual(r.clips, []); r.recorder.dispose();
}

// Disconnection, recorder failure, oversized clips, and empty recordings all
// report actionable errors and never hand an invalid clip to the fitting UI.
for (const fault of ['ended', 'error', 'too-large', 'empty']) {
  const r = rig({maxBytes: 5}), {input, capture} = await r.begin();
  if (fault === 'ended') input.track.dispatchEvent(new Event('ended'));
  if (fault === 'error') capture.onerror({error: new Error('codec')});
  if (fault === 'too-large') { capture.data('123'); capture.data('456'); }
  if (fault === 'empty') { r.recorder.stop(); capture.flushStop(''); }
  assert.equal(input.track.stops, 1, fault);
  assert.equal(r.errors.length, 1, fault); assert.equal(r.states.at(-1), 'idle');
  assert.equal(r.time.pending, 0); assert.deepEqual(r.clips, []);
  r.recorder.dispose();
}
for (const state of ['missing', 'ended']) {
  const r = rig(), input = stream();
  if (state === 'missing') input.getAudioTracks = () => [];
  else input.track.readyState = 'ended';
  await r.begin(input);
  assert.match(r.errors[0], /No active microphone/);
  assert.equal(input.track.stops, 1); assert.equal(r.recorders.length, 0);
  r.recorder.dispose();
}

console.log('Microphone recorder: permission, lifecycle, duration/size limits, MIME negotiation, stale events, and cleanup checks passed.');
