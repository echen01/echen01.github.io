const MIME_TYPES = ['audio/webm;codecs=opus', 'audio/ogg;codecs=opus', 'audio/mp4'];

function permissionMessage(error) {
  switch (error?.name) {
    case 'NotAllowedError':
    case 'SecurityError':
      return 'Microphone access was denied. Allow microphone access in your browser and try again.';
    case 'NotFoundError':
    case 'DevicesNotFoundError':
      return 'No microphone was found. Connect a microphone and try again.';
    case 'NotReadableError':
    case 'TrackStartError':
      return 'The microphone could not be opened. Close other apps using it and try again.';
    default:
      return 'The microphone could not be started. Check your microphone and browser permissions, then try again.';
  }
}

/** Records a short microphone clip; permission is requested only by start(). */
export function createMicrophoneRecorder({
  maxSeconds = 10,
  maxBytes = 20 * 1024 * 1024,
  onState = () => {},
  onTick = () => {},
  onComplete = () => {},
  onError = () => {},
  getUserMedia = globalThis.navigator?.mediaDevices?.getUserMedia?.bind(globalThis.navigator.mediaDevices),
  MediaRecorder = globalThis.MediaRecorder,
  Blob = globalThis.Blob,
  now = () => globalThis.performance.now(),
  setTimeout = globalThis.setTimeout.bind(globalThis),
  clearTimeout = globalThis.clearTimeout.bind(globalThis),
  setInterval = globalThis.setInterval.bind(globalThis),
  clearInterval = globalThis.clearInterval.bind(globalThis),
  pageTarget = globalThis.window,
} = {}) {
  if (!(Number.isFinite(maxSeconds) && maxSeconds > 0)) throw new RangeError('maxSeconds must be positive.');
  if (!(Number.isFinite(maxBytes) && maxBytes > 0)) throw new RangeError('maxBytes must be positive.');
  const supported = typeof getUserMedia === 'function' && typeof MediaRecorder === 'function' && typeof Blob === 'function';
  let active = null;
  let disposed = false;

  function stopTracks(session) {
    for (const [track, listener] of session.trackListeners) track.removeEventListener?.('ended', listener);
    session.trackListeners.length = 0;
    for (const track of session.stream?.getTracks() ?? []) {
      try { track.stop(); } catch { /* Release every track even if one fails. */ }
    }
    session.stream = null;
  }

  function clearTimers(session) {
    if (session.interval !== null) clearInterval(session.interval);
    if (session.timeout !== null) clearTimeout(session.timeout);
    session.interval = session.timeout = null;
  }

  function detachRecorder(session) {
    if (!session.recorder) return;
    session.recorder.ondataavailable = null;
    session.recorder.onerror = null;
    session.recorder.onstop = null;
  }

  function discard(session, message) {
    if (active !== session) return;
    active = null;
    clearTimers(session);
    detachRecorder(session);
    // Invalidate first: synchronous or queued recorder events must not revive a
    // cancelled clip, including when a new permission request is already open.
    try {
      if (session.recorder && session.recorder.state !== 'inactive') session.recorder.stop();
    } catch { /* Tracks still need releasing if the recorder has failed. */ }
    stopTracks(session);
    session.chunks.length = 0;
    onState('idle');
    if (message) onError(message);
  }

  function finish(session) {
    if (active !== session) return;
    active = null;
    clearTimers(session);
    detachRecorder(session);
    stopTracks(session);
    const blob = new Blob(session.chunks, {type: session.recorder.mimeType || session.chunks[0]?.type || ''});
    session.chunks.length = 0;
    onState('idle');
    if (blob.size) onComplete(blob);
    else onError('The recording was empty. Record a short phrase and try again.');
  }

  function elapsed(session) {
    return Math.min(maxSeconds, Math.max(0, (now() - session.startedAt) / 1000));
  }

  function stop() {
    const session = active;
    if (!session || session.stopping) return;
    if (!session.recorder) {
      discard(session);
      return;
    }
    session.stopping = true;
    clearTimers(session);
    onTick(elapsed(session));
    onState('stopping');
    try {
      if (session.recorder.state !== 'inactive') session.recorder.stop();
      // Release hardware immediately rather than waiting for the final Blob.
      stopTracks(session);
    } catch {
      discard(session, 'The recording could not be saved. Please try recording again.');
    }
  }

  async function start() {
    if (disposed || active) return;
    if (!supported) {
      onError('Microphone recording is unavailable in this browser. Open the page on localhost or HTTPS, or upload an audio file.');
      return;
    }
    const session = {stream: null, recorder: null, trackListeners: [], chunks: [], bytes: 0,
      interval: null, timeout: null, startedAt: 0, stopping: false};
    active = session;
    onState('requesting');
    let stream;
    try {
      stream = await getUserMedia({audio: {channelCount: 1, echoCancellation: false,
        noiseSuppression: false, autoGainControl: false}, video: false});
    } catch (error) {
      discard(session, permissionMessage(error));
      return;
    }
    session.stream = stream;
    if (active !== session) {
      stopTracks(session);
      return;
    }
    const audioTracks = stream.getAudioTracks();
    if (!audioTracks.length || audioTracks.every(track => track.readyState === 'ended')) {
      discard(session, 'No active microphone was found. Connect a microphone and try again.');
      return;
    }
    try {
      const mimeType = MIME_TYPES.find(type => MediaRecorder.isTypeSupported?.(type));
      const recorder = new MediaRecorder(stream, mimeType ? {mimeType} : undefined);
      session.recorder = recorder;
      recorder.ondataavailable = event => {
        if (active !== session || !event.data?.size) return;
        session.bytes += event.data.size;
        if (session.bytes > maxBytes) {
          discard(session, 'The recording is too large. Please record a shorter sample.');
          return;
        }
        session.chunks.push(event.data);
      };
      recorder.onerror = () => discard(session, 'The microphone recording failed. Please try recording again.');
      recorder.onstop = () => finish(session);
      for (const track of audioTracks) {
        const ended = () => discard(session, 'The microphone disconnected. Reconnect it and record again.');
        track.addEventListener?.('ended', ended);
        session.trackListeners.push([track, ended]);
      }
      recorder.start(250);
      // A recorder may dispatch a failure synchronously from start().
      if (active !== session) return;
      session.startedAt = now();
      onState('recording');
      onTick(0);
      session.interval = setInterval(() => {
        if (active === session && !session.stopping) onTick(elapsed(session));
      }, 100);
      session.timeout = setTimeout(() => {
        if (active === session) stop();
      }, maxSeconds * 1000);
    } catch {
      discard(session, 'This browser could not start audio recording. Try another browser or upload an audio file.');
    }
  }

  function cancel() {
    if (active) discard(active);
  }

  function dispose() {
    if (disposed) return;
    disposed = true;
    cancel();
    pageTarget?.removeEventListener?.('pagehide', cancel);
  }

  pageTarget?.addEventListener?.('pagehide', cancel);
  return {supported, start, stop, cancel, dispose};
}
