/* PaperCheckAI — VoiceOut: reliable text-to-speech for AI answers (Web Speech API).
 * Source of truth for the engine embedded in mentor_app.html (run scripts/sync_voice.py after editing).
 *
 * Fixes the "sometimes speaks, sometimes silent" behaviour:
 *  - waits for voices that load late (voiceschanged) instead of choosing before they exist
 *  - speaks ONE chunk at a time (no 20 queued utterances), with a generation token so audio never overlaps
 *  - Chrome drops an utterance spoken right after cancel(): small settle delay
 *  - watchdog: if an utterance never starts, cancel + retry; real errors retry (voice -> language -> default)
 *  - pause/resume is "soft" (remember sentence, cancel, replay it): identical on Chrome/Edge/Firefox/Android
 *  - every state is reported to the UI; text answers are never affected by voice failures
 * No fake audio: if the browser cannot speak, state becomes "unsupported"/"error" with a reason.
 */
(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.PCVoiceOut = factory();
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  var MAX_CHUNK = 180, START_TIMEOUT = 4000, SETTLE_MS = 80, MAX_ATTEMPTS = 3, VOICE_WAIT = 1500;

  function chunkText(text, max) {
    max = max || MAX_CHUNK;
    var sentences = String(text || "").replace(/\s+/g, " ").trim().match(/[^.!?\u0964\u0965\n]+[.!?\u0964\u0965]*/g) || [];
    var out = [], buf = "";
    function pushLong(s) {                       // hard-split a very long sentence at commas, then spaces
      while (s.length > max) {
        var cut = s.lastIndexOf(",", max); if (cut < max * 0.4) cut = s.lastIndexOf(" ", max); if (cut < 1) cut = max;
        out.push(s.slice(0, cut + 1).trim()); s = s.slice(cut + 1);
      }
      if (s.trim()) out.push(s.trim());
    }
    sentences.forEach(function (s) {
      s = s.trim(); if (!s) return;
      if (s.length > max) { if (buf) { out.push(buf); buf = ""; } pushLong(s); return; }
      if ((buf + " " + s).trim().length > max) { out.push(buf); buf = s; } else buf = (buf + " " + s).trim();
    });
    if (buf.trim()) out.push(buf.trim());
    return out.filter(Boolean);
  }

  var DEVANAGARI = /[\u0900-\u097F]/;
  /** Language tag to speak in. Roman Hinglish is read by an Indian-English voice (hi-IN voices mangle Latin text). */
  function pickLang(text, opts) {
    opts = opts || {};
    if (opts.lang) return opts.lang;
    if (opts.replyLang === "hindi") return "hi-IN";
    if (opts.replyLang === "english" || opts.replyLang === "hinglish") return "en-IN";
    if (DEVANAGARI.test(text || "")) return "hi-IN";
    return "en-IN";
  }

  /** {voice, lang, missing}. missing=true -> a NON-Latin language with no installed voice (must not fake it). */
  function chooseVoice(voices, lang, preferredURI) {
    voices = voices || [];
    var base = String(lang).toLowerCase().split(/[-_]/)[0], i;
    if (preferredURI) for (i = 0; i < voices.length; i++) if (voices[i].voiceURI === preferredURI) return { voice: voices[i], lang: voices[i].lang, missing: false };
    var exact = voices.filter(function (v) { return v.lang && v.lang.toLowerCase().replace("_", "-") === String(lang).toLowerCase(); });
    var same = voices.filter(function (v) { return v.lang && v.lang.toLowerCase().split(/[-_]/)[0] === base; });
    var pool = exact.length ? exact : same;
    var v = pool.filter(function (x) { return x.localService; })[0] || pool[0] || null;
    if (v) return { voice: v, lang: v.lang, missing: false };
    var needsNative = base !== "en";           // English falls back to the browser default; Hindi etc. cannot
    return { voice: null, lang: lang, missing: needsNative && voices.length > 0 };
  }

  function create(env) {
    env = env || {};
    var synth = env.synth, U = env.Utterance;
    var setT = env.setTimeout || setTimeout, clrT = env.clearTimeout || clearTimeout;
    var onState = env.onState || function () {};
    var supported = !!(synth && U);

    var gen = 0, st = { state: supported ? "idle" : "unsupported", id: null, chunk: 0, total: 0, attempt: 0, message: supported ? "" : "This browser cannot read text aloud (try Chrome or Edge). The written answer is unaffected." };
    var chunks = [], idx = 0, cur = null, opts = {}, timers = [];

    function emit(patch) { for (var k in patch) st[k] = patch[k]; try { onState(JSON.parse(JSON.stringify(st))); } catch (e) {} }
    function later(fn, ms) { var t = setT(fn, ms); timers.push(t); return t; }
    function clearTimers() { timers.forEach(function (t) { clrT(t); }); timers = []; }
    function getVoices() { try { return (env.getVoices ? env.getVoices() : synth.getVoices()) || []; } catch (e) { return []; } }
    function hardCancel() { try { if (synth.paused) synth.resume(); } catch (e) {} try { synth.cancel(); } catch (e) {} }

    /** resolve once voices exist (they load asynchronously, esp. Chrome) or after a short timeout */
    function ensureVoices(cb) {
      if (getVoices().length) return cb();
      emit({ state: "loading", message: "Loading voices…" });
      var done = false;
      function fin() { if (done) return; done = true; try { synth.removeEventListener && synth.removeEventListener("voiceschanged", fin); } catch (e) {} cb(); }
      try { synth.addEventListener && synth.addEventListener("voiceschanged", fin); } catch (e) {}
      later(fin, VOICE_WAIT);
    }

    function fail(g, msg) {
      if (g !== gen) return;
      clearTimers(); hardCancel();
      emit({ state: "error", message: msg });
    }

    function playChunk(g) {
      if (g !== gen) return;
      if (idx >= chunks.length) { clearTimers(); emit({ state: "idle", message: "", chunk: 0, total: 0, attempt: 0 }); return; }
      var text = chunks[idx];
      var lang = pickLang(text, opts);
      var attempt = st.attempt;
      // attempt 0: chosen voice; attempt 1: language only (no explicit voice); attempt 2: bare default utterance
      var pick = chooseVoice(getVoices(), lang, opts.voiceURI);
      if (pick.missing) { return fail(g, "No " + lang + " voice is installed on this device, so this answer can't be read aloud in that language. The text answer is shown. Install a voice in your system settings or pick another voice."); }
      var u;
      try { u = new U(text); } catch (e) { return fail(g, "Speech could not start (" + e + "). The text answer is unaffected."); }
      u.lang = pick.lang || lang; u.rate = opts.rate || 1; u.pitch = 1; u.volume = 1;
      if (pick.voice && attempt === 0) u.voice = pick.voice;
      var started = false, finished = false;
      cur = u;
      var watchdog = later(function () {
        if (g !== gen || started || finished) return;
        retry(g, "Speech did not start");
      }, START_TIMEOUT);
      u.onstart = function () { if (g !== gen) return; started = true; clrT(watchdog); emit({ state: "speaking", message: "", chunk: idx + 1, total: chunks.length }); };
      u.onend = function () {
        if (g !== gen || finished) return;
        finished = true; clrT(watchdog); idx++; st.attempt = 0; playChunk(g);
      };
      u.onerror = function (ev) {
        if (g !== gen || finished) return;
        var err = ev && ev.error;
        if (err === "interrupted" || err === "canceled") return;       // caused by our own cancel()
        finished = true; clrT(watchdog);
        if (err === "not-allowed") return fail(g, "The browser blocked audio until you interact with the page. Press Speak again.");
        retry(g, "Speech error" + (err ? " (" + err + ")" : ""));
      };
      try { synth.speak(u); } catch (e) { retry(g, "speak() failed"); }
    }

    function retry(g, reason) {
      if (g !== gen) return;
      var next = st.attempt + 1;
      if (next >= MAX_ATTEMPTS) return fail(g, reason + " — retried " + (MAX_ATTEMPTS - 1) + " times. The text answer is unaffected; press Speak to try again.");
      st.attempt = next;
      emit({ state: "retrying", message: reason + " — retrying (" + next + "/" + (MAX_ATTEMPTS - 1) + ")…", attempt: next });
      hardCancel();
      later(function () { playChunk(g); }, SETTLE_MS * 2);
    }

    function start(from) {
      var g = ++gen;
      clearTimers(); hardCancel();
      st.attempt = 0; idx = from || 0;
      ensureVoices(function () {
        if (g !== gen) return;
        later(function () { playChunk(g); }, SETTLE_MS);                 // let cancel() settle (Chrome drops instant speak)
      });
    }

    return {
      get supported() { return supported; },
      get state() { return JSON.parse(JSON.stringify(st)); },
      chunkText: chunkText,
      speak: function (text, o) {
        if (!supported) { emit({ state: "unsupported" }); return false; }
        chunks = chunkText(text); opts = o || {};
        if (!chunks.length) { emit({ state: "error", message: "Nothing to read aloud." }); return false; }
        emit({ state: "loading", id: opts.id || null, chunk: 0, total: chunks.length, message: "Preparing voice…" });
        start(0); return true;
      },
      pause: function () {
        if (!supported || (st.state !== "speaking" && st.state !== "retrying" && st.state !== "loading")) return false;
        gen++; clearTimers(); hardCancel();                              // soft pause: replay the current sentence on resume
        emit({ state: "paused", message: "Paused" }); return true;
      },
      resume: function () {
        if (!supported || st.state !== "paused") return false;
        emit({ state: "loading", message: "Resuming…" }); start(idx); return true;
      },
      stop: function () {
        gen++; clearTimers(); if (supported) hardCancel(); idx = 0; chunks = [];
        if (supported) emit({ state: "idle", id: null, chunk: 0, total: 0, attempt: 0, message: "" });
      },
      /** the page calls this when the voice picker / reply language changes mid-playback */
      setOptions: function (o) { for (var k in o) opts[k] = o[k]; }
    };
  }

  return { create: create, chunkText: chunkText, pickLang: pickLang, chooseVoice: chooseVoice,
           constants: { MAX_CHUNK: MAX_CHUNK, START_TIMEOUT: START_TIMEOUT, SETTLE_MS: SETTLE_MS, MAX_ATTEMPTS: MAX_ATTEMPTS, VOICE_WAIT: VOICE_WAIT } };
});
