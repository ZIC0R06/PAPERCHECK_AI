// Run: node --test tests/js/        (Node 18+).  Deterministic: fake speech engine + fake clock.
const test = require("node:test");
const assert = require("node:assert/strict");
const V = require("../../frontend/voice_out.js");

function clock() {
  let now = 0, id = 0, q = [];
  return {
    setTimeout(fn, ms) { q.push({ id: ++id, at: now + ms, fn }); return id; },
    clearTimeout(i) { q = q.filter(t => t.id !== i); },
    advance(ms) { const end = now + ms; for (;;) { q.sort((a, b) => a.at - b.at); const t = q[0]; if (!t || t.at > end) break; q.shift(); now = t.at; t.fn(); } now = end; },
  };
}
/** Fake window.speechSynthesis. mode: "ok" | "silent" (never starts) | "error" (errors N times) */
function fakeSynth(opts = {}) {
  const s = { voices: opts.voices || [], active: [], spoken: [], cancels: 0, listeners: {}, paused: false, mode: opts.mode || "ok", errorsLeft: opts.errors || 0, maxActive: 0, autoEnd: opts.autoEnd !== false, errorName: opts.errorName || "synthesis-failed" };
  s.getVoices = () => s.voices;
  s.addEventListener = (n, f) => { (s.listeners[n] = s.listeners[n] || []).push(f); };
  s.removeEventListener = (n, f) => { s.listeners[n] = (s.listeners[n] || []).filter(x => x !== f); };
  s.fire = n => (s.listeners[n] || []).slice().forEach(f => f());
  s.resume = () => { s.paused = false; };
  s.cancel = () => { s.cancels++; const a = s.active; s.active = []; a.forEach(u => u.onerror && u.onerror({ error: "interrupted" })); };
  s.speak = u => {
    s.active.push(u); s.spoken.push(u); s.maxActive = Math.max(s.maxActive, s.active.length);
    if (s.mode === "silent") return;
    if (s.errorsLeft > 0) { s.errorsLeft--; s.active = s.active.filter(x => x !== u); setImmediate(() => u.onerror && u.onerror({ error: s.errorName })); return; }
    u.onstart && setImmediate(() => u.onstart());
  };
  s.finishCurrent = () => { const u = s.active.shift(); if (u && u.onend) u.onend(); };
  return s;
}
class FakeUtterance { constructor(t) { this.text = t; } }
const EN = { voiceURI: "en-in", lang: "en-IN", name: "English India", localService: true };
const HI = { voiceURI: "hi-in", lang: "hi-IN", name: "Hindi", localService: true };
const S1 = "First sentence is intentionally long enough to occupy its own chunk of speech output here today.";
const S2 = "Second sentence is also long enough to occupy its own separate chunk of speech output right now.";
const S3 = "Third sentence is again long enough to occupy its own separate chunk of speech output as well.";
const flush = () => new Promise(r => setImmediate(r));

function make(synthOpts) {
  const ck = clock(), synth = fakeSynth(synthOpts), states = [];
  const vo = V.create({ synth, Utterance: FakeUtterance, setTimeout: ck.setTimeout, clearTimeout: ck.clearTimeout, onState: s => states.push(s) });
  return { ck, synth, states, vo, last: () => states[states.length - 1] };
}
async function run(ctx, ms) { ctx.ck.advance(ms); await flush(); }

test("chunking: bounded size, no empty chunks, Hindi danda splits", () => {
  const long = ("This is a fairly long sentence without any terminal punctuation but with, some commas, " ).repeat(8);
  const c = V.chunkText(long + " Short end. और यह हिंदी वाक्य है। दूसरा वाक्य।");
  assert.ok(c.length >= 4); c.forEach(x => { assert.ok(x.length <= V.constants.MAX_CHUNK + 1, x.length); assert.ok(x.trim()); });
  assert.equal(V.chunkText("   ").length, 0);
});

test("language: Hinglish and English use an English voice, Devanagari uses Hindi", () => {
  assert.equal(V.pickLang("Mujhe stack samjhao", { replyLang: "hinglish" }), "en-IN");
  assert.equal(V.pickLang("What is a stack?", {}), "en-IN");
  assert.equal(V.pickLang("स्टैक क्या है", {}), "hi-IN");
  assert.equal(V.pickLang("hello", { replyLang: "hindi" }), "hi-IN");
});

test("plays chunks sequentially, one at a time, then returns to idle", async () => {
  const c = make({ voices: [EN] });
  c.vo.speak(S1 + " " + S2 + " " + S3, { id: "m1" });
  await run(c, 200);
  assert.equal(c.synth.active.length, 1);                 // only ONE utterance queued
  for (let i = 0; i < 3; i++) { await run(c, 10); assert.equal(c.last().state, "speaking"); c.synth.finishCurrent(); await run(c, 200); }
  assert.equal(c.last().state, "idle");
  assert.equal(c.synth.maxActive, 1);
});

test("voices that load late: waits for voiceschanged instead of speaking silently", async () => {
  const c = make({ voices: [] });
  c.vo.speak("Hello there.", {});
  await run(c, 50);
  assert.equal(c.last().state, "loading");
  assert.equal(c.synth.spoken.length, 0);
  c.synth.voices = [EN]; c.synth.fire("voiceschanged");
  await run(c, 200);
  assert.equal(c.synth.spoken.length, 1);
  assert.equal(c.synth.spoken[0].voice, EN);               // the proper voice was selected
});

test("no voices at all after the wait: English falls back to browser default (still speaks)", async () => {
  const c = make({ voices: [] });
  c.vo.speak("Hello there.", {});
  await run(c, V.constants.VOICE_WAIT + 300);
  assert.equal(c.synth.spoken.length, 1);
  assert.equal(c.synth.spoken[0].voice, undefined);
});

test("a second speak() cancels the first: audio never overlaps", async () => {
  const c = make({ voices: [EN] });
  c.vo.speak(S1 + " " + S2, { id: "a" }); await run(c, 200);
  c.vo.speak("Answer two.", { id: "b" }); await run(c, 400);
  assert.equal(c.synth.maxActive, 1);
  assert.equal(c.synth.spoken[c.synth.spoken.length - 1].text, "Answer two.");
  assert.equal(c.last().id, "b");
});

test("synthesis error → automatic retry → success", async () => {
  const c = make({ voices: [EN], errors: 1 });
  c.vo.speak("Retry me.", {}); await run(c, 200); await run(c, 10);
  assert.ok(c.states.some(s => s.state === "retrying"));
  await run(c, 500);
  assert.equal(c.last().state, "speaking");
  c.synth.finishCurrent(); await run(c, 100);
  assert.equal(c.last().state, "idle");
});

test("persistent errors → clear error state with reason (text unaffected)", async () => {
  const c = make({ voices: [EN], errors: 99 });
  c.vo.speak("Never works.", {});
  for (let i = 0; i < 8; i++) await run(c, 400);
  assert.equal(c.last().state, "error");
  assert.match(c.last().message, /text answer is unaffected/i);
  assert.equal(c.synth.active.length, 0);
});

test("watchdog: an utterance that never starts is cancelled and retried", async () => {
  const c = make({ voices: [EN], mode: "silent" });
  c.vo.speak("Stuck engine.", {});
  await run(c, 300);
  assert.equal(c.synth.spoken.length, 1);
  await run(c, V.constants.START_TIMEOUT + 50);
  assert.ok(c.states.some(s => s.state === "retrying"));
  for (let i = 0; i < 4; i++) await run(c, V.constants.START_TIMEOUT + 400);
  assert.equal(c.last().state, "error");
  assert.ok(c.synth.spoken.length >= 3);
});

test("browser autoplay block (not-allowed) is reported, not retried forever", async () => {
  const c = make({ voices: [EN], errors: 1, errorName: "not-allowed" });
  c.vo.speak("Blocked.", {}); await run(c, 300); await run(c, 50);
  assert.equal(c.last().state, "error");
  assert.match(c.last().message, /blocked audio/i);
});

test("pause and resume continue from the current sentence; stop resets", async () => {
  const c = make({ voices: [EN] });
  c.vo.speak(S1 + " " + S2 + " " + S3, {}); await run(c, 200); await run(c, 10);
  c.synth.finishCurrent(); await run(c, 200); await run(c, 10);      // now playing "Two."
  assert.ok(c.vo.pause()); assert.equal(c.last().state, "paused");
  assert.equal(c.synth.active.length, 0);
  assert.ok(c.vo.resume()); await run(c, 300); await run(c, 10);
  assert.equal(c.synth.spoken[c.synth.spoken.length - 1].text, S2);   // replays the interrupted sentence
  assert.equal(c.last().state, "speaking");
  c.vo.stop(); await run(c, 100);
  assert.equal(c.last().state, "idle"); assert.equal(c.synth.active.length, 0);
  assert.equal(c.vo.resume(), false);
});

test("pause during loading/stop in the middle leaves no stray audio", async () => {
  const c = make({ voices: [EN] });
  c.vo.speak("Alpha. Beta.", {}); c.vo.stop(); await run(c, 500);
  assert.equal(c.synth.spoken.length, 0);
});

test("Hindi requested but no Hindi voice installed: honest error, nothing spoken", async () => {
  const c = make({ voices: [EN] });
  c.vo.speak("स्टैक एक डेटा संरचना है।", {}); await run(c, 400);
  assert.equal(c.last().state, "error"); assert.match(c.last().message, /No hi-IN voice/);
  assert.equal(c.synth.spoken.length, 0);
});

test("Hindi voice present: Devanagari text is spoken with it", async () => {
  const c = make({ voices: [EN, HI] });
  c.vo.speak("स्टैक एक डेटा संरचना है।", {}); await run(c, 400);
  assert.equal(c.synth.spoken[0].voice, HI);
});

test("user-picked voice wins", async () => {
  const other = { voiceURI: "x", lang: "en-GB", name: "UK", localService: true };
  const c = make({ voices: [EN, other] });
  c.vo.speak("Hello.", { voiceURI: "x" }); await run(c, 300);
  assert.equal(c.synth.spoken[0].voice, other);
});

test("unsupported browser: clear state, no crash", () => {
  const states = []; const vo = V.create({ onState: s => states.push(s) });
  assert.equal(vo.supported, false); assert.equal(vo.speak("hi"), false);
  assert.equal(vo.state.state, "unsupported"); vo.stop(); vo.pause();
});
