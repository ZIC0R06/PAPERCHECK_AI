"""The voice engine in mentor_app.html must be byte-identical to frontend/voice_out.js (single source of truth),
and every assistant message must carry the Play/Pause/Stop controls and a live status element."""
import os
import re
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
read = lambda *p: open(os.path.join(ROOT, *p), encoding="utf-8").read()    # noqa: E731


class TestVoiceEmbed(unittest.TestCase):
    def test_embedded_engine_matches_source(self):
        html, js = read("frontend", "mentor_app.html"), read("frontend", "voice_out.js").strip()
        m = re.search(r"/\*VOICE_OUT_BEGIN\*/\n(.*?)\n/\*VOICE_OUT_END\*/", html, re.S)
        self.assertIsNotNone(m, "markers missing")
        self.assertEqual(m.group(1), js, "run: python scripts/sync_voice.py")

    def test_every_answer_has_voice_controls_and_status(self):
        html = read("frontend", "mentor_app.html")
        for t in ('data-t="play"', 'data-t="pause"', 'data-t="stop"', 'data-t="vstat"'):
            self.assertIn(t, html)
        self.assertIn("function paintVoice", html)
        self.assertIn("PCVoiceOut.create", html)

    def test_old_fragile_playback_is_gone(self):
        html = read("frontend", "mentor_app.html")
        self.assertNotIn("chunks.forEach((c,idx)=>{", html)       # queued every chunk at once
        self.assertNotIn("synth.pause()", html.split("/*VOICE_OUT_END*/")[1])


if __name__ == "__main__":
    unittest.main()
