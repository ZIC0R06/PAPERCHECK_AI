"""Reply-language rules for the AI Mentor (explicit selection > latest message; history/docs never decide)."""
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from backend.language_policy import (detect_language, language_instruction, offline_notice, reply_matches,
                                     resolve_language)

EN = "What is the difference between a stack and a queue?"
HI = "स्टैक और क्यू में क्या अंतर है?"
HING = "Mujhe stack aur queue ka difference samjhao"


class TestResolve(unittest.TestCase):
    def test_auto_english(self):
        self.assertEqual(resolve_language("auto", EN).code, "en")

    def test_auto_hindi(self):
        self.assertEqual(resolve_language("auto", HI).code, "hi")

    def test_auto_hinglish(self):
        self.assertEqual(resolve_language("auto", HING).code, "hinglish")

    def test_explicit_english_beats_any_question_language(self):
        for q in (EN, HI, HING):
            l = resolve_language("English", q)
            self.assertEqual((l.code, l.explicit), ("en", True))
        self.assertEqual(resolve_language("English (United States)", HI).code, "en")

    def test_explicit_hindi(self):
        for q in (EN, HI, HING):
            self.assertEqual(resolve_language("Hindi", q).code, "hi")

    def test_explicit_hinglish(self):
        for q in (EN, HI, HING):
            self.assertEqual(resolve_language("Hinglish", q).code, "hinglish")

    def test_explicit_other_language(self):
        l = resolve_language("Tamil", EN)
        self.assertEqual((l.code, l.name), ("other", "Tamil"))

    def test_english_words_that_used_to_trigger_hinglish(self):
        # Old regex treated "marks", "help", "next time" as Hinglish -> English questions got Hinglish/Hindi.
        for q in ("Why did I lose marks in question 2?", "Please help me improve next time",
                  "How are marks distributed in this paper?", "Can you help with my weak topics?"):
            self.assertEqual(resolve_language("auto", q).code, "en", q)

    def test_only_latest_message_matters(self):
        # a Hindi earlier turn cannot influence a new English question: the function only sees the latest message
        self.assertEqual(resolve_language("auto", "Now explain it again in detail").code, "en")

    def test_hinglish_short_messages(self):
        for q in ("mere marks kyu kate?", "Question 3 samjhao", "isme galti kya hai"):
            self.assertEqual(resolve_language("auto", q).code, "hinglish", q)

    def test_unknown_script_auto_falls_back_to_generic_rule(self):
        l = resolve_language("auto", "பைனரி ட்ரீ என்றால் என்ன?")
        self.assertEqual(l.code, "unknown")
        self.assertIn("latest message", language_instruction(l))


class TestInstructionsAndChecks(unittest.TestCase):
    def test_instruction_names_priority_and_ignores_sources(self):
        en = language_instruction(resolve_language("English", HI))
        self.assertIn("English only", en)
        self.assertIn("must NOT change", en)
        self.assertIn("Devanagari", en)
        self.assertIn("ROMAN", language_instruction(resolve_language("Hinglish", EN)))
        self.assertIn("Devanagari script", language_instruction(resolve_language("Hindi", EN)))

    def test_reply_matches(self):
        en, hi = resolve_language("English", EN), resolve_language("Hindi", EN)
        hing = resolve_language("Hinglish", EN)
        self.assertTrue(reply_matches(en, "A stack is LIFO and a queue is FIFO."))
        self.assertFalse(reply_matches(en, "स्टैक एक डेटा संरचना है जिसमें LIFO क्रम चलता है और बहुत कुछ"))
        self.assertTrue(reply_matches(hi, "स्टैक एक डेटा संरचना है जिसमें LIFO क्रम चलता है।"))
        self.assertFalse(reply_matches(hi, "A stack is a data structure that follows the LIFO order of access."))
        self.assertTrue(reply_matches(hing, "Stack mein LIFO order hota hai, isliye marks kate."))
        self.assertFalse(reply_matches(hing, "स्टैक में LIFO क्रम होता है और इसलिए अंक कटे हैं यहाँ"))

    def test_offline_notice_in_each_language(self):
        self.assertIn("Ollama", offline_notice(resolve_language("English", EN)))
        self.assertTrue(any("\u0900" <= ch <= "\u097F" for ch in offline_notice(resolve_language("Hindi", EN))))
        self.assertIn("nahi", offline_notice(resolve_language("Hinglish", EN)))


if __name__ == "__main__":
    unittest.main()
