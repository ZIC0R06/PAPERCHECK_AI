"""
Regression tests for the handwritten-OCR failure (needs only Tesseract + Pillow/numpy/opencv; no server).

Root causes covered
  1. Tesseract was only found via TESSERACT_CMD, never in the standard Windows install folder → reading failed.
  2. A crash in local OCR was swallowed: pages became blank and were graded as "not attempted" (silent zero marks).
  3. PDF rendering depended on PyMuPDF alone.
  4. Handwritten "Q1" is read as "Q7"/"| Q7" and must be reconciled with the question paper.
  5. Tests were not isolated from a real local Ollama.
"""
import os
import shutil
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from backend import handwriting_reader as H                      # noqa: E402
from backend import tesseract_locator as TL                      # noqa: E402

FIX = os.path.join(os.path.dirname(__file__), "fixtures", "handwritten_answer_sheet.pdf")
QS = [{"question_number": n, "question_text": "q", "marks": 10} for n in (1, 2, 3, 4)]
HAVE_TESS = shutil.which("tesseract") is not None or TL.find_tesseract() is not None


class TestTesseractLocator(unittest.TestCase):
    def test_env_var_path_is_used(self):
        with mock.patch.dict(os.environ, {"TESSERACT_CMD": __file__}):
            self.assertEqual(TL.find_tesseract(), __file__)

    def test_standard_install_folder_is_searched_when_not_on_path(self):
        fake = os.path.join(os.path.dirname(__file__), "mock_ollama.py")
        with mock.patch.dict(os.environ, {"TESSERACT_CMD": ""}), mock.patch("shutil.which", return_value=None), \
                mock.patch.object(TL, "_CANDIDATES", [r"C:\nope\tesseract.exe", fake]):
            self.assertEqual(TL.find_tesseract(), fake)

    def test_missing_tesseract_gives_actionable_error(self):
        with mock.patch.object(TL, "find_tesseract", return_value=None):
            st = TL.configure()
            self.assertFalse(st["available"])
            self.assertIn("TESSERACT_CMD", st["error"])
            with self.assertRaises(TL.TesseractMissingError):
                TL.require()


class TestReaderFailurePolicy(unittest.TestCase):
    def _img(self):
        from PIL import Image
        return [Image.new("RGB", (400, 400), "white")]

    def test_missing_tesseract_is_not_swallowed(self):
        from backend.tesseract_locator import TesseractMissingError
        with mock.patch.object(H, "configured_providers", return_value=[]), \
                mock.patch.object(H, "read_page_local", side_effect=TesseractMissingError("install it")):
            with self.assertRaises(TesseractMissingError):
                H.read_answer_sheet("x.pdf", QS, pages=self._img())

    def test_ocr_failing_on_every_page_raises_instead_of_blank_answers(self):
        with mock.patch.object(H, "configured_providers", return_value=[]), \
                mock.patch.object(H, "read_page_local", side_effect=ValueError("boom")):
            with self.assertRaises(RuntimeError) as cm:
                H.read_answer_sheet("x.pdf", QS, pages=self._img() * 2)
            self.assertIn("could not be read", str(cm.exception))


@unittest.skipUnless(HAVE_TESS, "Tesseract not installed")
class TestHandwrittenFixture(unittest.TestCase):
    """The exact sheet used by tests/test_end_to_end_handwritten.py (image-only PDF, 2 pages)."""

    @classmethod
    def setUpClass(cls):
        with mock.patch.object(H, "configured_providers", return_value=[]):
            cls.pages = H.load_pages(FIX)
            cls.reading = H.read_answer_sheet(FIX, QS, pages=cls.pages)

    def test_pdf_renders_both_pages_without_pymupdf_dependency(self):
        self.assertEqual(len(self.pages), 2)

    def test_misread_question_one_is_reconciled(self):
        nums = [a.question_number for a in self.reading.answers]
        self.assertEqual(nums, [1, 2, 3])            # "Q7" on the sheet is Q1 on the paper

    def test_answers_land_on_correct_pages_with_text(self):
        a = {x.question_number: x for x in self.reading.answers}
        self.assertIn("primary key", a[1].text.lower())
        self.assertIn("deadlock", a[3].text.lower())
        self.assertEqual(a[1].page_refs[0]["page"], 1)
        self.assertEqual(a[3].page_refs[0]["page"], 2)

    def test_engine_and_review_warning(self):
        self.assertEqual(self.reading.engine, "local-tesseract")
        self.assertTrue(any("checked against the original" in w for w in self.reading.warnings))
        self.assertTrue(all(0 < x.confidence < 0.9 for x in self.reading.answers))   # never overconfident

    def test_unanswered_question_four_is_absent_not_invented(self):
        self.assertNotIn(4, [a.question_number for a in self.reading.answers])


if __name__ == "__main__":
    unittest.main()
