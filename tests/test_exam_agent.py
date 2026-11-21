"""
Examination Assistant — routing, role scoping contract, grounding and fallbacks.
Runs WITHOUT a database: FakeRepo mimics the role-scoped SqlRepo (faculty see only their subject/batch).
(SqlRepo itself is covered by tests/test_exam_agent_api.py, which needs the full stack.)
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from backend.exam_agent import ExamAgent, extract_entities, numbers_grounded, route  # noqa: E402

ALL = [
    dict(reg="2022CS001", name="Asha Rao", batch="CSE-A", course_code="CS301", course_name="DBMS", exam="Mid", total=18, max_total=32, percentage=56.25, grade="C", passed=True, status="published"),
    dict(reg="2022CS002", name="Ravi Kumar", batch="CSE-A", course_code="CS301", course_name="DBMS", exam="Mid", total=9, max_total=32, percentage=28.1, grade="F", passed=False, status="approved"),
    dict(reg="2022CS003", name="Neha Singh", batch="CSE-A", course_code="CS301", course_name="DBMS", exam="Mid", total=30, max_total=32, percentage=93.8, grade="A+", passed=True, status="published"),
    dict(reg="2022MA001", name="Om Verma", batch="MTH-B", course_code="MA101", course_name="Maths", exam="Mid", total=20, max_total=50, percentage=40.0, grade="D", passed=True, status="draft"),
]
SHEET = {"booklet_id": 7, "course_code": "CS301", "exam": "Mid", "status": "approved", "questions": [
    dict(question_number=1, awarded=7.0, max_marks=10.0, flagged=False, deductions=[dict(marks=3, criterion="example", reason="No example given")], missing_concepts=["example of primary key"]),
    dict(question_number=3, awarded=3.0, max_marks=10.0, flagged=True, deductions=[dict(marks=7, criterion="prevention", reason="Only timeouts mentioned")], missing_concepts=["lock ordering", "wait-die"])]}


class FakeRepo:
    """admin: everything. faculty: only CS301 (their assigned subject)."""
    def __init__(self, admin):
        self.is_admin = admin
        self.calls = []

    def _vis(self):
        return ALL if self.is_admin else [r for r in ALL if r["course_code"] == "CS301"]

    def scope_description(self): return "everything" if self.is_admin else "CS301 for CSE-A"
    def known_course_codes(self): return ["CS301", "MA101"]
    def find_students(self, q): return [dict(reg=r["reg"], name=r["name"], batch=r["batch"], semester="S5") for r in self._vis() if q.lower() in (r["reg"] + r["name"]).lower()]

    def results(self, reg=None, course=None, exam=None):
        self.calls.append(("results", reg, course))
        return [r for r in self._vis() if (not reg or r["reg"] == reg) and (not course or r["course_code"] == course)]

    def question_wise(self, reg, course=None):
        return [SHEET] if reg in {r["reg"] for r in self._vis()} and reg == "2022CS001" else []

    def missing_concepts(self, course=None): return ["lock ordering", "wait-die", "lock ordering", "normal forms"]
    def pending(self): return dict(by_status={"pending": 2, "approved": 1, "published": 4}, flagged_answers=5, oldest=[dict(booklet_id=9, reg="2022CS002", course="CS301", status="pending")])
    def revaluations(self, status=None): return [dict(id=1, booklet_id=7, reg="2022CS001", question_number=3, reason="Q3 should get more marks", status="pending", score_before=3.0, score_after=None)]
    def audit(self, limit=15): return [dict(when="2026-11-01 10:00", who="t1", role="faculty", action="MARKS_OVERRIDE", target="booklet 7")]
    def faculty_activity(self): return [dict(who="t1", actions=12, top=["MARKS_OVERRIDE", "PUBLISH"])]


class FakeLLM:
    def __init__(self, reply="ok", up=True):
        self.reply, self.up, self.systems, self.prompts = reply, up, [], []
    def is_available(self): return self.up
    def complete(self, system, user, max_tokens=0, history=None, **kw):
        self.systems.append(system); self.prompts.append(user)
        return self.reply


def ask(msg, admin=True, llm=None, **kw):
    repo = FakeRepo(admin)
    return ExamAgent(repo, llm or FakeLLM(up=False)).chat(msg, **kw), repo


class TestRouting(unittest.TestCase):
    codes = ["CS301", "MA101"]

    def test_entities(self):
        e = extract_entities("why were marks cut for 2022CS001 in CS301 q3", self.codes)
        self.assertEqual((e["reg"], e["course"], e["question"]), ("2022CS001", "CS301", 3))
        self.assertEqual(extract_entities("students below 35% in MA101", self.codes)["threshold"], 35.0)
        self.assertEqual(extract_entities("students below 35 in MA101", self.codes)["threshold"], 35.0)
        self.assertIsNone(extract_entities("average of CS301", self.codes)["reg"])   # a course code is not a student

    def test_routes(self):
        r = lambda m: route(m, extract_entities(m, self.codes), True)   # noqa: E731
        self.assertEqual(r("results of 2022CS001"), ["student_results"])
        self.assertIn("question_wise", r("why did 2022CS001 lose marks in q3"))
        self.assertIn("exam_stats", r("how did the class perform in CS301"))
        self.assertIn("support_needed", r("which students need help below 40"))
        self.assertIn("pending", r("show pending checking jobs"))
        self.assertIn("revaluations", r("any revaluation requests"))
        self.assertIn("weak_topics", r("what are the weak topics in CS301"))
        self.assertIn("report", r("generate an examination report"))
        self.assertEqual(r("tell me a joke"), [])


class TestRoleScoping(unittest.TestCase):
    def test_admin_sees_all_subjects(self):
        out, _ = ask("how did the class perform", admin=True)
        self.assertIn("MA101", out["reply"]); self.assertIn("CS301", out["reply"])

    def test_faculty_never_sees_other_subjects(self):
        out, _ = ask("how did the class perform", admin=False)
        self.assertNotIn("MA101", out["reply"]); self.assertNotIn("Om Verma", out["reply"])
        out, _ = ask("results of 2022MA001", admin=False)
        self.assertIn("No result records", out["reply"])            # same answer as "does not exist": no leak
        self.assertNotIn("Om Verma", out["reply"])

    def test_faculty_support_list_is_scoped(self):
        out, _ = ask("students below 45 who need support", admin=False)
        self.assertIn("2022CS002", out["reply"]); self.assertNotIn("2022MA001", out["reply"])

    def test_faculty_activity_is_admin_only(self):
        self.assertIn("administrators only", ask("faculty activity", admin=False)[0]["reply"])
        self.assertIn("t1", ask("faculty activity monitor", admin=True)[0]["reply"])


class TestAnswers(unittest.TestCase):
    def test_student_results_and_flags(self):
        out, _ = ask("results of 2022CS001")
        self.assertIn("18/32", out["reply"]); self.assertIn("PASS", out["reply"])

    def test_deductions_are_explained_from_stored_data(self):
        out, _ = ask("why were marks cut for 2022CS001 in q3")
        self.assertIn("Only timeouts mentioned", out["reply"]); self.assertIn("3/10", out["reply"])
        self.assertIn("flagged for teacher verification", out["reply"]); self.assertNotIn("Q1:", out["reply"])

    def test_stats_numbers(self):
        out, _ = ask("statistics for CS301")
        self.assertIn("3 students", out["reply"]); self.assertIn("pass rate 66.7%", out["reply"])
        self.assertIn("highest 93.8% (2022CS003)", out["reply"])

    def test_support_needed_threshold(self):
        out, _ = ask("students below 40 in CS301")
        self.assertIn("2022CS002", out["reply"]); self.assertNotIn("2022CS003", out["reply"])

    def test_report_contains_sections_and_is_downloadable(self):
        out, _ = ask("generate a full examination report for CS301")
        self.assertTrue(out["report"].startswith("# Examination report"))
        for s in ("Examination statistics", "Students needing academic support", "Checking queue"):
            self.assertIn(s, out["report"])

    def test_followup_inherits_student(self):
        hist = [dict(role="user", content="results of 2022CS001"), dict(role="assistant", content="...")]
        out, _ = ask("and why q3?", history=hist)
        self.assertIn("question_wise", out["tools"])

    def test_unknown_question_gets_help_not_a_guess(self):
        out, _ = ask("what is the weather")
        self.assertEqual(out["mode"], "help"); self.assertEqual(out["tools"], [])

    def test_read_only_notice_for_revaluation(self):
        self.assertIn("read-only", ask("revaluation requests")[0]["reply"])


class TestGroundingAndFallback(unittest.TestCase):
    def test_numbers_grounded(self):
        self.assertTrue(numbers_grounded("Asha scored 18/32.", "18/32 (56.2%)"))
        self.assertFalse(numbers_grounded("Asha scored 25/32.", "18/32 (56.2%)"))
        self.assertTrue(numbers_grounded("3 students failed", "x"))                # tiny counts allowed

    def test_ai_answer_used_when_grounded_and_language_rule_sent(self):
        llm = FakeLLM("2022CS001 scored 18/32 (56.2%) and passed.")
        out, _ = ask("results of 2022CS001", llm=llm, language="English")
        self.assertEqual(out["mode"], "ai"); self.assertIn("18/32", out["reply"])
        self.assertIn("English only", llm.systems[-1]); self.assertIn("VERIFIED DATA", llm.prompts[-1])
        self.assertIn("never invent", llm.systems[-1].lower())

    def test_invented_marks_are_rejected(self):
        out, _ = ask("results of 2022CS001", llm=FakeLLM("2022CS001 scored 29/32 (90%)."))
        self.assertEqual(out["mode"], "data-verified"); self.assertIn("18/32", out["reply"])
        self.assertNotIn("29/32", out["reply"])

    def test_ollama_down_returns_verified_data_and_says_so(self):
        out, _ = ask("results of 2022CS001", llm=FakeLLM(up=False))
        self.assertEqual(out["mode"], "data-only"); self.assertIn("18/32", out["reply"]); self.assertIn("Ollama", out["reply"])

    def test_model_error_falls_back(self):
        class Boom(FakeLLM):
            def complete(self, *a, **k): raise RuntimeError("timeout")
        out, _ = ask("results of 2022CS001", llm=Boom())
        self.assertEqual(out["mode"], "data-only"); self.assertIn("18/32", out["reply"])

    def test_wrong_language_reply_is_retried(self):
        replies = iter(["2022CS001 ne 18/32 score kiya. यह हिंदी है और बहुत सारे शब्द यहाँ हैं।", "2022CS001 scored 18/32."])
        class Seq(FakeLLM):
            def complete(self, *a, **k): return next(replies)
        out, _ = ask("results of 2022CS001", llm=Seq(), language="English")
        self.assertEqual(out["reply"], "2022CS001 scored 18/32.")

    def test_hindi_offline_notice(self):
        out, _ = ask("results of 2022CS001", language="Hindi")
        self.assertTrue(any("\u0900" <= c <= "\u097F" for c in out["reply"]))


if __name__ == "__main__":
    unittest.main()
