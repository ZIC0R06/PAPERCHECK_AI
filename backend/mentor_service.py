"""
AI Paper Checking — Personalized AI Mentor Chatbot (RAG Engine)
================================================================
Empowers students with an interactive AI academic mentor.
Capabilities:
  • Explains why marks were deducted from specific exam questions
  • Identifies missing concepts, misconceptions, and structural issues
  • Provides actionable suggestions to format high-scoring descriptive answers
  • Generates targeted practice questions & quizzes based on weak areas
  • Retrieves verified course notes, rubrics, and published exam evaluations (RAG)
  • Enforces strict student data privacy: never reveals another student's marks
"""

import json
import re
from typing import Dict, List, Optional, Tuple
from sqlalchemy.orm import Session

from backend.database import (
    Student, User, StudentBooklet, StudentAnswerText,
    Submission, Result, CourseMaterial, ExamQuestion, Rubric,
    ChatSession, ChatMessage, StudyDocument, DocumentChunk, WeakTopic
)
from backend.llm_provider import get_llm_client
from backend import mentor_grounding as G
from backend.language_policy import (Lang, resolve_language, language_instruction, reply_matches,
                                      offline_notice)

STYLE_INSTRUCTIONS = {
    "hinglish": "Reply in simple Hinglish: Hindi written in Roman script mixed naturally with English technical terms. Short, friendly sentences, like a senior explaining to a junior.",
    "detailed": "Give a detailed, well-structured explanation with headings, step-by-step reasoning and an example where useful.",
    "short_notes": "Reply as concise, exam-ready short notes: crisp bullet points, key definitions and keywords only (max about 12 bullets).",
    "quiz_me": "Answer briefly, then test the student with 2-3 quick check questions. Put the answers at the very end under a heading 'Answers'.",
}

DOC_ACTIONS = {"summarize", "mcq", "short_questions", "important_questions", "explain_hinglish"}
MAX_DOC_CONTEXT_CHARS = 14000

logger = __import__("logging").getLogger(__name__)


class AIMentorService:
    def __init__(self, db: Session):
        self.db = db
        self.llm_client = get_llm_client()
        self.lang: Lang = Lang("unknown", "the same language as the latest message", False)
        self.reply_language: Optional[str] = None   # kept for backward compatibility (explicit non-en/hi/hinglish name)
        self.on_token = None                        # streaming callback (set by /mentor/chat/stream)

    def handle_student_query(
        self,
        current_user: User,
        student: Student,
        message: str,
        course_code: Optional[str] = None,
        booklet_id: Optional[int] = None,
        question_number: Optional[int] = None,
        session_id: Optional[int] = None,
        document_ids: Optional[List[int]] = None,
        response_style: Optional[str] = None,
        action: Optional[str] = None,
        response_language: Optional[str] = None,
        regenerate: bool = False,
    ) -> Dict:
        """
        Processes a student query with RAG grounding.
        Strictly limits context to the authenticated student's published results.
        """
        # 1. Retrieve or create session
        if session_id:
            chat_session = self.db.query(ChatSession).filter_by(id=session_id, student_id=student.id).first()
        else:
            chat_session = None

        if not chat_session:
            title = f"Consultation: {message[:35]}..." if message else "Academic Mentor Session"
            chat_session = ChatSession(
                student_id=student.id,
                course_code=course_code,
                booklet_id=booklet_id,
                title=title
            )
            self.db.add(chat_session)
            self.db.commit()
            self.db.refresh(chat_session)

        # Regenerate: drop the previous answer (and its question, re-saved below) before answering again
        if regenerate:
            last2 = (self.db.query(ChatMessage).filter(ChatMessage.session_id == chat_session.id)
                     .order_by(ChatMessage.id.desc()).limit(2).all())
            if len(last2) == 2 and last2[0].sender == "assistant" and last2[1].sender == "user" \
                    and last2[1].content.strip() == (message or "").strip():
                self.db.delete(last2[0])
                self.db.delete(last2[1])
                self.db.commit()

        # 2. Conversation history (for follow-ups) — read BEFORE saving the new message
        history = self._recent_history(chat_session.id)

        # Save user message
        user_msg = ChatMessage(session_id=chat_session.id, sender="user", content=message)
        self.db.add(user_msg)
        self.db.commit()

        style = (response_style or "").strip().lower() or None
        # LANGUAGE: explicit selector > latest message (Auto). History/documents never decide.
        selected = (response_language or "").strip()
        if style == "hinglish" and selected.lower() in ("", "auto", "same as question"):
            selected = "Hinglish"            # legacy "Hinglish" style chip behaves like the Hinglish language
        self.lang = resolve_language(selected, message)
        if style == "hinglish":
            style = None                     # language is handled by self.lang, keep other styles intact
        self.reply_language = self.lang.name if self.lang.code == "other" else None
        action = (action or "").strip().lower() or None
        docs = self._owned_documents(current_user.id, document_ids)

        if not action and not docs:
            action = self._infer_action(message)

        if docs or action:
            context_data = self._gather_rag_context(student, course_code, booklet_id, question_number, message)
            reply_text, sources_cited, meta = self._handle_grounded(
                student, current_user, message, docs, action, style, history, context_data
            )
            reply_text = self._language_note(reply_text, meta.get("mode"))
            asst_msg = ChatMessage(session_id=chat_session.id, sender="assistant",
                                   content=reply_text, sources_cited=json.dumps(sources_cited))
            self.db.add(asst_msg)
            self.db.commit()
            return {
                "session_id": chat_session.id,
                "session_title": chat_session.title,
                "reply": reply_text,
                "sources_cited": sources_cited,
                "question_number": question_number,
                "student_name": student.name,
                **meta,
            }

        # 3. Detect question number if student mentions e.g. "Question 3" or "Q3"
        q_match = re.search(r'\b(?:question|q|problem)\s*#?\s*(\d+)\b', message, re.IGNORECASE)
        if q_match:
            question_number = int(q_match.group(1))

        # 4. RAG Context Gathering
        context_data = self._gather_rag_context(student, course_code, booklet_id, question_number, message)

        # 5. Generate AI Mentor Response
        reply_text, sources_cited = self._generate_response(
            student, message, context_data, question_number, history=history, style=style, user_id=current_user.id
        )

        reply_text = self._language_note(
            reply_text, "llm" if "AI Mentor Academic Reasoning Model" in sources_cited else "rules")

        # 6. Save assistant message
        asst_msg = ChatMessage(
            session_id=chat_session.id,
            sender="assistant",
            content=reply_text,
            sources_cited=json.dumps(sources_cited)
        )
        self.db.add(asst_msg)
        self.db.commit()

        return {
            "session_id": chat_session.id,
            "session_title": chat_session.title,
            "reply": reply_text,
            "sources_cited": sources_cited,
            "question_number": question_number,
            "student_name": student.name,
            "mode": "llm" if "AI Mentor Academic Reasoning Model" in sources_cited else "rules",
        }

    def _gather_rag_context(
        self,
        student: Student,
        course_code: Optional[str],
        booklet_id: Optional[int],
        question_number: Optional[int],
        query: str
    ) -> Dict:
        """
        Gathers approved course notes, question rubrics, and published marks.
        """
        context = {
            "booklet_info": None,
            "question_eval": None,
            "course_materials": [],
            "weak_topics": [],
            "all_results": []
        }

        # Find target booklet (published only)
        booklet_query = self.db.query(StudentBooklet).filter(
            StudentBooklet.student_id == student.id,
            StudentBooklet.is_published == True
        )
        if booklet_id:
            booklet = booklet_query.filter(StudentBooklet.id == booklet_id).first()
        else:
            booklet = booklet_query.order_by(StudentBooklet.id.desc()).first()

        if booklet:
            context["booklet_info"] = {
                "id": booklet.id,
                "course_code": booklet.course_code,
                "course_name": booklet.course_name,
                "exam_name": booklet.exam_name,
                "total_score": booklet.total_final_score or booklet.total_ai_score
            }
            if not course_code:
                course_code = booklet.course_code

            # Gather question evaluations
            for ans in booklet.answers:
                if ans.submission and ans.submission.result:
                    res = ans.submission.result
                    max_m = res.max_marks or 10.0
                    score = res.examiner_score if res.examiner_approved and res.examiner_score is not None else res.final_score
                    q_data = {
                        "question_number": ans.question_number,
                        "question_type": res.question_type,
                        "score": score,
                        "max_marks": max_m,
                        "is_approved": res.examiner_approved,
                        "strengths": json.loads(res.strengths) if res.strengths and res.strengths.startswith("[") else ([res.strengths] if res.strengths else []),
                        "missing_concepts": json.loads(res.missing_concepts) if res.missing_concepts and res.missing_concepts.startswith("[") else ([res.missing_concepts] if res.missing_concepts else []),
                        "feedback": res.feedback,
                        "examiner_comments": res.examiner_comments,
                        "student_answer": ans.answer_text,
                    }
                    context["all_results"].append(q_data)

                    if (score < max_m or score < (max_m * 0.85)) and q_data["missing_concepts"]:
                        context["weak_topics"].extend(q_data["missing_concepts"])

                    if question_number and ans.question_number == question_number:
                        context["question_eval"] = q_data

        # Course Materials (RAG)
        if course_code:
            materials = self.db.query(CourseMaterial).filter(CourseMaterial.course_code == course_code).all()
            for mat in materials:
                context["course_materials"].append({
                    "topic": mat.topic,
                    "content": mat.content_text,
                    "keywords": mat.keywords
                })

        return context

    def _generate_response(self, student: Student, query: str, context: Dict, question_number: Optional[int],
                           history: Optional[List[Dict]] = None, style: Optional[str] = None,
                           user_id: Optional[int] = None) -> tuple:
        """
        Builds the mentor response using RAG context with clear distinction
        between official examiner grades and AI mentoring suggestions.
        Supports both natural English and Hinglish inquiries.
        """
        q_eval = context.get("question_eval")
        sources_cited = []

        # Language comes from the central policy (explicit selection > latest message), NOT from keywords:
        # the old regex treated common English words such as "marks" and "help" as Hinglish.
        is_hinglish = self.lang.code == "hinglish"

        # Hindi / other explicit languages have no fixed template: let the AI write the reply from the
        # official facts below (it may not change marks). If AI is down, English template + notice is used.
        use_templates = not (self.lang.code in ("hi", "other") and self.llm_available())
        facts = ""
        if q_eval:
            facts = (f"OFFICIAL RESULT (do not change these numbers): question {question_number}, "
                     f"{q_eval['score']}/{q_eval['max_marks']} marks; missing concepts: {q_eval['missing_concepts']}; "
                     f"strengths: {q_eval['strengths']}. ")

        # If question-specific inquiry
        if use_templates and question_number and q_eval:
            sources_cited.append(f"Official Exam Script Q{question_number} Result")
            awarded = q_eval['score']
            max_m = q_eval['max_marks']
            status = "Verified by Chief Examiner" if q_eval['is_approved'] else "AI Automated Evaluation"
            missing = q_eval['missing_concepts']
            strengths = q_eval['strengths']

            # Find matching course notes
            relevant_notes = []
            for mat in context["course_materials"]:
                if any(k.strip().lower() in mat["keywords"].lower() for k in missing + [mat["topic"]]):
                    relevant_notes.append(mat)
                    sources_cited.append(f"Course Material: {mat['topic']}")

            missing_str = "\n".join(f"• **{m}**" for m in missing) if missing else "• Detailed depth could be expanded."
            strengths_str = "\n".join(f"• {s}" for s in strengths) if strengths else "• Response attempted and relevant."

            notes_summary = ""
            if relevant_notes:
                heading = "Reference Material from Course Notes" if not is_hinglish else "Course Notes se Reference"
                notes_summary = f"\n\n### 📖 {heading}:\n" + "\n".join(
                    f"**{n['topic']}**: {n['content'][:250]}..." for n in relevant_notes[:2]
                )

            if is_hinglish:
                reply = f"""### 🎓 AI Mentor Analysis for Question {question_number} (Hinglish Guide)

**Official Grade Record:** **{awarded:.1f} / {max_m:.1f} Marks** ({status})

#### 🔍 Marks kyu kate (Deduction Reasons):
Aapke answer mein examiner evaluation ke according ye key concepts missing ya incomplete the:
{missing_str}

#### ✅ Aapne kya accha kiya (Strengths):
{strengths_str}

#### 💡 Next time full marks laane ke liye kya accha kar sakte hain:
1. **Answer Structure:** Pehle 1-2 lines mein formal definition ya core formula likhein, uske baad step-by-step operational flow samjhayein.
2. **Missing points include karein:** Khas taur par {', '.join(missing[:3]) if missing else 'technical completeness'} ko point-wise answer mein highlight karein.
3. **Diagrams & Complexity:** Computer Science/Engineering exams mein agar algorithm ya architecture ka question ho, toh ek neat block diagram aur asymptotic time/space complexity zaroor mention karein.{notes_summary}

---
*Note: Official marks University Examiner dwara verify kiye gaye hain. AI Mentor recommendations aapko aage aane wale exams mein top score karne ke liye diye gaye hain.*"""
            else:
                reply = f"""### 🎓 AI Mentor Analysis for Question {question_number}

**Official Grade Record:** **{awarded:.1f} / {max_m:.1f} Marks** ({status})

#### 🔍 Why marks were deducted:
The evaluation identifies that your answer had key conceptual gaps:
{missing_str}

#### ✅ What you did well:
{strengths_str}

#### 💡 How to improve this answer next time:
1. **Structure your answer:** Start with a formal definition or equation, then explain the operational mechanism step-by-step.
2. **Include missed key concepts:** Specifically address {', '.join(missing[:3]) if missing else 'theoretical rigor'}.
3. **Illustrate with an example or complexity analysis:** Always state edge cases, time/space complexity, or practical applications for full marks.{notes_summary}

---
*Note: Official marks are certified by the University Examiner. Mentoring recommendations are AI-generated to help you succeed in upcoming examinations.*"""
            return reply, sources_cited

        # General inquiry: "Which topics should I revise?", "How can I improve?"
        if use_templates and any(w in query.lower() for w in ["revise", "weak", "topics", "improve", "galti", "sudhaar", "accha"]):
            weak_topics = list(set(context.get("weak_topics", [])))
            if weak_topics:
                sources_cited.append("Published Examination Analytics")
                topics_str = "\n".join(f"1. **{t}**" for t in weak_topics[:5])
                if is_hinglish:
                    reply = f"""### 📚 Personalized Revision Roadmap for {student.name}

Aapke evaluated exam performance ke hisaab se, in concepts par marks kate hain aur inpar special revision chahiye:

{topics_str}

#### 🎯 Agle exam ke liye Action Plan:
- **Core Topics Revise karein:** In topics ke lecture slides aur reference book chapters dobara dhyan se padhein.
- **Descriptive Diagrams:** Jahan bhi possible ho, clean diagrams aur step-by-step flow chart banayein.
- **Practice Quizzes:** Aap mujhse *"Generate practice quiz"* keh sakte hain taaki turant test de sakein!"""
                else:
                    reply = f"""### 📚 Personalized Revision Roadmap for {student.name}

Based on your verified examination results, here are the core concepts that cost you marks and need focused revision:

{topics_str}

#### 🎯 Recommended Action Plan:
- **Review Course Modules:** Revisit lecture slides and textbook chapters specifically covering these topics.
- **Practice Descriptive Diagrams:** Where applicable, draw diagrams or step-by-step flowcharts.
- **Take Practice Quizzes:** Ask me *"Generate a practice quiz on these topics"* to test yourself right now!"""
                return reply, sources_cited

        # Generic RAG / Open-ended query
        # Try generating response via LLM client (Ollama by default) if query is specific or conversational
        if self.llm_client and hasattr(self.llm_client, "complete"):
            try:
                system_prompt = (
                    "You are a helpful, encouraging university AI Academic Mentor for students. "
                    f"Student Name: {student.name}, Department: {student.department}. "
                    "You have verified access only to this student's authorized performance records and approved syllabus. "
                    "Keep answers concise, actionable, and academically rigorous. "
                    "Prefer the retrieved course notes when they are relevant and say when something is not covered by them. "
                    "Never expose other students' private records. Use Markdown (headings, lists, code blocks) where helpful."
                )
                if style in STYLE_INSTRUCTIONS:
                    system_prompt += " " + self._style_text(style)
                notes = self._retrieve_notes(query, user_id, context)
                notes_block = "\n\n".join(f"[{n['source']}]\n{n['content'][:900]}" for n in notes) or "(none matched)"
                user_prompt = (
                    f"{self._format_history(history)}"
                    f"Student's question: {query}\n\n{facts}\n"
                    f"Retrieved course notes / study material:\n{notes_block}\n\n"
                    f"Student's published exam context: weak topics: {list(set(context.get('weak_topics', [])))[:8]}, "
                    f"latest booklet: {context.get('booklet_info')}"
                )
                llm_out = self._llm(system_prompt, user_prompt, max_tokens=1200)
                if llm_out:
                    sources_cited.append("AI Mentor Academic Reasoning Model")
                    sources_cited.extend(n["source"] for n in notes)
                    return llm_out, sources_cited
            except Exception as e:
                logger.warning("LLM generation fallback in mentor service: %s", e)

        # Offline (no LLM key): answer from retrieved notes if they genuinely match the question
        notes = self._retrieve_notes(query, user_id, context)
        if notes and G.coverage(query, notes) >= 0.5 and len(G.tokenize(query)) >= 2:
            lines = "\n".join(f"- {n['content'][:320].strip()}… *({n['source']})*" for n in notes[:3])
            head = "Aapke notes mein ye relevant points mile:" if is_hinglish else "Here is what your course notes say about this:"
            reply = (f"### 📖 From your course material\n\n{head}\n\n{lines}\n\n"
                     "_" + offline_notice(self.lang) + "_")
            return reply, [n["source"] for n in notes[:3]]

        sources_cited.append("University Curriculum & Performance Profile")
        if self.lang.code == "hi":
            reply = (f"नमस्ते {student.name}! मैं आपका **AI अकादमिक मेंटर** हूँ। 🎓\n\n"
                     "आप मुझसे पूछ सकते हैं:\n- **\"प्रश्न 1 में मेरे अंक क्यों कटे?\"**\n"
                     "- **\"मेरे उत्तर में क्या कमी थी?\"**\n- **\"मुझे कौन से टॉपिक दोबारा पढ़ने चाहिए?\"**\n"
                     "- **\"इन टॉपिक पर प्रैक्टिस क्विज़ बनाओ।\"**")
        elif is_hinglish:
            reply = f"""Namaste {student.name}! Main aapka **AI Academic Mentor** hoon. 🎓

Main aapke published exam evaluation, marking rubrics aur course notes se connected hoon. Aap mujhse pooch sakte hain:
- **"Question 1 mein mere marks kyu kate?"**
- **"Mere answer mein kya missing tha?"**
- **"Next time full marks ke liye answer kaise likhein?"**
- **"Mujhe kaunse weak topics revise karne chahiye?"**
- **"In topics par practice quiz generate karo."**

Aapke academic improvement ke liye main kaise madad karoon?"""
        else:
            reply = f"""Hello {student.name}! I am your **AI Academic Mentor**. 

I have access to your published examination results and course materials. You can ask me:
- **"Why did I lose marks in Question 1?"**
- **"What was missing from my answer in Question 2?"**
- **"How can I structure a better 10-mark answer?"**
- **"Which topics should I revise for my upcoming tests?"**
- **"Generate a practice quiz for me."**

How can I help you improve your academic performance today?"""
        if not self.llm_available():
            reply += "\n\n_" + offline_notice(self.lang) + "_"
        return reply, sources_cited

    # ══════════════════════════════════════════════════════════════════════
    # AI Mentor dashboard: history, files, grounded actions
    # ══════════════════════════════════════════════════════════════════════

    def llm_available(self) -> bool:
        c = self.llm_client
        try:
            return bool(c and c.is_available())
        except Exception:
            return False

    def _llm(self, system: str, user: str, max_tokens: int = 1200) -> Optional[str]:
        """Call the configured LLM. Returns None when Ollama is unavailable (callers then use their grounded,
        non-AI fallback and say so). Re-asks once, more strictly, if the reply is in the wrong language."""
        if not self.llm_available() or not hasattr(self.llm_client, "complete"):
            return None
        base_system = system
        out = None
        for attempt in (1, 2):
            sys_prompt = base_system + "\n\n" + self._language_instruction()
            if attempt == 2:
                sys_prompt += (" YOUR PREVIOUS ANSWER USED THE WRONG LANGUAGE. Rewrite it now strictly in the "
                               "required language.")
            try:
                if self.on_token and hasattr(self.llm_client, "stream"):
                    self.on_token({"reset": True})       # a fresh LLM answer starts
                    out = self.llm_client.stream(sys_prompt, user, max_tokens=max_tokens, on_token=self.on_token)
                else:
                    out = self.llm_client.complete(sys_prompt, user, max_tokens=max_tokens)
            except Exception as e:
                logger.warning("LLM call failed in mentor: %s", e)
                return None
            out = (out or "").strip()
            if not out:
                return None
            if reply_matches(self.lang, out):
                return out
            logger.info("Mentor reply language mismatch (wanted %s) — retrying once", self.lang.code)
        return out

    def _style_text(self, style: Optional[str]) -> str:
        if style == "hinglish":
            return "Use very simple, friendly language with short sentences, like a senior explaining to a junior."
        return STYLE_INSTRUCTIONS.get(style or "", "")

    def _language_instruction(self) -> str:
        return language_instruction(self.lang)

    def _language_note(self, reply: str, mode: Optional[str]) -> str:
        """Without the AI model nothing can be translated — say so (in the right language) instead of pretending."""
        if mode != "llm" and self.lang.explicit and self.lang.code in ("hi", "other"):
            reply += "\n\n_🌐 " + offline_notice(self.lang) + "_"
        return reply

    def _recent_history(self, session_id: int, limit: int = 8) -> List[Dict]:
        msgs = (self.db.query(ChatMessage).filter(ChatMessage.session_id == session_id)
                .order_by(ChatMessage.id.desc()).limit(limit).all())
        return [{"sender": m.sender, "content": m.content} for m in reversed(msgs)]

    @staticmethod
    def _format_history(history: Optional[List[Dict]]) -> str:
        if not history:
            return ""
        lines = []
        for m in history[-6:]:
            who = "Student" if m["sender"] == "user" else "Mentor"
            lines.append(f"{who}: {m['content'][:700]}")
        return "Earlier in this conversation:\n" + "\n".join(lines) + "\n\n"

    @staticmethod
    def _infer_action(message: str) -> Optional[str]:
        """Typed/spoken requests that mean the same as a dashboard button."""
        m = message.lower()
        if re.search(r'\b(?:question|q|problem)\s*#?\s*\d+\b', m):
            return None  # question-specific → existing rubric analysis
        if re.search(r"\b(explain|show|check|samjha\w*|batao|breakdown)\b.*\b(marks?|results?|scores?|grades?)\b", m) or \
           re.search(r"\b(my|mere|meri|mera)\s+(marks|result|results|score)\b", m):
            return "explain_marks"
        if re.search(r"\b(study\s*plan|time\s*table|timetable|revision\s*plan|schedule)\b", m):
            return "study_plan"
        return None

    def _owned_documents(self, user_id: int, document_ids: Optional[List[int]]) -> List[StudyDocument]:
        """Only documents uploaded by THIS user — never another student's file."""
        if not document_ids:
            return []
        ids = [int(i) for i in document_ids if str(i).isdigit()][:5]
        if not ids:
            return []
        docs = self.db.query(StudyDocument).filter(
            StudyDocument.id.in_(ids), StudyDocument.user_id == user_id
        ).all()
        return docs

    def _doc_passages(self, docs: List[StudyDocument]) -> List[Dict]:
        out = []
        for d in docs:
            chunks = (self.db.query(DocumentChunk).filter(DocumentChunk.document_id == d.id)
                      .order_by(DocumentChunk.chunk_index).all())
            for c in chunks:
                out.append({"doc_id": d.id, "title": d.filename or d.title, "page": c.page_number or 1,
                            "content": c.content})
        return out

    def _retrieve_notes(self, query: str, user_id: Optional[int], context: Dict, top_k: int = 4) -> List[Dict]:
        """RAG over approved course materials + the student's own study documents."""
        passages = []
        for m in context.get("course_materials", []):
            passages.append({"title": m["topic"], "page": 1, "content": m["content"],
                             "source": f"Course notes: {m['topic']}"})
        if user_id:
            q = self.db.query(DocumentChunk).join(StudyDocument).filter(
                (StudyDocument.user_id == user_id) | (StudyDocument.user_id == None)  # noqa: E711 (shared dept material)
            ).limit(3000)
            for c in q.all():
                title = c.document.title if c.document else "Document"
                passages.append({"title": title, "page": c.page_number or 1, "content": c.content,
                                 "source": f"{title} (p.{c.page_number or 1})"})
        ranked = G.retrieve(passages, query, top_k=top_k)
        return [p for _, p in ranked]

    @staticmethod
    def _sources_for(passages: List[Dict]) -> List[str]:
        seen, out = set(), []
        for p in passages:
            k = f"{p['title']} (p.{p['page']})"
            if k not in seen:
                seen.add(k)
                out.append(k)
        return out[:8]

    def _handle_grounded(self, student: Student, user: User, message: str, docs: List[StudyDocument],
                         action: Optional[str], style: Optional[str], history: List[Dict], context: Dict):
        """Route dashboard actions and questions about attached files."""
        meta = {"mode": "llm" if self.llm_available() else "offline",
                "documents": [d.filename or d.title for d in docs], "grounded": True}

        if action == "explain_marks":
            return (*self._explain_marks(student, context, style), meta)
        if action == "weak_topics":
            return (*self._weak_topics_reply(user, context, style), meta)
        if action == "study_plan":
            return (*self._study_plan(student, user, context, style, message), meta)

        # File-grounded work: attached files first, else approved course notes
        passages = self._doc_passages(docs)
        source_kind = "file"
        if not passages:
            if action in DOC_ACTIONS:
                for m in context.get("course_materials", []):
                    passages.append({"title": f"Course notes: {m['topic']}", "page": 1, "content": m["content"]})
                source_kind = "course notes"
            if not passages:
                meta["grounded"] = False
                if docs:
                    msg = ("I couldn't read any text from the attached file, so I can't answer from it. "
                           "If it's a scanned PDF or photo, OCR may not be installed on the server.")
                else:
                    msg = ("📎 Please attach a PDF, DOCX, TXT or image first (use the **paperclip** or **Upload File**), "
                           "then ask again — I only generate questions and summaries from material you give me, "
                           "so nothing is made up.")
                return msg, [], meta

        if action in DOC_ACTIONS:
            reply, src = self._doc_action(action, passages, style, message, history, source_kind)
        else:
            reply, src, grounded = self._doc_question(message, passages, style, history)
            meta["grounded"] = grounded
        return reply, src, meta

    # ── file actions ─────────────────────────────────────────────────────────

    def _doc_context_block(self, passages: List[Dict], query: Optional[str] = None) -> Tuple[str, List[Dict]]:
        chosen = passages
        total = sum(len(p["content"]) for p in passages)
        if total > MAX_DOC_CONTEXT_CHARS and query:
            ranked = [p for _, p in G.retrieve(passages, query, top_k=24)]
            # keep document order, add opening passages for overall context
            keep = {id(p) for p in ranked} | {id(p) for p in passages[:4]}
            chosen = [p for p in passages if id(p) in keep]
        block, used, size = [], [], 0
        for p in chosen:
            if size + len(p["content"]) > MAX_DOC_CONTEXT_CHARS:
                break
            block.append(f"[{p['title']} — page {p['page']}]\n{p['content']}")
            used.append(p)
            size += len(p["content"])
        return "\n\n".join(block), used

    def _doc_action(self, action: str, passages: List[Dict], style: Optional[str], message: str,
                    history: List[Dict], source_kind: str) -> Tuple[str, List[str]]:
        block, used = self._doc_context_block(passages, message)
        sources = self._sources_for(used)
        style_line = self._style_text(style)
        if action == "explain_hinglish":
            style_line = STYLE_INSTRUCTIONS["hinglish"]
            if not self.lang.explicit:               # explicit language selection still wins
                self.lang = resolve_language("Hinglish", message)

        tasks = {
            "summarize": "Summarize the material: a 2-line overview, then the key points as bullets, then 'Key terms'.",
            "mcq": ("Create 5 multiple-choice questions from the material. For each: the question, options A-D, "
                    "'**Answer:** <letter>' and a one-line '**Explanation:**' quoting or paraphrasing the material with its page."),
            "short_questions": "Create 6 short-answer questions with model answers of 2-3 lines each, taken from the material.",
            "important_questions": ("List the 6-8 most important exam questions on this material, from most to least important, "
                                    "each with a one-line hint of what the answer must cover and the page."),
            "explain_hinglish": "Explain the main topics of this material simply, step by step, with a small example.",
        }
        system = ("You are Mentor AI, a university study mentor. Use ONLY the material provided below. "
                  "If something the student asks is not in the material, say clearly that it is not covered in it "
                  "— never invent facts. Cite pages like (p.3). Use Markdown. " + style_line)
        user_prompt = (f"{self._format_history(history)}Student request: {message}\n\nTask: {tasks[action]}\n\n"
                       f"MATERIAL ({source_kind}):\n{block}")
        out = self._llm(system, user_prompt, max_tokens=1800)
        if out:
            return out, sources

        # ── Offline, deterministic but strictly grounded ──
        note = ("\n\n---\n_" + offline_notice(self.lang) + " (Built directly from sentences in your "
                + source_kind + ".)_")
        hing = self.lang.code == "hinglish" or action == "explain_hinglish"
        if action in ("summarize", "explain_hinglish"):
            points, terms = G.extractive_summary(used, max_points=7)
            if not points:
                return "The material has too little readable text to summarize." + note, sources
            head = "### 📄 Is material ke main points" if hing else "### 📄 Summary"
            body = "\n".join(f"- {s} *(p.{p['page']})*" for s, p in points)
            kt = ("\n\n**Key terms:** " + ", ".join(terms)) if terms else ""
            extra = ("\n\n_Hinglish translation needs the LLM; points are quoted from your file as written._"
                     if hing else "")
            return f"{head}\n\n{body}{kt}{extra}{note}", sources
        if action == "mcq":
            mcqs = G.make_mcqs(used, n=5)
            if not mcqs:
                return "There isn't enough distinct content in this material to build MCQs." + note, sources
            parts = ["### 📝 MCQs from your material"]
            for i, q in enumerate(mcqs, 1):
                opts = "\n".join(f"   {'ABCD'[j]}. {o}" for j, o in enumerate(q["options"]))
                parts.append(f"**Q{i}. {q['question']}**\n{opts}\n\n**Answer:** {q['answer_letter']} — {q['answer']}  \n"
                             f"**Explanation:** \"{q['explanation']}\" *({q['source']})*")
            return "\n\n".join(parts) + note, sources
        if action == "short_questions":
            qs = G.make_short_questions(used, n=6)
            if not qs:
                return "I couldn't find clear definitions in this material to build short questions." + note, sources
            parts = ["### ✍️ Short-answer questions"]
            for i, q in enumerate(qs, 1):
                parts.append(f"**Q{i}. {q['question']}**  \n*Model answer:* {q['answer']} *({q['source']})*")
            return "\n\n".join(parts) + note, sources
        # important_questions
        qs = G.make_important_questions(used, n=6)
        if not qs:
            return "I couldn't identify recurring concepts in this material." + note, sources
        parts = ["### ⭐ Important questions (ranked by how often the concept appears)"]
        for i, q in enumerate(qs, 1):
            parts.append(f"{i}. **{q['question']}**  \n   _Must cover:_ {q['hint']} *({q['source']}, mentioned {q['weight']}×)*")
        return "\n\n".join(parts) + note, sources

    def _doc_question(self, message: str, passages: List[Dict], style: Optional[str],
                      history: List[Dict]) -> Tuple[str, List[str], bool]:
        query = message
        # short follow-ups ("aur detail me", "explain more") borrow the previous question's terms
        content_terms = [t for t in G.tokenize(message) if t not in G.QUERY_FILLER]
        follow_up = bool(re.search(r"\b(more|aur|detail|details|elaborate|iske|uske|isko|usko|this|that|it|example|again|phir)\b",
                                   message, re.I))
        if history and (not content_terms or (follow_up and len(content_terms) < 2)):
            prev = next((m["content"] for m in reversed(history) if m["sender"] == "user"), "")
            query = f"{prev} {message}"
        ranked = G.retrieve(passages, query, top_k=6)
        top = [p for _, p in ranked]
        cov = G.coverage(query, passages)
        overview_intent = bool(re.search(
            r"\b(important|main|key)\s+(topics?|points?|concepts?)|summar|overview|samjha|samjhao|samjhado|"
            r"explain (this|the) (file|pdf|document|notes)|what is (this|the) (file|pdf|document) about", message, re.I))
        broad = overview_intent or not [t for t in G.tokenize(query) if t not in G.QUERY_FILLER]

        system = ("You are Mentor AI, a university study mentor. Answer ONLY from the student's file excerpts below. "
                  "If the answer is not in them, reply that it is not covered in the uploaded file (you may then add one "
                  "line suggesting what to look for), and never invent content. Cite pages like (p.2). Use Markdown. "
                  + self._style_text(style))
        if broad or not top:
            block, used = self._doc_context_block(passages, query)
        else:
            used = top
            block = "\n\n".join(f"[{p['title']} — page {p['page']}]\n{p['content']}" for p in top)
        out = self._llm(system, f"{self._format_history(history)}Question: {message}\n\nFILE EXCERPTS:\n{block}", 1500)
        if out:
            return out, self._sources_for(used), True

        note = "\n\n---\n_" + offline_notice(self.lang) + " (These are the matching passages from your file.)_"
        hing = self.lang.code == "hinglish"
        if broad:
            points, terms = G.extractive_summary(passages, max_points=7)
            head = "### 📄 Aapki file ke main topics" if hing else "### 📄 Main points in your file"
            body = "\n".join(f"- {s} *(p.{p['page']})*" for s, p in points)
            kt = ("\n\n**Key terms:** " + ", ".join(terms)) if terms else ""
            if hing:
                note += " _Simple Hinglish explanation ke liye Ollama chalu hona chahiye._"
            return f"{head}\n\n{body}{kt}{note}", self._sources_for([p for _, p in points]), True
        if not top or cov < 0.5:
            return ("🔎 I searched your uploaded file but **this isn't covered in it**, so I won't guess. "
                    "Try rephrasing with words used in the file, or attach the chapter that covers it."), [], False
        # show the sentences that actually match, not whole chunks
        q_terms = {t for t in G.tokenize(query) if t not in G.QUERY_FILLER}
        hits, seen = [], set()
        for p in top:
            for sent in G.split_sentences(p["content"]):
                st = set(G.tokenize(sent))
                if q_terms & st and sent not in seen:
                    seen.add(sent)
                    hits.append((len(q_terms & st), sent, p))
        hits.sort(key=lambda x: -x[0])
        if hits:
            body = "\n".join(f"- {sent} *(p.{p['page']})*" for _, sent, p in hits[:5])
            used = [p for _, _, p in hits[:5]]
        else:
            body = "\n\n".join(f"> {G.unwrap(p['content'])[:600].strip()}  \n*— p.{p['page']}*" for p in top[:2])
            used = top[:2]
        return f"### 📖 What your file says\n\n{body}{note}", self._sources_for(used), True

    # ── exam-result actions ─────────────────────────────────────────────────

    def _explain_marks(self, student: Student, context: Dict, style: Optional[str]) -> Tuple[str, List[str]]:
        results = context.get("all_results") or []
        info = context.get("booklet_info")
        if not results:
            return ("I couldn't find any **published** exam results for your account yet. Once your examiner "
                    "publishes a result, I can explain every deduction question by question."), []
        hing = self.lang.code == "hinglish"
        rows = []
        total = sum(r["score"] or 0 for r in results)
        maxt = sum(r["max_marks"] or 0 for r in results)
        for r in sorted(results, key=lambda x: x["question_number"] or 0):
            lost = (r["max_marks"] or 0) - (r["score"] or 0)
            miss = ", ".join(r["missing_concepts"][:3]) or "—"
            rows.append(f"| Q{r['question_number']} | {r['score']:.1f} / {r['max_marks']:.1f} | {lost:.1f} | {miss} |")
        table = "| Question | Marks | Lost | Missing concepts |\n|---|---|---|---|\n" + "\n".join(rows)
        worst = max(results, key=lambda r: (r["max_marks"] or 0) - (r["score"] or 0))
        head = (f"### 📊 {info['course_code']} — {info['exam_name']}\n\n**Total: {total:.1f} / {maxt:.1f}**\n\n"
                if info else f"### 📊 Your published result\n\n**Total: {total:.1f} / {maxt:.1f}**\n\n")
        tip = (f"\n\nSabse zyada marks **Q{worst['question_number']}** mein kate. Mujhse poochiye: "
               f"*\"Q{worst['question_number']} mein marks kyu kate?\"* — main step-by-step samjhaunga."
               if hing else
               f"\n\nYou lost the most marks in **Q{worst['question_number']}**. Ask me "
               f"*\"Why did I lose marks in Q{worst['question_number']}?\"* for a detailed breakdown.")
        reply = head + table + tip + "\n\n*Official marks are as certified by your examiner.*"
        return reply, ["Official published exam result"]

    def _collect_weak(self, user: User, context: Dict) -> List[Tuple[str, str]]:
        out, seen = [], set()
        for t in context.get("weak_topics", []):
            k = t.strip().lower()
            if k and k not in seen:
                seen.add(k)
                out.append((t.strip(), "missed in published exam"))
        for w in self.db.query(WeakTopic).filter(WeakTopic.user_id == user.id).order_by(WeakTopic.accuracy.asc()).all():
            k = w.topic_name.strip().lower()
            if k not in seen and w.status != "Mastered":
                seen.add(k)
                out.append((w.topic_name, f"quiz accuracy {w.accuracy:.0f}%"))
        return out

    def _weak_topics_reply(self, user: User, context: Dict, style: Optional[str]) -> Tuple[str, List[str]]:
        weak = self._collect_weak(user, context)
        if not weak:
            return ("I don't see any weak topics yet — there are no missed concepts in your published results and "
                    "no low-scoring practice quizzes. Take a quiz from your notes and I'll track it."), []
        lines = "\n".join(f"{i}. **{t}** — _{why}_" for i, (t, why) in enumerate(weak[:10], 1))
        head = "### 🎯 Aapke weak topics" if self.lang.code == "hinglish" else "### 🎯 Your weak topics"
        return (f"{head}\n\n{lines}\n\nPick one and ask me to explain it, or click **Study Plan** for a day-wise plan.",
                ["Published exam results", "Practice quiz history"])

    def _study_plan(self, student: Student, user: User, context: Dict, style: Optional[str],
                    message: str) -> Tuple[str, List[str]]:
        weak = [t for t, _ in self._collect_weak(user, context)][:7]
        if not weak:
            return ("To build a plan I need evidence of what to work on — there are no published results or quiz "
                    "attempts yet. Attach your notes and ask me for MCQs to get started."), []
        system = ("You are Mentor AI. Build a realistic 7-day study plan ONLY around the weak topics given. "
                  "Use a Markdown table (Day | Focus | Tasks | Time). " + self._style_text(style))
        out = self._llm(system, f"Student: {student.name}\nRequest: {message}\nWeak topics: {weak}", 1200)
        if out:
            return out, ["Published exam results", "Practice quiz history"]
        rows = []
        for d in range(7):
            t = weak[d % len(weak)]
            task = ["Re-read notes + write definitions", "Solve 5 practice questions",
                    "Write one full exam-style answer", "Revise with flashcards"][d % 4]
            rows.append(f"| Day {d + 1} | {t} | {task} | 1.5 h |")
        rows[-1] = f"| Day 7 | All topics | Timed self-test on {', '.join(weak[:3])} | 2 h |"
        return ("### 🗓️ 7-day study plan\n\n| Day | Focus | Tasks | Time |\n|---|---|---|---|\n" + "\n".join(rows) +
                "\n\n_Built from your weak topics (offline mode — no LLM key configured)._",
                ["Published exam results", "Practice quiz history"])

    def generate_practice_quiz(self, student: Student, course_code: str, weak_topics: Optional[List[str]] = None) -> Dict:
        """
        Generates 3 customized practice questions focused on identified weak areas.
        """
        topics = weak_topics if weak_topics else ["Algorithms", "Database Normalization", "System Optimization"]
        topic_sample = ", ".join(topics[:3])

        questions = [
            {
                "id": 1,
                "type": "short_answer",
                "question": f"Explain the core principle of {topics[0] if topics else 'the concept'} and state its practical significance.",
                "max_marks": 5.0,
                "hint": "Start with a 1-sentence definition followed by an operational example."
            },
            {
                "id": 2,
                "type": "descriptive",
                "question": f"Compare and contrast the primary approaches used in {topics[1] if len(topics) > 1 else 'this topic'}. Highlight advantages and limitations.",
                "max_marks": 10.0,
                "hint": "Use a comparison table and state asymptotic complexities where applicable."
            },
            {
                "id": 3,
                "type": "scenario",
                "question": f"Given a real-world scenario with high data volume, how would you apply {topics[2] if len(topics) > 2 else 'this mechanism'} to ensure reliability and performance?",
                "max_marks": 5.0,
                "hint": "Discuss trade-offs, edge cases, and recovery."
            }
        ]

        return {
            "course_code": course_code,
            "targeted_topics": topics,
            "generated_quiz": questions,
            "instructions": "Submit answers directly to this mentor for automated feedback and scoring suggestions."
        }
