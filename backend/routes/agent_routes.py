"""AI Examination Assistant endpoints (Admin + Faculty). Read-only, role-scoped, Ollama-powered."""
import logging
from typing import List, Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from backend import access as AC
from backend.auth import require_role
from backend.database import User, get_db
from backend.exam_agent import ExamAgent, HELP, SqlRepo
from backend.llm_provider import get_llm_client

logger = logging.getLogger(__name__)
router = APIRouter()


class AgentTurn(BaseModel):
    role: str
    content: str


class AgentChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    history: List[AgentTurn] = []
    response_language: Optional[str] = "auto"
    job_id: Optional[int] = None          # the checking batch open in the robot UI (optional)


@router.get("/capabilities", summary="What the assistant can do for this role, and live AI status")
def capabilities(user: User = Depends(require_role(["admin", "faculty"])), db: Session = Depends(get_db)):
    h = get_llm_client().health()
    repo = SqlRepo(db, user)
    return {"role": user.role, "scope": repo.scope_description(), "help": HELP, "read_only": True,
            "actions": ["Check all uploaded answer sheets", "Checking progress", "Stop checking", "Class performance report"],
            "ai": {"provider": h.get("provider"), "model": h.get("model"), "ready": bool(h.get("reachable") and h.get("model_available")),
                   "error": h.get("error")},
            "admin_only": ["faculty activity", "all students / all examinations"] if repo.is_admin else []}


@router.post("/chat", summary="Ask the Examination Assistant")
def agent_chat(data: AgentChatRequest, user: User = Depends(require_role(["admin", "faculty"])),
               db: Session = Depends(get_db)):
    from backend import robot_agent
    kind = robot_agent.intent(data.message)
    if kind:   # robot actions on bulk checking (start / status / stop / report) — real backend work, audited
        out = robot_agent.handle(db, user, kind, data.job_id)
        try:
            AC.audit(db, user, "AGENT_QUERY", "agent", "", {"question": data.message[:200], "tools": out.get("tools"),
                                                            "mode": out.get("mode"), "job": out.get("job_id")})
            db.commit()
        except Exception:
            logger.exception("agent audit failed")
            db.rollback()
        return out
    agent = ExamAgent(SqlRepo(db, user), get_llm_client())
    out = agent.chat(data.message, [{"role": t.role, "content": t.content} for t in data.history][-8:], data.response_language or "auto")
    try:   # every assistant query is audited (who asked what; no answer content stored)
        AC.audit(db, user, "AGENT_QUERY", "agent", "", {"question": data.message[:200], "tools": out.get("tools"), "mode": out.get("mode")})
        db.commit()
    except Exception:       # auditing must never break the answer
        logger.exception("agent audit failed")
        db.rollback()
    return out
