"""
System status (prefix /system) — which AI engines are actually active, so the UI can tell teachers honestly
whether handwriting is read by a vision AI or by local OCR, and whether marking is semantic (AI) or offline.
"""
import os
import shutil

from fastapi import APIRouter, Depends

from backend.auth import require_role

router = APIRouter()


@router.get("/ai-status", summary="Active handwriting reader and answer grader")
def ai_status(_user=Depends(require_role(["admin", "faculty"]))):
    from backend.handwriting_reader import configured_providers, _model
    from backend.llm_provider import get_llm_client
    vision = configured_providers()
    client = get_llm_client()
    from backend.tesseract_locator import find_tesseract
    tess = find_tesseract() is not None
    reader = f"{vision[0]} ({_model(vision[0])})" if vision else ("local OCR (Tesseract)" if tess else "none")
    grader = client.active_provider if client.has_real_provider else "offline marking-point matching (flagged for review)"
    notes = []
    if not vision:
        notes.append("No vision-AI key: handwriting is read with local OCR, which is unreliable for cursive or messy "
                     "writing. Every answer is flagged for the teacher to check against the original page. Set one of "
                     "ANTHROPIC_API_KEY / GEMINI_API_KEY / OPENAI_API_KEY / GROQ_API_KEY in .env (optional cloud "
                     "vision). llama3.2:3b via Ollama is text-only; install a vision model (e.g. `ollama pull "
                     "llama3.2-vision`) and set HANDWRITING_PROVIDER=ollama to read handwriting locally.")
    if not client.has_real_provider:
        notes.append("Ollama is not reachable or the model is not installed: marks are estimated by matching the "
                     "answer to the marking points (wording, not meaning) and every answer is flagged for "
                     "verification. Start Ollama and run `ollama pull llama3.2:3b`.")
    if not tess:
        notes.append("Tesseract is not installed: the local OCR fallback is unavailable.")
    return {"handwriting_reader": reader, "vision_ai": bool(vision), "vision_providers": vision,
            "grader": grader, "semantic_grading": client.has_real_provider, "tesseract": tess, "notes": notes}


@router.get("/llm-status", summary="Live Ollama / LLM provider status")
def llm_status(refresh: bool = False):
    """Public, read-only: is the AI engine reachable and is the configured model installed?"""
    from backend.llm_provider import get_llm_client
    h = get_llm_client().health(force=refresh)
    return {"provider": h.get("provider"), "configured_provider": h.get("configured_provider"),
            "model": h.get("model"), "base_url": h.get("base_url"), "reachable": h.get("reachable"),
            "model_available": h.get("model_available"), "installed_models": h.get("installed_models"),
            "latency_ms": h.get("latency_ms"), "error": h.get("error"),
            "ready": bool(h.get("reachable") and h.get("model_available"))}
