"""
main.py – FastAPI server for HiveRift Intelligence Brain (Full-KB Edition)
"""

import os
import json
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

app = FastAPI(title="HiveRift Intelligence Brain", version="3.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.isdir(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/", include_in_schema=False)
async def serve_ui():
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.isfile(index_path):
        return FileResponse(index_path)
    return {"message": "HiveRift Brain API running."}


class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    reply: str
    phase: str = ""


# ── /chat ─────────────────────────────────────────────────────────────────────
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if not req.message.strip():
        return ChatResponse(reply="Please enter a question.")
    from backend.brain import query, _detect_phase
    user_msg = req.message.strip()
    return ChatResponse(reply=query(user_msg), phase=_detect_phase(user_msg))


# ── /chat/stream ──────────────────────────────────────────────────────────────
@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    if not req.message.strip():
        async def _empty():
            yield 'data: {"token": "Please enter a question."}\n\n'
            yield 'data: {"done": true}\n\n'
        return StreamingResponse(_empty(), media_type="text/event-stream")

    user_msg = req.message.strip()

    def _generate():
        from backend.brain import query_stream, _detect_phase
        yield f'data: {json.dumps({"phase": _detect_phase(user_msg)})}\n\n'
        for token in query_stream(user_msg):
            yield f'data: {json.dumps({"token": token})}\n\n'
        yield 'data: {"done": true}\n\n'

    return StreamingResponse(_generate(), media_type="text/event-stream")


# ── /reset ────────────────────────────────────────────────────────────────────
@app.post("/reset")
async def reset():
    from backend.brain import clear_memory
    clear_memory()
    return {"status": "Conversation memory cleared."}


# ── /health ───────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    kb_path = os.path.join(os.path.dirname(__file__), "..", "Knowledge Base", "hiverift_source_of_truth.md")
    hf_token_set = bool(os.environ.get("GROQ_API_KEY", ""))
    return {
        "status": "ok",
        "version": "3.1.0 – Full-KB Edition",
        "knowledge_base_loaded": os.path.isfile(kb_path),
        "groq_key_configured": hf_token_set,
        "mode": "Llama-3 reads full KB" if hf_token_set else "⚠️ GROQ_API_KEY not set",
    }
