"""
FastAPI Server for Kryptix LLM Chatbot.
Provides REST API endpoints for chat, history management, context stats, and serves the Web UI.
"""
import sys
from pathlib import Path
from typing import Optional

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import config
from context_manager import (
    get_or_create_session,
    clear_session,
    ConversationContext,
    _sessions,
)
from llm_service import (
    generate_llm_response,
    summarize_conversation_turns,
    LLMServiceError,
)

app = FastAPI(
    title="Kryptix LLM Context-Aware Chatbot",
    description="Multi-turn LLM Chatbot with Context Window Management (Sliding Window & Running Summarization)",
    version="1.0.0",
)

# Mount static directory
STATIC_DIR = BASE_DIR / "static"
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# Pydantic Request Models
class ChatRequest(BaseModel):
    conversation_id: Optional[str] = None
    message: str = Field(..., min_length=1, description="The user's prompt")
    strategy: Optional[str] = None  # 'sliding_window' or 'summarization'
    max_messages: Optional[int] = None
    system_prompt: Optional[str] = None


class ResetRequest(BaseModel):
    conversation_id: str


# Custom Exception Handler
@app.exception_handler(LLMServiceError)
async def llm_service_error_handler(request: Request, exc: LLMServiceError):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.message,
            "status_code": exc.status_code,
            "details": exc.details,
        },
    )


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    """Serves the main chat application UI."""
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        return HTMLResponse("<h1>Chatbot UI is loading...</h1>", status_code=200)
    return HTMLResponse(content=index_file.read_text(encoding="utf-8"))


@app.get("/api/status")
async def get_status():
    """Returns runtime status and configuration of the LLM provider."""
    return config.get_active_provider_info()


@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest):
    """
    Core conversational endpoint.
    Processes multi-turn dialogue with active context window management.
    """
    strategy = req.strategy or config.CONTEXT_STRATEGY
    max_msg = req.max_messages or config.MAX_CONTEXT_MESSAGES
    sys_prompt = req.system_prompt or config.SYSTEM_PROMPT

    session = get_or_create_session(
        conversation_id=req.conversation_id,
        system_prompt=sys_prompt,
        strategy=strategy,
        max_messages=max_msg,
    )

    # 1. Append user message
    session.add_message("user", req.message)

    # 2. Extract windowed context
    summarizer = summarize_conversation_turns if session.strategy == "summarization" else None
    context_payload = session.get_context_for_llm(summarizer_fn=summarizer)

    # 3. Call LLM Service
    try:
        reply_text, call_meta = generate_llm_response(context_payload)
    except LLMServiceError:
        raise
    except Exception as e:
        raise LLMServiceError(f"Unexpected server error during generation: {str(e)}", status_code=500)

    # 4. Append assistant response
    session.add_message("assistant", reply_text)

    return {
        "conversation_id": session.conversation_id,
        "reply": reply_text,
        "metadata": context_payload["metadata"],
        "call_meta": call_meta,
    }


@app.get("/api/history/{conversation_id}")
async def get_history(conversation_id: str):
    """Retrieves full conversation history and active context state."""
    if conversation_id not in _sessions:
        raise HTTPException(status_code=404, detail="Conversation session not found.")

    session = _sessions[conversation_id]
    return {
        "conversation_id": session.conversation_id,
        "strategy": session.strategy,
        "max_messages": session.max_messages,
        "messages": session.messages,
        "running_summary": session.running_summary,
        "total_messages": len(session.messages),
    }


@app.post("/api/reset")
async def reset_conversation(req: ResetRequest):
    """Clears history and context for a given conversation ID."""
    success = clear_session(req.conversation_id)
    return {"conversation_id": req.conversation_id, "cleared": success}


def start_server():
    print(f"Starting Kryptix LLM Chatbot server at http://{config.SERVER_HOST}:{config.SERVER_PORT}")
    uvicorn.run(
        "server:app",
        host=config.SERVER_HOST,
        port=config.SERVER_PORT,
        reload=False,
    )


if __name__ == "__main__":
    start_server()
