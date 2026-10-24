"""
Configuration module for the Kryptix LLM Chatbot.
Loads settings from environment variables with safe fallbacks.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from project root
BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH)

# Provider configuration
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").strip().lower()

# Google Gemini settings
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()

# OpenAI settings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()

# Context window management settings
# Options: 'sliding_window' (truncation) or 'summarization'
CONTEXT_STRATEGY = os.getenv("CONTEXT_STRATEGY", "sliding_window").strip().lower()

# Maximum messages to keep in the active sliding window
try:
    MAX_CONTEXT_MESSAGES = int(os.getenv("MAX_CONTEXT_MESSAGES", "6"))
except ValueError:
    MAX_CONTEXT_MESSAGES = 6

# Server configuration
SERVER_HOST = os.getenv("SERVER_HOST", "127.0.0.1")
try:
    SERVER_PORT = int(os.getenv("SERVER_PORT", "8000"))
except ValueError:
    SERVER_PORT = 8000

# Defined System Prompt (gives the bot a distinct role and personality)
SYSTEM_PROMPT = """You are Kryptix AI, a razor-sharp, supportive, and knowledgeable Senior AI Architect & Engineering Mentor at Valentius Kryptix.
Your purpose:
1. Guide developers through building robust, scalable AI applications, APIs, and modern software architectures.
2. Maintain strong conversational continuity: always remember and explicitly reference details the user shared earlier in the discussion (e.g. their project names, favorite languages, past questions).
3. Deliver answers that are concise, structured, pragmatic, and technically accurate.
4. If code is requested, provide clean, idiomatic snippets with brief explanations.
5. Keep an encouraging, professional, and mentor-like tone."""


def get_active_provider_info() -> dict:
    """Returns runtime details about the active provider and configuration."""
    provider = LLM_PROVIDER
    if provider == "gemini":
        has_key = bool(GEMINI_API_KEY)
        model = GEMINI_MODEL
    elif provider == "openai":
        has_key = bool(OPENAI_API_KEY)
        model = OPENAI_MODEL
    else:
        has_key = False
        model = "unknown"

    return {
        "provider": provider,
        "model": model,
        "has_api_key": has_key,
        "context_strategy": CONTEXT_STRATEGY,
        "max_context_messages": MAX_CONTEXT_MESSAGES,
    }
