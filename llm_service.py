"""
Unified LLM Service supporting Google Gemini and OpenAI.
Provides robust error handling for API timeouts, rate limits, invalid keys, and network issues.
"""
import logging
from typing import Dict, List, Optional, Tuple
import config

logger = logging.getLogger("llm_service")

# Lazy-loaded clients
_gemini_configured = False
_openai_client = None


def _init_gemini(api_key: str):
    global _gemini_configured
    if not _gemini_configured or api_key:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        _gemini_configured = True


def _get_openai_client(api_key: str):
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        _openai_client = OpenAI(api_key=api_key)
    return _openai_client


class LLMServiceError(Exception):
    """Custom exception containing user-friendly error messages."""
    def __init__(self, message: str, status_code: int = 500, details: Optional[str] = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.details = details or ""


def generate_llm_response(
    context_payload: Dict,
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> Tuple[str, Dict]:
    """
    Sends the prepared multi-turn context to the chosen LLM provider.
    Returns (response_text, usage_or_model_metadata).
    Raises LLMServiceError on handled failures.
    """
    active_provider = (provider or config.LLM_PROVIDER).lower()
    system_prompt = context_payload.get("system_prompt", config.SYSTEM_PROMPT)
    messages = context_payload.get("messages", [])

    if not messages:
        raise LLMServiceError("No messages provided to send to the LLM.", status_code=400)

    if active_provider == "gemini":
        return _call_gemini(system_prompt, messages, model_name=model or config.GEMINI_MODEL)
    elif active_provider == "openai":
        return _call_openai(system_prompt, messages, model_name=model or config.OPENAI_MODEL)
    elif active_provider == "mock":
        return _call_mock(system_prompt, messages)
    else:
        raise LLMServiceError(f"Unsupported provider: '{active_provider}'. Use 'gemini' or 'openai'.", status_code=400)


def _call_gemini(system_prompt: str, messages: List[Dict[str, str]], model_name: str) -> Tuple[str, Dict]:
    """Calls Google Gemini API with system instruction and conversational turns."""
    api_key = config.GEMINI_API_KEY
    if not api_key:
        raise LLMServiceError(
            "GEMINI_API_KEY is not configured in your .env file. "
            "Please obtain a free key from https://aistudio.google.com/ and set it in .env.",
            status_code=401,
        )

    try:
        import google.generativeai as genai
        from google.api_core import exceptions as google_exceptions

        _init_gemini(api_key)
        gemini_model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=system_prompt if system_prompt else None
        )

        # Format conversation messages for Gemini API
        contents = []
        for msg in messages:
            role = "user" if msg["role"] == "user" else "model"
            contents.append({
                "role": role,
                "parts": [msg["content"]]
            })

        response = gemini_model.generate_content(contents)
        text = response.text if hasattr(response, "text") and response.text else ""

        if not text:
            # Handle potential blocked or empty responses
            if hasattr(response, "prompt_feedback"):
                feedback = str(response.prompt_feedback)
                logger.warning(f"Gemini prompt feedback: {feedback}")
            text = "I received your message, but the response was filtered or empty."

        return text, {
            "provider": "gemini",
            "model": model_name,
            "turns_sent": len(messages)
        }

    except google_exceptions.InvalidArgument as e:
        logger.error(f"Gemini InvalidArgument: {e}")
        raise LLMServiceError(
            "Invalid API Key or configuration parameter for Google Gemini.",
            status_code=401,
            details=str(e)
        )
    except google_exceptions.ResourceExhausted as e:
        logger.error(f"Gemini Rate Limit (429): {e}")
        raise LLMServiceError(
            "Google Gemini API rate limit or quota exceeded. Please wait a moment and try again.",
            status_code=429,
            details=str(e)
        )
    except google_exceptions.NotFound as e:
        logger.error(f"Gemini Model Not Found (404): {e}")
        # Try a safe fallback like gemini-flash-latest or gemini-2.5-flash
        fallback = "gemini-2.5-flash" if model_name != "gemini-2.5-flash" else "gemini-flash-latest"
        raise LLMServiceError(
            f"Model '{model_name}' was not found. Try updating GEMINI_MODEL to '{fallback}' in your .env file.",
            status_code=404,
            details=str(e)
        )
    except google_exceptions.GoogleAPICallError as e:
        logger.error(f"Gemini API Error: {e}")
        raise LLMServiceError(
            f"Gemini API returned an error: {e.message or 'API call failed.'}",
            status_code=502,
            details=str(e)
        )
    except Exception as e:
        logger.error(f"Unexpected error calling Gemini: {e}")
        raise LLMServiceError(
            f"Failed to communicate with Gemini: {str(e)}",
            status_code=500,
            details=str(e)
        )


def _call_openai(system_prompt: str, messages: List[Dict[str, str]], model_name: str) -> Tuple[str, Dict]:
    """Calls OpenAI API with system instruction and conversational turns."""
    api_key = config.OPENAI_API_KEY
    if not api_key:
        raise LLMServiceError(
            "OPENAI_API_KEY is not configured in your .env file. "
            "Please add your OpenAI API key or switch LLM_PROVIDER to 'gemini'.",
            status_code=401,
        )

    try:
        from openai import AuthenticationError, RateLimitError, APIConnectionError, APITimeoutError
        client = _get_openai_client(api_key)

        formatted_messages = []
        if system_prompt:
            formatted_messages.append({"role": "system", "content": system_prompt})

        for msg in messages:
            role = msg["role"] if msg["role"] in ("user", "assistant", "system") else "user"
            formatted_messages.append({"role": role, "content": msg["content"]})

        completion = client.chat.completions.create(
            model=model_name,
            messages=formatted_messages,
            temperature=0.7,
        )

        response_text = completion.choices[0].message.content or ""
        usage = {
            "provider": "openai",
            "model": model_name,
            "prompt_tokens": getattr(completion.usage, "prompt_tokens", 0) if completion.usage else 0,
            "completion_tokens": getattr(completion.usage, "completion_tokens", 0) if completion.usage else 0,
            "total_tokens": getattr(completion.usage, "total_tokens", 0) if completion.usage else 0,
            "turns_sent": len(messages)
        }
        return response_text, usage

    except AuthenticationError as e:
        logger.error(f"OpenAI Authentication error: {e}")
        raise LLMServiceError(
            "Invalid OpenAI API Key. Please verify your OPENAI_API_KEY in .env.",
            status_code=401,
            details=str(e)
        )
    except RateLimitError as e:
        logger.error(f"OpenAI Rate limit error: {e}")
        raise LLMServiceError(
            "OpenAI API rate limit or quota exceeded. Please check your billing or wait a moment.",
            status_code=429,
            details=str(e)
        )
    except (APIConnectionError, APITimeoutError) as e:
        logger.error(f"OpenAI Connection timeout: {e}")
        raise LLMServiceError(
            "Connection to OpenAI API timed out. Please check your internet connectivity.",
            status_code=504,
            details=str(e)
        )
    except Exception as e:
        logger.error(f"Unexpected error calling OpenAI: {e}")
        raise LLMServiceError(
            f"Failed to communicate with OpenAI: {str(e)}",
            status_code=500,
            details=str(e)
        )


def _call_mock(system_prompt: str, messages: List[Dict[str, str]]) -> Tuple[str, Dict]:
    """Fallback mock provider for offline or test execution."""
    last_user_message = messages[-1]["content"] if messages else ""
    return (
        f"[Mock Mode]: I received your message: \"{last_user_message}\". "
        f"Context currently contains {len(messages)} messages. "
        f"(Configure your GEMINI_API_KEY in .env to connect to live LLM).",
        {"provider": "mock", "model": "mock-simulator", "turns_sent": len(messages)}
    )


def summarize_conversation_turns(existing_summary: str, older_messages: List[Dict[str, str]]) -> str:
    """
    Summarizes older conversation turns into a succinct memory card for running summarization.
    """
    if not older_messages:
        return existing_summary

    dialogue_text = ""
    for m in older_messages:
        role = "User" if m["role"] == "user" else "Assistant"
        dialogue_text += f"{role}: {m['content']}\n"

    prompt = (
        "You are an AI conversation archivist. Below is a prior running summary (if any) "
        "and a set of older dialogue turns from an ongoing conversation.\n"
        "Generate a concise, updated summary capturing key facts, user preferences, names, and decisions. "
        "Keep it under 100 words. Do not include pleasantries.\n\n"
        f"Prior Summary:\n{existing_summary or 'None'}\n\n"
        f"New Turns to Incorporate:\n{dialogue_text}\n\n"
        "Updated Summary:"
    )

    try:
        payload = {
            "system_prompt": "You are a concise conversation summarizer. Output only the factual summary.",
            "messages": [{"role": "user", "content": prompt}]
        }
        summary_text, _ = generate_llm_response(payload)
        return summary_text.strip()
    except Exception as e:
        logger.warning(f"Failed to generate summary via LLM: {e}")
        # Fallback heuristic summary if API call fails
        topics = [m["content"][:40] for m in older_messages if m["role"] == "user"]
        return (existing_summary + " | " if existing_summary else "") + f"Topics discussed: {', '.join(topics)}"
