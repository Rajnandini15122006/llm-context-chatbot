"""
Context Window Management module.
Implements:
1. Multi-turn conversation tracking per session.
2. Strategy 1: Sliding Window Truncation (keeps system prompt + last N messages).
3. Strategy 2: Running Summarization (compresses older messages and prepends them).
"""
import uuid
from typing import List, Dict, Optional, Callable


class ConversationContext:
    def __init__(
        self,
        conversation_id: Optional[str] = None,
        system_prompt: str = "",
        strategy: str = "sliding_window",
        max_messages: int = 6,
    ):
        self.conversation_id = conversation_id or str(uuid.uuid4())
        self.system_prompt = system_prompt
        self.strategy = strategy.lower()  # 'sliding_window' or 'summarization'
        self.max_messages = max(2, max_messages)  # Ensure at least 1 turn
        self.messages: List[Dict[str, str]] = []
        self.running_summary: str = ""
        self.summarized_up_to_index: int = 0

    def add_message(self, role: str, content: str):
        """Adds a message (user or assistant) to conversation history."""
        self.messages.append({
            "role": role,
            "content": content
        })

    def reset(self):
        """Clears all conversation turns and resets running summary."""
        self.messages = []
        self.running_summary = ""
        self.summarized_up_to_index = 0

    def get_context_for_llm(
        self,
        summarizer_fn: Optional[Callable[[str, List[Dict[str, str]]], str]] = None
    ) -> Dict:
        """
        Prepares the context payload for the LLM according to the chosen strategy.
        Returns a dict containing:
          - 'system_prompt': effective system prompt (including running summary if present)
          - 'messages': messages to send to the LLM
          - 'metadata': info about truncation, summary, and message counts
        """
        total_count = len(self.messages)

        if self.strategy == "summarization":
            return self._build_summarized_context(summarizer_fn, total_count)
        else:
            return self._build_sliding_window_context(total_count)

    def _build_sliding_window_context(self, total_count: int) -> Dict:
        """Strategy 1: Sliding window truncation keeping the most recent N messages."""
        if total_count <= self.max_messages:
            active_messages = list(self.messages)
            truncated_count = 0
        else:
            active_messages = self.messages[-self.max_messages:]
            truncated_count = total_count - self.max_messages

        return {
            "system_prompt": self.system_prompt,
            "messages": active_messages,
            "metadata": {
                "strategy": "sliding_window",
                "total_messages": total_count,
                "retained_messages": len(active_messages),
                "truncated_messages": truncated_count,
                "max_allowed": self.max_messages,
                "running_summary": "",
            }
        }

    def _build_summarized_context(
        self,
        summarizer_fn: Optional[Callable[[str, List[Dict[str, str]]], str]],
        total_count: int
    ) -> Dict:
        """
        Strategy 2: Running summarization.
        Older messages exceeding the max window are summarized into an ongoing summary.
        The summary is prepended to the system prompt or active context.
        """
        if total_count <= self.max_messages:
            active_messages = list(self.messages)
            truncated_count = 0
        else:
            # Messages that fall outside the active window
            overflow_cutoff = total_count - self.max_messages
            messages_to_summarize = self.messages[self.summarized_up_to_index:overflow_cutoff]

            if messages_to_summarize and summarizer_fn:
                try:
                    new_summary = summarizer_fn(self.running_summary, messages_to_summarize)
                    if new_summary and new_summary.strip():
                        self.running_summary = new_summary.strip()
                        self.summarized_up_to_index = overflow_cutoff
                except Exception as e:
                    # Fallback gracefully if summarization call fails
                    print(f"[ContextManager] Summarization error: {e}")

            active_messages = self.messages[-self.max_messages:]
            truncated_count = total_count - len(active_messages)

        # Prepend the running summary to give the model continuity
        effective_system_prompt = self.system_prompt
        if self.running_summary:
            effective_system_prompt += (
                f"\n\n[CONTEXT MEMORY CARD - SUMMARY OF PREVIOUS TURNS]\n"
                f"{self.running_summary}\n"
                f"[END CONTEXT MEMORY CARD]"
            )

        return {
            "system_prompt": effective_system_prompt,
            "messages": active_messages,
            "metadata": {
                "strategy": "summarization",
                "total_messages": total_count,
                "retained_messages": len(active_messages),
                "truncated_messages": truncated_count,
                "max_allowed": self.max_messages,
                "running_summary": self.running_summary,
            }
        }

    def get_stats(self) -> Dict:
        """Returns statistics about current session context."""
        return {
            "conversation_id": self.conversation_id,
            "total_messages": len(self.messages),
            "max_messages": self.max_messages,
            "strategy": self.strategy,
            "has_summary": bool(self.running_summary),
            "summary_snippet": (self.running_summary[:120] + "...") if len(self.running_summary) > 120 else self.running_summary,
        }


# Session store for multi-turn conversations
_sessions: Dict[str, ConversationContext] = {}


def get_or_create_session(
    conversation_id: Optional[str] = None,
    system_prompt: str = "",
    strategy: str = "sliding_window",
    max_messages: int = 6,
) -> ConversationContext:
    """Retrieves an existing conversation session or initializes a new one."""
    if not conversation_id or conversation_id not in _sessions:
        new_id = conversation_id or str(uuid.uuid4())
        _sessions[new_id] = ConversationContext(
            conversation_id=new_id,
            system_prompt=system_prompt,
            strategy=strategy,
            max_messages=max_messages
        )
        return _sessions[new_id]
    
    session = _sessions[conversation_id]
    if system_prompt:
        session.system_prompt = system_prompt
    if strategy:
        session.strategy = strategy
    if max_messages:
        session.max_messages = max_messages
    return session


def clear_session(conversation_id: str) -> bool:
    """Removes a session from memory."""
    if conversation_id in _sessions:
        del _sessions[conversation_id]
        return True
    return False
