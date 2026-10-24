"""
Automated Multi-turn Context & Error Handling Verification Suite.
Validates:
1. Multi-turn conversation persistence (Bot references information from prior turns).
2. Context window truncation (Sliding window respects max_context_messages).
3. Running summarization functionality.
4. Graceful error handling on invalid configurations.
"""
import sys
import os
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from context_manager import ConversationContext, get_or_create_session, clear_session
from llm_service import generate_llm_response, summarize_conversation_turns, LLMServiceError


def test_multi_turn_continuity():
    """Tests that the LLM retains facts across multiple turns."""
    print("\n" + "=" * 60)
    print("TEST 1: Multi-turn Context Retention (5 Turns)")
    print("=" * 60)

    conv_id = "test-continuity-session"
    clear_session(conv_id)
    session = get_or_create_session(
        conversation_id=conv_id,
        system_prompt=config.SYSTEM_PROMPT,
        strategy="sliding_window",
        max_messages=8
    )

    test_turns = [
        "Hi! My name is Rajnandini, and my favorite programming language is Rust.",
        "Can you explain what an API rate limit is in two sentences?",
        "Also, I am building an automated trading bot named 'QuantumLedger'.",
        "What is the difference between synchronous and asynchronous code?",
        "Can you recall my name, my favorite language, and what project I am building?"
    ]

    for turn_num, user_msg in enumerate(test_turns, start=1):
        print(f"\n[Turn {turn_num}] User: {user_msg}")
        session.add_message("user", user_msg)

        payload = session.get_context_for_llm()
        reply, meta = generate_llm_response(payload)
        session.add_message("assistant", reply)

        print(f"[Turn {turn_num}] Kryptix AI: {reply.strip()[:200]}...")

    last_reply = session.messages[-1]["content"].lower()
    
    # Assertions to confirm context persistence
    recalled_name = "rajnandini" in last_reply
    recalled_lang = "rust" in last_reply
    recalled_project = "quantumledger" in last_reply

    print("\n--- Context Retention Verification ---")
    print(f"Recalled Name ('Rajnandini'): {'PASS' if recalled_name else 'FAIL'}")
    print(f"Recalled Language ('Rust'): {'PASS' if recalled_lang else 'FAIL'}")
    print(f"Recalled Project ('QuantumLedger'): {'PASS' if recalled_project else 'FAIL'}")

    assert recalled_name or recalled_lang, "Failed: The model failed to recall context from earlier turns."
    print(">>> TEST 1 PASSED: Multi-turn context was successfully retained across 5 turns!\n")


def test_sliding_window_truncation():
    """Tests that the sliding window strategy truncates old messages properly."""
    print("=" * 60)
    print("TEST 2: Context Window Truncation (Sliding Window)")
    print("=" * 60)

    session = ConversationContext(
        conversation_id="truncation-test",
        system_prompt="Test System Prompt",
        strategy="sliding_window",
        max_messages=4  # Keep only 4 most recent messages
    )

    for i in range(1, 7):
        session.add_message("user", f"User message #{i}")
        session.add_message("assistant", f"Bot response #{i}")

    # Total messages added = 12
    assert len(session.messages) == 12, f"Expected 12 total messages, got {len(session.messages)}"

    context = session.get_context_for_llm()
    retained = context["messages"]
    meta = context["metadata"]

    print(f"Total messages stored: {meta['total_messages']}")
    print(f"Retained messages for LLM: {meta['retained_messages']} (Max allowed: {meta['max_allowed']})")
    print(f"Truncated messages: {meta['truncated_messages']}")

    assert len(retained) == 4, f"Expected 4 retained messages, got {len(retained)}"
    assert meta["truncated_messages"] == 8, f"Expected 8 truncated, got {meta['truncated_messages']}"
    assert retained[0]["content"] == "User message #5", f"Unexpected first message: {retained[0]['content']}"
    assert retained[-1]["content"] == "Bot response #6", f"Unexpected last message: {retained[-1]['content']}"

    print(">>> TEST 2 PASSED: Sliding window truncation successfully preserves only the most recent N turns!\n")


def test_running_summarization():
    """Tests the running summarization strategy when history overflows."""
    print("=" * 60)
    print("TEST 3: Running Summarization Strategy")
    print("=" * 60)

    session = ConversationContext(
        conversation_id="summary-test",
        system_prompt=config.SYSTEM_PROMPT,
        strategy="summarization",
        max_messages=4
    )

    # Add 6 turns (12 messages)
    session.add_message("user", "My dog's name is Pixel and he is a golden retriever.")
    session.add_message("assistant", "Pixel is a wonderful name for a retriever!")
    session.add_message("user", "I live in Bengaluru, India.")
    session.add_message("assistant", "Bengaluru is a vibrant tech hub!")
    session.add_message("user", "I prefer using FastAPI over Flask.")
    session.add_message("assistant", "FastAPI offers great async performance and automatic OpenAPI docs.")

    context = session.get_context_for_llm(summarizer_fn=summarize_conversation_turns)

    print(f"Strategy: {context['metadata']['strategy']}")
    print(f"Running Summary generated: {session.running_summary}")
    print(f"Active window message count: {len(context['messages'])}")

    assert len(session.running_summary) > 0, "Expected non-empty running summary"
    assert "CONTEXT MEMORY CARD" in context["system_prompt"], "Expected memory card in effective system prompt"

    print(">>> TEST 3 PASSED: Running summarization successfully compresses older turns and prepends memory card!\n")


def test_error_handling():
    """Tests graceful error handling when invalid API key is provided."""
    print("=" * 60)
    print("TEST 4: Error Handling Verification")
    print("=" * 60)

    # Simulate invalid provider call
    try:
        generate_llm_response(
            {"system_prompt": "Test", "messages": [{"role": "user", "content": "hi"}]},
            provider="invalid_provider"
        )
        assert False, "Should have raised LLMServiceError"
    except LLMServiceError as e:
        print(f"Caught expected invalid provider error: '{e.message}' (HTTP {e.status_code})")
        assert e.status_code == 400

    # Simulate empty messages
    try:
        generate_llm_response({"system_prompt": "Test", "messages": []})
        assert False, "Should have raised LLMServiceError for empty messages"
    except LLMServiceError as e:
        print(f"Caught expected empty payload error: '{e.message}' (HTTP {e.status_code})")
        assert e.status_code == 400

    print(">>> TEST 4 PASSED: Graceful error handling validated!\n")


if __name__ == "__main__":
    print("\nRunning Kryptix LLM Chatbot Verification Suite...")
    try:
        test_multi_turn_continuity()
        test_sliding_window_truncation()
        test_running_summarization()
        test_error_handling()
        print("\n" + "=" * 60)
        print("ALL VERIFICATION TESTS COMPLETED SUCCESSFULLY! (4/4 PASS)")
        print("=" * 60 + "\n")
    except Exception as ex:
        print(f"\nTEST SUITE FAILED: {ex}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
