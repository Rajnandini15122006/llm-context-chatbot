"""
Interactive Command-Line Interface (CLI) for the Kryptix LLM Chatbot.
Supports multi-turn chat, context window inspection, on-the-fly strategy switching, and history clearing.
"""
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from context_manager import get_or_create_session, clear_session
from llm_service import generate_llm_response, summarize_conversation_turns, LLMServiceError


def print_banner():
    print("""
======================================================================
  🤖 KRYPTIX LLM CHATBOT - CONTEXT-AWARE TERMINAL
======================================================================
  Active Provider : {provider} ({model})
  Context Strategy: {strategy}
  Max Window Size : {max_msg} messages
----------------------------------------------------------------------
  Commands:
    /stats               Inspect context window & memory stats
    /strategy [type]     Switch strategy ('sliding_window' or 'summarization')
    /window [N]          Set active window message limit
    /clear               Clear conversation history and reset
    /exit                Exit chat
======================================================================
""".format(
        provider=config.LLM_PROVIDER.upper(),
        model=config.GEMINI_MODEL if config.LLM_PROVIDER == "gemini" else config.OPENAI_MODEL,
        strategy=config.CONTEXT_STRATEGY,
        max_msg=config.MAX_CONTEXT_MESSAGES
    ))


def run_cli():
    session_id = "cli-session"
    session = get_or_create_session(
        conversation_id=session_id,
        system_prompt=config.SYSTEM_PROMPT,
        strategy=config.CONTEXT_STRATEGY,
        max_messages=config.MAX_CONTEXT_MESSAGES,
    )

    print_banner()

    turn_count = 0
    while True:
        try:
            user_input = input("\nYou > ").strip()
            if not user_input:
                continue

            # Command handling
            if user_input.startswith("/"):
                cmd_parts = user_input.split(maxsplit=1)
                cmd = cmd_parts[0].lower()
                arg = cmd_parts[1].strip() if len(cmd_parts) > 1 else ""

                if cmd in ("/exit", "/quit"):
                    print("Exiting chat. Goodbye!")
                    break

                elif cmd == "/clear":
                    session.reset()
                    turn_count = 0
                    print("🔄 Conversation history and memory context cleared.")
                    continue

                elif cmd == "/stats":
                    stats = session.get_stats()
                    print("\n--- Context Window Statistics ---")
                    print(f"  Session ID       : {stats['conversation_id']}")
                    print(f"  Total Messages   : {stats['total_messages']}")
                    print(f"  Max Context Limit: {stats['max_messages']}")
                    print(f"  Active Strategy  : {stats['strategy']}")
                    print(f"  Running Summary  : {stats['has_summary']}")
                    if stats["has_summary"]:
                        print(f"  Summary Content  : {session.running_summary}")
                    print("---------------------------------\n")
                    continue

                elif cmd == "/strategy":
                    if arg in ("sliding_window", "summarization"):
                        session.strategy = arg
                        print(f" Context strategy switched to: {arg}")
                    else:
                        print(" Invalid strategy. Choose 'sliding_window' or 'summarization'.")
                    continue

                elif cmd == "/window":
                    try:
                        n = int(arg)
                        if n < 2:
                            print(" Window size must be at least 2.")
                        else:
                            session.max_messages = n
                            print(f" Context window size updated to {n} messages.")
                    except ValueError:
                        print(" Please provide a valid integer. Example: /window 6")
                    continue

                elif cmd == "/help":
                    print_banner()
                    continue

                else:
                    print(f" Unknown command '{cmd}'. Type /help for available commands.")
                    continue

            # Process user chat turn
            turn_count += 1
            session.add_message("user", user_input)

            # Build context according to active strategy
            payload = session.get_context_for_llm(
                summarizer_fn=summarize_conversation_turns if session.strategy == "summarization" else None
            )

            meta = payload["metadata"]
            if meta["truncated_messages"] > 0:
                print(f"  [Context Window: {meta['retained_messages']}/{meta['total_messages']} messages retained, {meta['truncated_messages']} truncated]")

            # Call LLM
            reply_text, call_meta = generate_llm_response(payload)
            session.add_message("assistant", reply_text)

            print(f"\nKryptix AI > {reply_text.strip()}")

        except KeyboardInterrupt:
            print("\nSession interrupted. Goodbye!")
            break
        except LLMServiceError as e:
            print(f"\n⚠️  [LLM Error {e.status_code}]: {e.message}")
            if e.details:
                print(f"   Details: {e.details}")
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")


if __name__ == "__main__":
    run_cli()
