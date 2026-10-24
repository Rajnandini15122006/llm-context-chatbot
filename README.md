# 🤖 Kryptix AI: Multi-Turn LLM Chatbot with Context Window Management

[![FastAPI](https://img.shields.io/badge/FastAPI-0.116+-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Google Gemini](https://img.shields.io/badge/Provider-Google%20Gemini-4285F4.svg?logo=google&logoColor=white)](https://ai.google.dev/)
[![OpenAI](https://img.shields.io/badge/Provider-OpenAI%20(Optional)-412991.svg?logo=openai&logoColor=white)](https://openai.com/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

An enterprise-grade, multi-turn AI chatbot featuring real-time **Context Window Management**, dual provider support (**Google Gemini** & **OpenAI**), customizable system prompts, robust error handling, and both a sleek **Interactive Web UI** and **Terminal CLI**.

Developed for the **Valentius Kryptix Intern Task: AI-1 (LLM API Integration)**.

---

## 🌟 Key Features

- **Multi-Turn Conversational Memory**: Preserves session state and history so the model reliably recalls details mentioned turns earlier.
- **Context Window Management (Dual Strategy)**:
  - 🪟 **Sliding Window (Truncation)**: Retains the foundational system instruction and the most recent $N$ dialogue turns, discarding older messages to strictly bound token consumption and avoid context overflow.
  - 🧠 **Running Summarization (Stretch Goal)**: When dialogue exceeds $N$ turns, an automated LLM summarization pipeline compresses older turns into an executive memory card and prepends it to the active window.
- **Defined Persona (System Prompt)**: Configured as **Kryptix AI**, a Senior AI Architect & Engineering Mentor delivering structured, pragmatic technical guidance.
- **Resilient Error Handling**: Gracefully catches rate limits (HTTP 429), authentication failures (HTTP 401/403), connection timeouts, and model availability issues with actionable user notices.
- **Dual Interfaces**:
  - 🖥️ **Modern Dark-Mode Web App**: Live chat stream with Markdown rendering, typing indicator, and a real-time **Context Inspector Sidebar** displaying active turns, truncation count, and memory cards.
  - 💻 **Terminal CLI (`cli.py`)**: Terminal chat with slash commands (`/stats`, `/strategy`, `/window`, `/clear`, `/exit`).
- **Security-First**: Zero hardcoded secrets. Environment variables managed via `.env` and strictly guarded by `.gitignore`.

---

## 🏗️ Architecture & Context Management

```
                       +---------------------------------------+
                       |           User Message                |
                       +---------------------------------------+
                                           |
                                           v
                       +---------------------------------------+
                       |       Session History Store           |
                       +---------------------------------------+
                                           |
                        +------------------+------------------+
                        |                                     |
           [Sliding Window Strategy]             [Summarization Strategy]
                        |                                     |
                        v                                     v
         +-----------------------------+       +-----------------------------+
         | Keep system instruction     |       | Summarize turns > N via LLM |
         | + Slice last N messages     |       | Prepend Memory Card to      |
         | Discard older turns         |       | system prompt + last N turns|
         +-----------------------------+       +-----------------------------+
                        |                                     |
                        +------------------+------------------+
                                           |
                                           v
                       +---------------------------------------+
                       |         Unified LLM Service           |
                       |    (Google Gemini / OpenAI / Mock)    |
                       +---------------------------------------+
                                           |
                                           v
                       +---------------------------------------+
                       |   Response + Context Stats Payload    |
                       +---------------------------------------+
```

### Why Context Window Management Matters

LLMs have finite context limits (and compute costs scale quadratically with context length in naive attention mechanisms). Unchecked message histories lead to:
1. **Token Exhaustion / Rate Limits**: Breaking API constraints.
2. **Context Degradation ("Lost in the Middle")**: Models struggle to attend to key facts when overwhelmed by long backlogs.
3. **Escalating Latency & Costs**: Sending thousands of repetitive tokens on every turn.

This project implements:
1. **Sliding Window**: Bounds maximum payload size to $O(N)$ messages, preserving recent turns with $100\%$ fidelity.
2. **Running Summarization**: Compresses historical context to preserve key decisions, user preferences, and named entities without context bloat.

---

## 🚀 Quickstart Guide

### 1. Prerequisites
- Python 3.10 or higher
- Git

### 2. Clone the Repository
```bash
git clone <YOUR_PUBLIC_GITHUB_REPO_URL>
cd llm-context-chatbot
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Copy the template and edit your `.env`:
```bash
cp .env.example .env
```
Open `.env` and add your API key:
```env
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_google_gemini_api_key_here
GEMINI_MODEL=gemini-2.5-flash

# Optional: OpenAI
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4o-mini

CONTEXT_STRATEGY=sliding_window
MAX_CONTEXT_MESSAGES=6
```
> 💡 **Get a free Google Gemini API Key**: Visit [Google AI Studio](https://aistudio.google.com/) to generate a free API key in seconds.

---

## 💻 Running the Application

### Option A: Modern Web Interface (FastAPI + HTML/CSS/JS)
```bash
python server.py
```
Open your browser at: **[http://localhost:8000](http://localhost:8000)**

#### Features in Web UI:
- **Interactive Chat**: Chat with Kryptix AI with Markdown code highlights.
- **Context Inspector**: Click **"Context Inspector"** in the top right to watch the sliding window and running summary update live!
- **Dynamic Controls**: Adjust window size ($N$) or switch between Sliding Window and Summarization on the fly.
- **New Chat**: Instantly reset conversation and clear memory.

### Option B: Terminal CLI Interface
```bash
python cli.py
```
#### CLI Slash Commands:
- `/stats` — View current context window metrics and active memory summary.
- `/strategy sliding_window` or `/strategy summarization` — Switch context strategy.
- `/window 8` — Change active window size to 8 messages.
- `/clear` — Reset session history.
- `/exit` — Quit terminal.

---

## 🧪 Automated Verification Suite

Run the full end-to-end verification test suite:
```bash
python test_context.py
```

### Test Suite Results:
- ✅ **Test 1: Multi-Turn Context Retention**: Confirms the model retains the user's name, preferred language, and project title across 5 consecutive turns.
- ✅ **Test 2: Sliding Window Truncation**: Validates that 12 messages are truncated to the configured limit (4) while discarding older turns.
- ✅ **Test 3: Running Summarization**: Verifies that overflowing dialogue turns are condensed into a memory card prepended to the system prompt.
- ✅ **Test 4: Error Handling**: Confirms graceful error responses (HTTP 400/401) on empty payloads and invalid providers without crashing.

---

## 📝 5-Turn Conversation Proof (Transcript)

Here is a verified multi-turn transcript demonstrating context persistence:

```
[Turn 1]
User: Hi! My name is Rajnandini, and my favorite programming language is Rust.
Kryptix AI: Hello Rajnandini! It's a pleasure to meet you. Rust is an excellent choice for systems programming and high-performance applications; its focus on safety and speed is truly impressive. I'm Kryptix, your Senior AI Architect mentor. What are we building today?

[Turn 2]
User: Can you explain what an API rate limit is in two sentences?
Kryptix AI: An API rate limit restricts the number of requests a user or client can make to an API within a specific timeframe. Its primary purpose is to prevent abuse, ensure fair usage among all consumers, and protect backend infrastructure from being overwhelmed.

[Turn 3]
User: Also, I am building an automated trading bot named 'QuantumLedger'.
Kryptix AI: That's fantastic, Rajnandini! 'QuantumLedger' sounds like an ambitious and exciting project. Building an automated trading bot will definitely leverage Rust's strengths in performance and memory safety.

[Turn 4]
User: What is the difference between synchronous and asynchronous code?
Kryptix AI: Synchronous execution blocks subsequent tasks until the current one completes, creating sequential bottlenecks. Asynchronous execution allows tasks to initiate and run concurrently without waiting, making it indispensable for high-throughput I/O in applications like 'QuantumLedger'.

[Turn 5]
User: Can you recall my name, my favorite language, and what project I am building?
Kryptix AI: Absolutely, Rajnandini! You are Rajnandini, your preferred language is Rust, and you are building 'QuantumLedger', an automated trading bot.
```

---

## 🛡️ Interview Defense Guide (AI Usage Declaration)

If asked about the architecture and design decisions in an interview, here are key defense points:

### Q1: Why did you implement context window management rather than sending all past messages?
> **Answer**: Sending unlimited conversation history causes three critical failure modes:
> 1. It eventually exceeds the model's context window limit ($N_{\text{max}}$), throwing hard API errors.
> 2. API costs and latency scale with token count. Re-sending redundant early messages on every turn is expensive and slow.
> 3. Models suffer from the "lost in the middle" phenomenon, where attention degrades when relevant instructions are buried under massive conversational logs. Managing context with a sliding window or running summary keeps token budgets bounded while preserving recent intent.

### Q2: What is the trade-off between Sliding Window Truncation and Running Summarization?
> **Answer**:
> - **Sliding Window (Truncation)** is computationally deterministic, fast ($O(1)$ extra compute), and zero-cost (no extra LLM calls). However, it completely loses facts mentioned earlier than turn $N$.
> - **Running Summarization** preserves long-term continuity by condensing older facts into a compact memory block. However, it requires an auxiliary LLM summarization call when history overflows, slightly increasing latency and token usage during summary updates.

### Q3: How is security handled for API keys?
> **Answer**: Keys are never hardcoded in source files. They are injected via environment variables through `.env`, and `.env` is explicitly registered in `.gitignore` prior to the first git commit. A sanitized `.env.example` is committed so collaborators understand the expected configuration without exposing credentials.

### Q4: How does error handling prevent broken user interfaces?
> **Answer**: The backend catches vendor-specific exceptions (`ResourceExhausted` for 429 rate limits, `InvalidArgument` for 401 key errors, and network timeouts) and translates them into uniform `LLMServiceError` objects with HTTP status codes and human-readable recovery guidance. The frontend listens for non-200 responses and renders an interactive dismissible banner rather than hanging in a loading state.

---

## 📁 Repository Structure

```
llm-context-chatbot/
├── .env.example          # Environment variable template
├── .gitignore            # Git exclusion file (protects .env and cache)
├── requirements.txt      # Python dependencies
├── config.py             # Configuration loader & system prompt
├── context_manager.py    # Sliding window & running summarization logic
├── llm_service.py        # Unified Gemini & OpenAI API connector with error handling
├── server.py             # FastAPI REST backend & static file server
├── cli.py                # Terminal interactive chat
├── test_context.py       # 4-stage automated verification suite
├── static/
│   ├── index.html        # Modern chat interface + Context Inspector
│   ├── style.css         # Responsive dark-theme styling
│   └── app.js            # Frontend chat client & state management
└── README.md             # Project documentation & interview defense
```

---

## 📄 License

MIT License. Developed for the Valentius Kryptix Internship Program.
