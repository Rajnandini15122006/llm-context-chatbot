// Kryptix Chatbot Client App
document.addEventListener("DOMContentLoaded", () => {
  // Elements
  const userInput = document.getElementById("userInput");
  const sendBtn = document.getElementById("sendBtn");
  const messagesArea = document.getElementById("messagesArea");
  const welcomeCard = document.getElementById("welcomeCard");
  const charCounter = document.getElementById("charCounter");
  const resetChatBtn = document.getElementById("resetChatBtn");
  const toggleInspectorBtn = document.getElementById("toggleInspectorBtn");
  const closeInspectorBtn = document.getElementById("closeInspectorBtn");
  const inspectorSidebar = document.getElementById("inspectorSidebar");
  const errorBanner = document.getElementById("errorBanner");
  const errorMessage = document.getElementById("errorMessage");
  const closeErrorBtn = document.getElementById("closeErrorBtn");

  // Metrics elements
  const metricTotal = document.getElementById("metricTotal");
  const metricRetained = document.getElementById("metricRetained");
  const metricTruncated = document.getElementById("metricTruncated");
  const metricLimit = document.getElementById("metricLimit");
  const windowSlider = document.getElementById("windowSlider");
  const sliderValueBadge = document.getElementById("sliderValueBadge");
  const summaryCardContent = document.getElementById("summaryCardContent");
  const sessionIdDisplay = document.getElementById("sessionIdDisplay");
  const providerNameDisplay = document.getElementById("providerNameDisplay");
  const modelNameDisplay = document.getElementById("modelNameDisplay");
  const providerBadge = document.getElementById("providerBadge");

  // State
  let conversationId = "session-" + Math.random().toString(36).substring(2, 9);
  let isGenerating = false;

  // Initialize
  sessionIdDisplay.textContent = conversationId;
  initApp();

  async function initApp() {
    try {
      const res = await fetch("/api/status");
      if (res.ok) {
        const data = await res.json();
        providerNameDisplay.textContent = data.provider;
        modelNameDisplay.textContent = data.model;
        providerBadge.textContent = `${data.provider.toUpperCase()} (${data.model})`;

        if (data.max_context_messages) {
          windowSlider.value = data.max_context_messages;
          sliderValueBadge.textContent = data.max_context_messages;
          metricLimit.textContent = data.max_context_messages;
        }

        const strategyRadio = document.querySelector(`input[name="strategyOption"][value="${data.context_strategy}"]`);
        if (strategyRadio) strategyRadio.checked = true;
      }
    } catch (err) {
      console.warn("Could not fetch server status:", err);
      providerBadge.textContent = "Offline / Server Error";
    }
  }

  // Auto-resize textarea & character counter
  userInput.addEventListener("input", () => {
    userInput.style.height = "auto";
    userInput.style.height = Math.min(userInput.scrollHeight, 160) + "px";
    charCounter.textContent = `${userInput.value.length} / 2000`;
  });

  // Handle enter key (Enter to send, Shift+Enter for new line)
  userInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });

  sendBtn.addEventListener("click", sendMessage);

  // Chips click handler
  document.querySelectorAll(".chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      const prompt = chip.getAttribute("data-prompt");
      if (prompt) {
        userInput.value = prompt;
        sendMessage();
      }
    });
  });

  // Slider change
  windowSlider.addEventListener("input", (e) => {
    const val = e.target.value;
    sliderValueBadge.textContent = val;
    metricLimit.textContent = val;
  });

  // Toggle Inspector Sidebar
  toggleInspectorBtn.addEventListener("click", () => {
    inspectorSidebar.classList.toggle("hidden");
  });
  closeInspectorBtn.addEventListener("click", () => {
    inspectorSidebar.classList.add("hidden");
  });

  // Close Error Banner
  closeErrorBtn.addEventListener("click", () => {
    errorBanner.classList.add("hidden");
  });

  function showError(msg) {
    errorMessage.textContent = msg;
    errorBanner.classList.remove("hidden");
  }

  // Reset chat
  resetChatBtn.addEventListener("click", async () => {
    if (confirm("Reset conversation history? This will clear memory and start fresh.")) {
      try {
        await fetch("/api/reset", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ conversation_id: conversationId }),
        });
      } catch (e) {
        console.warn("Reset call error:", e);
      }
      conversationId = "session-" + Math.random().toString(36).substring(2, 9);
      sessionIdDisplay.textContent = conversationId;
      messagesArea.innerHTML = "";
      if (welcomeCard) messagesArea.appendChild(welcomeCard);

      // Reset metrics
      metricTotal.textContent = "0";
      metricRetained.textContent = "0";
      metricTruncated.textContent = "0";
      summaryCardContent.className = "summary-card-empty";
      summaryCardContent.textContent = "No summary generated yet. When older turns overflow the window in Summarization mode, the LLM compresses them here.";
      errorBanner.classList.add("hidden");
    }
  });

  // Send message implementation
  async function sendMessage() {
    const text = userInput.value.trim();
    if (!text || isGenerating) return;

    if (welcomeCard && welcomeCard.parentNode === messagesArea) {
      welcomeCard.remove();
    }

    // Append user message to DOM
    appendMessage("user", text);
    userInput.value = "";
    userInput.style.height = "auto";
    charCounter.textContent = "0 / 2000";

    // Show typing indicator
    const typingRow = appendTypingIndicator();
    isGenerating = true;
    sendBtn.disabled = true;
    errorBanner.classList.add("hidden");

    // Gather active configuration from UI
    const selectedStrategy = document.querySelector('input[name="strategyOption"]:checked')?.value || "sliding_window";
    const maxMessages = parseInt(windowSlider.value, 10) || 6;

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          conversation_id: conversationId,
          message: text,
          strategy: selectedStrategy,
          max_messages: maxMessages,
        }),
      });

      typingRow.remove();

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        const errTxt = errorData.error || `HTTP ${response.status}: Failed to get response.`;
        showError(errTxt);
        appendMessage("assistant", `⚠️ **Error**: ${errTxt}\n\n*Check your API key in \`.env\` or server console.*`);
        return;
      }

      const data = await response.json();
      appendMessage("assistant", data.reply);

      // Update Inspector Metrics
      if (data.metadata) {
        metricTotal.textContent = data.metadata.total_messages;
        metricRetained.textContent = data.metadata.retained_messages;
        metricTruncated.textContent = data.metadata.truncated_messages;
        metricLimit.textContent = data.metadata.max_allowed;

        if (data.metadata.running_summary) {
          summaryCardContent.className = "summary-card-active";
          summaryCardContent.textContent = data.metadata.running_summary;
        }
      }
    } catch (err) {
      typingRow.remove();
      showError("Connection failed: " + err.message);
      appendMessage("assistant", "⚠️ **Network Error**: Unable to reach backend server. Ensure `python server.py` is running.");
    } finally {
      isGenerating = false;
      sendBtn.disabled = false;
      messagesArea.scrollTop = messagesArea.scrollHeight;
      userInput.focus();
    }
  }

  function appendMessage(role, content) {
    const row = document.createElement("div");
    row.className = `message-row ${role}`;

    const avatar = document.createElement("div");
    avatar.className = "message-avatar";
    avatar.textContent = role === "user" ? "👤" : "⚡";

    const bubble = document.createElement("div");
    bubble.className = "message-bubble";
    bubble.innerHTML = renderMarkdown(content);

    row.appendChild(avatar);
    row.appendChild(bubble);
    messagesArea.appendChild(row);
    messagesArea.scrollTop = messagesArea.scrollHeight;
    return row;
  }

  function appendTypingIndicator() {
    const row = document.createElement("div");
    row.className = "message-row assistant";

    const avatar = document.createElement("div");
    avatar.className = "message-avatar";
    avatar.textContent = "⚡";

    const bubble = document.createElement("div");
    bubble.className = "message-bubble";
    bubble.innerHTML = `
      <div class="typing-dots">
        <span class="typing-dot"></span>
        <span class="typing-dot"></span>
        <span class="typing-dot"></span>
      </div>
    `;

    row.appendChild(avatar);
    row.appendChild(bubble);
    messagesArea.appendChild(row);
    messagesArea.scrollTop = messagesArea.scrollHeight;
    return row;
  }

  // Lightweight safe markdown renderer
  function renderMarkdown(raw) {
    if (!raw) return "";

    // Escape basic HTML
    let escaped = raw
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");

    // Code blocks: ```lang ... ```
    escaped = escaped.replace(/```([\s\S]*?)```/g, (match, code) => {
      return `<pre><code>${code.trim()}</code></pre>`;
    });

    // Inline code: `code`
    escaped = escaped.replace(/`([^`]+)`/g, "<code>$1</code>");

    // Bold: **text**
    escaped = escaped.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");

    // Italic: *text*
    escaped = escaped.replace(/\*([^*]+)\*/g, "<em>$1</em>");

    // Line breaks to <br> or paragraphs
    escaped = escaped.replace(/\n\n+/g, "</p><p>").replace(/\n/g, "<br>");

    return `<p>${escaped}</p>`;
  }
});
