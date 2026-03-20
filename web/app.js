const API_BASE = "http://127.0.0.1:8000";

const queryEl = document.getElementById("query");
const modeEl = document.getElementById("mode");
const showContextsEl = document.getElementById("showContexts");
const askBtn = document.getElementById("askBtn");
const answerEl = document.getElementById("answer");
const contextsEl = document.getElementById("contexts");

async function askAssistant() {
  const query = queryEl.value.trim();
  if (!query) {
    answerEl.textContent = "Please enter a question.";
    return;
  }

  askBtn.disabled = true;
  askBtn.textContent = "Loading...";
  answerEl.textContent = "Loading...";
  contextsEl.textContent = "Loading...";

  try {
    const response = await fetch(`${API_BASE}/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query,
        mode: modeEl.value,
        show_contexts: showContextsEl.checked,
      }),
    });

    const data = await response.json();

    if (!response.ok || data.error) {
      answerEl.textContent = `Error: ${data.error || response.statusText}`;
      contextsEl.textContent = "";
      return;
    }

    answerEl.textContent = data.answer || "";
    contextsEl.textContent = (data.contexts || []).join("\n\n---\n\n");
  } catch (error) {
    answerEl.textContent = `Request failed: ${error}`;
    contextsEl.textContent = "";
  } finally {
    askBtn.disabled = false;
    askBtn.textContent = "Ask";
  }
}

askBtn.addEventListener("click", askAssistant);
