const form = document.querySelector("#question-form");
const input = document.querySelector("#question");
const sendButton = document.querySelector("#send-button");
const messages = document.querySelector("#messages");

function escapeHtml(value) {
  const element = document.createElement("div");
  element.textContent = value;
  return element.innerHTML;
}

function addUserMessage(question) {
  messages.insertAdjacentHTML("beforeend", `
    <article class="message user-message">
      <div class="bubble"><p>${escapeHtml(question)}</p></div>
    </article>`);
}

function addLoadingMessage() {
  const id = `loading-${Date.now()}`;
  messages.insertAdjacentHTML("beforeend", `
    <article id="${id}" class="message assistant-message">
      <div class="avatar" aria-hidden="true">IF</div>
      <div class="bubble"><div class="typing" aria-label="Preparing answer"><span></span><span></span><span></span></div></div>
    </article>`);
  return id;
}

function answerMarkup(data) {
  const sources = data.sources.length
    ? `<div class="sources"><strong>Sources</strong><div class="source-list">${data.sources.map(source => `<span class="source-chip">${escapeHtml(source)}</span>`).join("")}</div></div>`
    : "";
  const score = data.refused || data.grounding_score === null
    ? ""
    : ` · Grounding ${(data.grounding_score * 100).toFixed(0)}%`;
  return `
    <article class="message assistant-message ${data.refused ? "refusal" : ""}">
      <div class="avatar" aria-hidden="true">IF</div>
      <div class="bubble">
        <p>${escapeHtml(data.answer)}</p>
        ${sources}
        <div class="meta">${data.latency_seconds.toFixed(2)}s${score}</div>
      </div>
    </article>`;
}

function scrollToLatest() {
  messages.scrollTo({ top: messages.scrollHeight, behavior: "smooth" });
}

async function submitQuestion(question) {
  addUserMessage(question);
  const loadingId = addLoadingMessage();
  input.value = "";
  input.style.height = "auto";
  sendButton.disabled = true;
  scrollToLatest();

  try {
    const response = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    if (!response.ok) throw new Error("The support service could not process that question.");
    const data = await response.json();
    document.querySelector(`#${loadingId}`).outerHTML = answerMarkup(data);
  } catch (error) {
    document.querySelector(`#${loadingId}`).outerHTML = `
      <article class="message assistant-message refusal">
        <div class="avatar" aria-hidden="true">IF</div>
        <div class="bubble"><p>${escapeHtml(error.message)} Please try again.</p></div>
      </article>`;
  } finally {
    sendButton.disabled = false;
    input.focus();
    scrollToLatest();
  }
}

form.addEventListener("submit", event => {
  event.preventDefault();
  const question = input.value.trim();
  if (question) submitQuestion(question);
});

input.addEventListener("input", () => {
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 150)}px`;
});

input.addEventListener("keydown", event => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});

document.querySelectorAll(".suggestion").forEach(button => {
  button.addEventListener("click", () => submitQuestion(button.textContent.trim()));
});
