const chatWindow = document.getElementById("chatWindow");
const input = document.getElementById("chatInput");
const sendBtn = document.getElementById("sendBtn");

function addMessage(text, who, sources) {
  const div = document.createElement("div");
  div.className = `msg ${who}`;
  div.textContent = text;
  if (sources && sources.length) {
    const s = document.createElement("span");
    s.className = "sources";
    s.innerHTML = "Manbalar: " + sources.map(
      src => `<a href="${src.url}" target="_blank">${src.title}</a>`
    ).join(" | ");
    div.appendChild(s);
  }
  chatWindow.appendChild(div);
  chatWindow.scrollTop = chatWindow.scrollHeight;
}

async function send() {
  const text = input.value.trim();
  if (!text) return;
  addMessage(text, "user");
  input.value = "";
  sendBtn.disabled = true;

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text }),
    });
    const data = await res.json();
    addMessage(data.answer, "bot", data.sources);
  } catch (e) {
    addMessage("Xatolik yuz berdi: " + e.message, "bot");
  } finally {
    sendBtn.disabled = false;
  }
}

sendBtn.addEventListener("click", send);
input.addEventListener("keydown", (e) => {
  if (e.key === "Enter") send();
});
