(() => {
  const form = document.getElementById("chat-form");
  const input = document.getElementById("message");
  const send = document.getElementById("send");
  const log = document.getElementById("log");
  const statusEl = document.getElementById("status");

  let timeoutMs = null;
  let inFlight = false;

  function setStatus(text, isError) {
    if (!text) {
      statusEl.hidden = true;
      statusEl.textContent = "";
      statusEl.classList.remove("error");
      return;
    }
    statusEl.hidden = false;
    statusEl.textContent = text;
    statusEl.classList.toggle("error", Boolean(isError));
  }

  function appendBubble(role, text) {
    const el = document.createElement("div");
    el.className = `bubble ${role}`;
    el.textContent = text;
    log.appendChild(el);
    log.scrollTop = log.scrollHeight;
  }

  function setComposerEnabled(enabled) {
    input.disabled = !enabled;
    send.disabled = !enabled;
  }

  async function loadHealth() {
    setComposerEnabled(false);
    let response;
    try {
      response = await fetch("/health");
    } catch (err) {
      setStatus(`无法读取服务配置: ${err}`, true);
      return;
    }
    if (!response.ok) {
      setStatus(`health 失败: HTTP ${response.status}`, true);
      return;
    }
    let data;
    try {
      data = await response.json();
    } catch (err) {
      setStatus(`health 响应不是 JSON: ${err}`, true);
      return;
    }
    const timeoutSeconds = Number(data.timeout_seconds);
    if (!(timeoutSeconds > 0)) {
      setStatus("health 缺少有效 timeout_seconds", true);
      return;
    }
    timeoutMs = timeoutSeconds * 1000;
    setComposerEnabled(true);
    if (data.status === "degraded" || data.mcp_ready === false) {
      setStatus("MCP 将在下次提问时重建");
    } else {
      setStatus("");
    }
  }

  async function onSubmit(event) {
    event.preventDefault();
    if (inFlight || timeoutMs == null) {
      return;
    }
    const currentInput = input.value.trim();
    if (!currentInput) {
      return;
    }

    inFlight = true;
    setComposerEnabled(false);
    send.textContent = "生成中…（可能需要 1–2 分钟）";
    appendBubble("user", currentInput);
    input.value = "";

    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: currentInput }),
        signal: controller.signal,
      });
      let payload = null;
      try {
        payload = await response.json();
      } catch (err) {
        throw new Error(`响应不是 JSON (HTTP ${response.status}): ${err}`);
      }
      if (!response.ok) {
        const message =
          payload && payload.error && payload.error.message
            ? payload.error.message
            : `HTTP ${response.status}`;
        throw new Error(message);
      }
      if (!payload || typeof payload.answer !== "string" || !payload.answer.trim()) {
        throw new Error("服务返回空答案");
      }
      appendBubble("assistant", payload.answer);
    } catch (err) {
      const text =
        err && err.name === "AbortError"
          ? "请求超时"
          : (err && err.message) || String(err);
      appendBubble("error", text);
    } finally {
      clearTimeout(timer);
      inFlight = false;
      send.textContent = "发送";
      setComposerEnabled(true);
      input.focus();
    }
  }

  form.addEventListener("submit", onSubmit);
  loadHealth();
})();
