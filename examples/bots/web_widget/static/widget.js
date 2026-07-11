/*
 * cognee web chat widget — embeddable snippet.
 *
 * Drop one tag on any page:
 *   <script src="https://your-host/widget.js"
 *           data-site-id="acme" data-api="https://your-host"></script>
 *
 * It renders a floating chat bubble, talks to the cognee-backed backend,
 * shows inline citations, and supports a /forget command + opt-out. No
 * build step, no framework.
 */
(function () {
  var script = document.currentScript;
  var API = (script && script.getAttribute("data-api")) || window.location.origin;
  var SITE_ID = (script && script.getAttribute("data-site-id")) || "demo";

  // Stable per-browser ids so a returning visitor keeps their conversation.
  function id(key, prefix) {
    var v = localStorage.getItem(key);
    if (!v) {
      v = prefix + "-" + Math.random().toString(36).slice(2, 10);
      localStorage.setItem(key, v);
    }
    return v;
  }
  var visitorId = id("cognee_visitor_id", "visitor");
  var conversationId = id("cognee_conversation_id", "conv");
  var optIn = localStorage.getItem("cognee_opt_in") !== "0";

  var css =
    ".cognee-w{position:fixed;bottom:20px;right:20px;width:360px;max-width:92vw;font:14px/1.5 system-ui,sans-serif;z-index:2147483000}" +
    ".cognee-box{display:none;flex-direction:column;background:#fff;border:1px solid #e5e7eb;border-radius:12px;box-shadow:0 12px 32px rgba(0,0,0,.18);overflow:hidden}" +
    ".cognee-box.open{display:flex}" +
    ".cognee-head{background:#111827;color:#fff;padding:10px 14px;display:flex;justify-content:space-between;align-items:center}" +
    ".cognee-head b{font-weight:600}" +
    ".cognee-log{padding:12px;height:340px;overflow-y:auto;background:#f9fafb}" +
    ".cognee-msg{margin:6px 0;padding:8px 10px;border-radius:10px;max-width:85%;white-space:pre-wrap;word-wrap:break-word}" +
    ".cognee-user{background:#2563eb;color:#fff;margin-left:auto}" +
    ".cognee-bot{background:#fff;border:1px solid #e5e7eb}" +
    ".cognee-cites{margin:4px 0 10px;font-size:12px;color:#6b7280}" +
    ".cognee-cite{border-left:3px solid #d1d5db;padding:2px 8px;margin:3px 0}" +
    ".cognee-in{display:flex;border-top:1px solid #e5e7eb}" +
    ".cognee-in input{flex:1;border:0;padding:11px;outline:none}" +
    ".cognee-in button{border:0;background:#2563eb;color:#fff;padding:0 16px;cursor:pointer}" +
    ".cognee-bar{padding:6px 12px;font-size:12px;color:#6b7280;display:flex;justify-content:space-between;background:#fff;border-top:1px solid #f3f4f6}" +
    ".cognee-bar a{color:#2563eb;cursor:pointer;text-decoration:none}" +
    ".cognee-launch{border:0;background:#111827;color:#fff;border-radius:24px;padding:12px 18px;cursor:pointer;box-shadow:0 8px 24px rgba(0,0,0,.2)}";
  var style = document.createElement("style");
  style.textContent = css;
  document.head.appendChild(style);

  var root = document.createElement("div");
  root.className = "cognee-w";
  root.innerHTML =
    '<div class="cognee-box" id="cognee-box">' +
    '  <div class="cognee-head"><b>Ask our docs</b>' +
    '    <span style="cursor:pointer" id="cognee-close">×</span></div>' +
    '  <div class="cognee-log" id="cognee-log"></div>' +
    '  <div class="cognee-bar">' +
    '    <label><input type="checkbox" id="cognee-optin"> Remember this chat</label>' +
    '    <a id="cognee-forget">Forget me</a></div>' +
    '  <div class="cognee-in">' +
    '    <input id="cognee-input" placeholder="Ask a question…" autocomplete="off"/>' +
    '    <button id="cognee-send">Send</button></div>' +
    "</div>" +
    '<button class="cognee-launch" id="cognee-launch">💬 Chat</button>';
  document.body.appendChild(root);

  var box = root.querySelector("#cognee-box");
  var log = root.querySelector("#cognee-log");
  var input = root.querySelector("#cognee-input");
  var optinBox = root.querySelector("#cognee-optin");
  optinBox.checked = optIn;

  function open(v) {
    box.classList.toggle("open", v);
  }
  root.querySelector("#cognee-launch").onclick = function () {
    open(true);
    input.focus();
  };
  root.querySelector("#cognee-close").onclick = function () {
    open(false);
  };
  optinBox.onchange = function () {
    optIn = optinBox.checked;
    localStorage.setItem("cognee_opt_in", optIn ? "1" : "0");
  };

  function el(cls, text) {
    var d = document.createElement("div");
    d.className = cls;
    d.textContent = text;
    return d;
  }
  function addMsg(role, text) {
    log.appendChild(el("cognee-msg cognee-" + role, text));
    log.scrollTop = log.scrollHeight;
  }
  function addCitations(cites) {
    if (!cites || !cites.length) return;
    var wrap = document.createElement("div");
    wrap.className = "cognee-cites";
    wrap.appendChild(el("", "Sources:"));
    cites.slice(0, 4).forEach(function (c) {
      var line = c.snippet + (c.document ? "  (" + c.document + ")" : "");
      wrap.appendChild(el("cognee-cite", line));
    });
    log.appendChild(wrap);
    log.scrollTop = log.scrollHeight;
  }

  async function send() {
    var text = input.value.trim();
    if (!text) return;
    input.value = "";
    addMsg("user", text);
    try {
      var res = await fetch(API + "/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: text,
          conversation_id: conversationId,
          visitor_id: visitorId,
          site_id: SITE_ID,
          opt_in: optIn,
        }),
      });
      var data = await res.json();
      addMsg("bot", data.answer || "…");
      addCitations(data.citations);
    } catch (e) {
      addMsg("bot", "Sorry — I couldn't reach memory right now.");
    }
  }
  root.querySelector("#cognee-send").onclick = send;
  input.addEventListener("keydown", function (e) {
    if (e.key === "Enter") send();
  });

  root.querySelector("#cognee-forget").onclick = async function () {
    await fetch(API + "/api/forget", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        conversation_id: conversationId,
        visitor_id: visitorId,
        site_id: SITE_ID,
      }),
    });
    log.innerHTML = "";
    addMsg("bot", "Done — I've forgotten this conversation.");
  };
})();
