/* Progressive enhancement: with JS, [data-tabs] groups show one pane at a time; without
   it every pane is visible. No inline script: the CSP is script-src 'self'. */
(function () {
  "use strict";
  document.documentElement.classList.add("js");
  var groups = document.querySelectorAll("[data-tabs]");
  groups.forEach(function (root) {
    var tabs = Array.prototype.slice.call(root.querySelectorAll('[role="tab"]'));
    var panes = {};
    tabs.forEach(function (t) { panes[t.getAttribute("aria-controls")] = document.getElementById(t.getAttribute("aria-controls")); });

    function show(id) {
      tabs.forEach(function (t) {
        var on = t.getAttribute("aria-controls") === id;
        t.setAttribute("aria-selected", on ? "true" : "false");
        t.tabIndex = on ? 0 : -1;
        panes[t.getAttribute("aria-controls")].hidden = !on;
      });
    }
    function fromHash() {
      var id = window.location.hash.slice(1);
      if (id && panes[id]) { show(id); return true; }
      return false;
    }
    tabs.forEach(function (t, i) {
      t.addEventListener("click", function () {
        var id = t.getAttribute("aria-controls");
        show(id);
        if (window.history && window.history.replaceState) window.history.replaceState(null, "", "#" + id);
      });
      t.addEventListener("keydown", function (e) {
        var next = e.key === "ArrowRight" ? i + 1 : e.key === "ArrowLeft" ? i - 1 : null;
        if (next === null) return;
        var target = tabs[(next + tabs.length) % tabs.length];
        target.focus(); target.click(); e.preventDefault();
      });
    });
    if (!fromHash()) {
      var initial = tabs.filter(function (t) { return t.getAttribute("aria-selected") === "true"; })[0] || tabs[0];
      show(initial.getAttribute("aria-controls"));
    }
    window.addEventListener("hashchange", fromHash);
  });
})();

/* Progress polling for an in-flight run. No inline script: the CSP is
   script-src 'self'. */
(function () {
  "use strict";
  var root = document.querySelector("[data-run-poll]");
  if (!root) return;

  var url = root.getAttribute("data-run-poll");
  var bar = document.querySelector("[data-progress-bar]");
  var text = document.querySelector("[data-progress-text]");
  var tries = 0;

  function tick() {
    if (tries > 600) return;               // stop after ~20 minutes
    tries += 1;
    fetch(url, { credentials: "same-origin", headers: { Accept: "application/json" } })
      .then(function (r) { return r.ok ? r.json() : null; })
      .then(function (data) {
        if (!data) return;
        var done = data.items_done || 0;
        var total = data.items_total || 0;
        var pct = total ? Math.round((done / total) * 100) : 0;
        if (bar) bar.setAttribute("width", pct + "%");
        if (text) text.textContent = done + " of " + total + " items — " + data.status;
        if (data.status === "done" || data.status === "failed") {
          window.location.reload();
          return;
        }
        window.setTimeout(tick, 2000);
      })
      .catch(function () { window.setTimeout(tick, 5000); });
  }
  window.setTimeout(tick, 1500);
})();

/* Model discovery on the "My models" form. */
(function () {
  "use strict";
  var button = document.querySelector("[data-load-models]");
  if (!button) return;

  var form = button.closest("form");
  var status = document.querySelector("[data-load-status]");
  var list = document.getElementById("model_options");

  button.addEventListener("click", function () {
    var data = new FormData();
    ["csrf_token", "provider", "endpoint", "api_key"].forEach(function (name) {
      data.append(name, form.elements[name].value);
    });
    status.textContent = "Loading…";
    fetch("/models/available", { method: "POST", body: data, credentials: "same-origin" })
      .then(function (r) { return r.json().then(function (body) { return { ok: r.ok, body: body }; }); })
      .then(function (res) {
        if (!res.ok) { status.textContent = res.body.error || "Could not load models."; return; }
        list.textContent = "";
        res.body.models.forEach(function (id) {
          var option = document.createElement("option");
          option.value = id;
          list.appendChild(option);
        });
        status.textContent = res.body.models.length + " models available — click the field to pick one.";
      })
      .catch(function () { status.textContent = "Could not load models."; });
  });
})();


/* Model selection: at most N ticked at once (data-max-select), a live status line, and the
   action button enabled only for a valid selection. The server enforces the same limit. */
(function () {
  "use strict";
  var boxes = Array.prototype.slice.call(document.querySelectorAll("input[data-max-select]"));
  if (!boxes.length) return;
  var max = parseInt(boxes[0].getAttribute("data-max-select"), 10) || 3;
  var status = document.querySelector("[data-select-status]");
  var button = document.querySelector("[data-run-selected]");

  function update() {
    var ticked = boxes.filter(function (b) { return b.checked; }).length;
    boxes.forEach(function (b) { b.disabled = !b.checked && ticked >= max; });
    if (button) button.disabled = ticked < 1;
    if (status) {
      status.textContent = ticked === 0 ? "Select 1 to " + max + " models."
        : ticked + " of " + max + " selected" + (ticked >= max ? " (the maximum)." : ".");
    }
  }
  boxes.forEach(function (b) { b.addEventListener("change", update); });
  update();
})();

/* Confirm before a destructive action (data-confirm on a submit button). */
document.addEventListener("click", function (e) {
  var el = e.target.closest && e.target.closest("[data-confirm]");
  if (el && !window.confirm(el.getAttribute("data-confirm"))) e.preventDefault();
});

/* Reload a page section that shows runs in progress. */
(function () {
  "use strict";
  var el = document.querySelector("[data-autorefresh]");
  if (!el) return;
  window.setTimeout(function () { window.location.reload(); },
                    parseInt(el.getAttribute("data-autorefresh"), 10) || 5000);
})();


/* Starter examples: insert the example text into the editor, just before the section that follows
   the list it belongs in (data-insert-before is a regular expression for that section's key). */
document.addEventListener("click", function (e) {
  var btn = e.target.closest && e.target.closest("[data-insert-from]");
  if (!btn) return;
  var editor = document.querySelector("[data-yaml-editor]");
  var source = document.getElementById(btn.getAttribute("data-insert-from"));
  if (!editor || !source) return;
  var snippet = source.textContent.replace(/\s+$/, "") + "\n\n";
  var re = new RegExp(btn.getAttribute("data-insert-before"), "m");
  var m = re.exec(editor.value);
  var at = m ? m.index : editor.value.length;
  editor.value = editor.value.slice(0, at) + snippet + editor.value.slice(at);
  editor.focus();
  editor.setSelectionRange(at, at + snippet.length);
  var line = editor.value.slice(0, at).split("\n").length;
  editor.scrollTop = Math.max(0, (line - 4) * 18);
});
