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

/* Model discovery on the models form: the user's own key is sent once to list the models it
   can use; the list fills the model field's suggestions, each labelled with what the provider
   publishes (name, context, price). The provider select also shows that provider's notes and
   hides the endpoint field for providers that only use their own fixed API. */
(function () {
  "use strict";
  var button = document.querySelector("[data-load-models]");
  if (!button) return;

  var form = button.closest("form");
  var status = document.querySelector("[data-load-status]");
  var list = document.getElementById("model_options");
  var select = form.querySelector("[data-provider-select]");
  var hint = form.querySelector("[data-provider-hint]");
  var endpointField = form.querySelector("[data-endpoint-field]");

  function onProvider() {
    if (!select) return;
    var opt = select.options[select.selectedIndex];
    if (hint) hint.textContent = opt.getAttribute("data-hint") || "";
    if (endpointField) endpointField.hidden = opt.getAttribute("data-endpoint") === "none";
    list.textContent = "";
    status.textContent = opt.getAttribute("data-catalog") === "public"
      ? "You can load this provider's model list before entering a key."
      : "Paste your API key, then load the models it can use.";
  }
  if (select) { select.addEventListener("change", onProvider); onProvider(); }

  button.addEventListener("click", function () {
    var data = new FormData();
    ["csrf_token", "provider", "endpoint", "api_key"].forEach(function (name) {
      data.append(name, form.elements[name].value);
    });
    status.textContent = "Loading…";
    fetch("/models/available", { method: "POST", body: data, credentials: "same-origin" })
      .then(function (r) {
        return r.json()
          .catch(function () { return { error: "Could not load models: the server answered " + r.status + " " + r.statusText + ". Reload the page and try again." }; })
          .then(function (body) { return { ok: r.ok, body: body }; });
      })
      .then(function (res) {
        if (!res.ok) { status.textContent = res.body.error || "Could not load models."; return; }
        var labels = res.body.labels || {};
        list.textContent = "";
        res.body.models.forEach(function (id) {
          var option = document.createElement("option");
          option.value = id;
          if (labels[id]) option.label = labels[id];   // text only: never parsed as HTML
          list.appendChild(option);
        });
        status.textContent = res.body.models.length +
          " models available. Click the model field and type to search, or type any id the provider accepts." +
          (res.body.notice ? " " + res.body.notice : "");
      })
      .catch(function () { status.textContent = "Could not load models: the server could not be reached."; });
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


/* Show a form field only while another field has a given value, e.g. data-show-when="scoring_mode=judge".
   Without JavaScript every field is visible, so the form still works. */
(function () {
  "use strict";
  function apply(form) {
    form.querySelectorAll("[data-show-when]").forEach(function (el) {
      var parts = el.getAttribute("data-show-when").split("=");
      var control = form.querySelector('[name="f_' + parts[0] + '"]');
      if (!control) return;
      el.hidden = parts[1].split("|").indexOf(control.value) === -1;
    });
  }
  document.querySelectorAll("[data-mode-form]").forEach(function (form) {
    apply(form);
    form.addEventListener("change", function () { apply(form); });
  });
})();


/* A switch that saves itself. Turning it on first shows its warning (data-confirm-on). */
(function () {
  "use strict";
  document.querySelectorAll("form[data-autosubmit] input[type=checkbox]").forEach(function (box) {
    box.addEventListener("change", function () {
      var warn = box.getAttribute("data-confirm-on");
      if (box.checked && warn && !window.confirm(warn)) { box.checked = false; return; }
      box.form.submit();
    });
  });
})();


/* Run form: when a restricted (Safety) suite is selected, show its guidance and list the public
   cloud models the user has ticked, so the outcome is clear before pressing Run. */
(function () {
  "use strict";
  var select = document.querySelector("[data-suite-select]");
  var panel = document.querySelector("[data-safety-panel]");
  if (!select || !panel) return;
  var form = select.form;
  var list = panel.querySelector("[data-safety-cloud-list]");
  var confirmBox = panel.querySelector("[data-safety-confirm]");
  var allowed = !!confirmBox;

  function update() {
    var restricted = select.options[select.selectedIndex].getAttribute("data-restricted") === "yes";
    panel.hidden = !restricted;
    var cloud = Array.prototype.slice.call(form.querySelectorAll("input[name=target_ids][data-cloud=yes]"))
      .filter(function (b) { return b.checked; })
      .map(function (b) { return b.getAttribute("data-label"); });
    if (list) {
      list.hidden = cloud.length === 0 && !allowed;
      list.textContent = cloud.length === 0
        ? (allowed ? "No public cloud models selected: nothing extra to confirm." : "")
        : (allowed ? "Public cloud models selected: " + cloud.join(", ") + "."
                   : "These selected models are public cloud models and will be refused: " + cloud.join(", ") + ".");
    }
    if (confirmBox) {
      var box = confirmBox.querySelector("input[name=confirm_cloud_safety]");
      box.parentNode.hidden = cloud.length === 0;
      box.required = restricted && cloud.length > 0;
    }
  }
  select.addEventListener("change", update);
  form.addEventListener("change", update);
  update();
})();
