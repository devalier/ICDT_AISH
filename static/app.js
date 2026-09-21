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
