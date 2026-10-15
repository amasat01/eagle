/* The capability chart of the site (.capmap).
 *
 * It is drawn on the server with every tool's list and the comparison table in the page; this script only switches tools: the
 * chip row is a single-select picker (arrow keys move it; tapping the chosen chip again picks none), the icons slide up or down
 * to the band of the chosen tool (CSS transition, none under reduced motion), and the chosen tool's sentence and list are shown.
 * An icon opens a small details card on tap, click or focus. Cross-highlight: anything carrying data-feature (a radar axis or
 * a table row) lights every other element of the same feature on the page while it is hovered or focused.
 * Nothing is hover-only: every fact is in the page, and a tap does what hover does. Vanilla JS. */
(function () {
  "use strict";

  /* ---- cross-highlight: delegated, so the radar's own elements take part without knowing about this file ---- */
  var xhKey = null;
  function setXh(key) {
    if (key === xhKey) { return; }
    var old = document.querySelectorAll(".xh");
    for (var i = 0; i < old.length; i++) { old[i].classList.remove("xh"); }
    xhKey = key;
    if (!key) { return; }
    var sel = '[data-feature="' + key + '"], [data-axis="' + key + '"]';
    var hit = document.querySelectorAll(sel);
    for (var j = 0; j < hit.length; j++) { hit[j].classList.add("xh"); }
  }
  function featureOf(target) {
    if (!target.closest) { return null; }
    var n = target.closest("[data-feature], .radar-axis[data-axis], .radar-icon[data-axis]");
    return n ? (n.getAttribute("data-feature") || n.getAttribute("data-axis")) : null;
  }
  document.addEventListener("mouseover", function (ev) { setXh(featureOf(ev.target)); });
  document.addEventListener("mouseout", function (ev) { if (!ev.relatedTarget || !featureOf(ev.relatedTarget)) { setXh(null); } });
  document.addEventListener("focusin", function (ev) { setXh(featureOf(ev.target)); });
  document.addEventListener("focusout", function () { setXh(null); });

  /* ---- the capability chart: one tool at a time; tapping the chosen chip again picks none ---- */
  function mountCapmap(root) {
    var chips = Array.prototype.slice.call(root.querySelectorAll(".cm-chip"));
    var icons = Array.prototype.slice.call(root.querySelectorAll(".cq-ic"));
    var panes = root.querySelectorAll(".cm-tool");
    var summary = root.querySelector(".cm-summary");
    var data = JSON.parse(root.querySelector(".cm-json").textContent);
    /* move every icon to the chosen tool's place (vertical only: x is fixed) and set its look: cov, ghost or neutral */
    function place(tool, delay) {
      icons.forEach(function (g, i) {
        var mode = g.parentNode.getAttribute("data-mode") || g.ownerSVGElement.getAttribute("data-mode");
        var p = data.modes[mode].pos[tool || "none"][g.getAttribute("data-block")];
        g.style.transitionDelay = delay ? ((i % 18) * 22) + "ms" : "";
        g.style.transform = "translate(" + p[0] + "px," + p[1] + "px)";
        g.setAttribute("class", "cq-ic is-" + p[2]);
      });
    }
    /* ---- the details card of one capability: opens on tap, click or focus; one at a time ---- */
    var card = root.querySelector(".cq-card");
    var chartBox = root.querySelector(".cm-chart");
    var openIc = null;
    function el(tag, cls, text) { var e = document.createElement(tag); if (cls) { e.className = cls; } if (text !== undefined) { e.textContent = text; } return e; }
    function tick(c, name) {
      if (c.c) { var a = el("a", "cm-tick", name + " \u2713"); a.href = c.s; return a; }
      return el("span", "cm-tick is-no", name + " \u2013");
    }
    function fill(block) {
      var d = data.cards[block], tool = root.getAttribute("data-tool");
      card.textContent = "";
      card.appendChild(el("div", "cc-h", d.l));
      card.appendChild(el("div", "cc-s", d.needs));
      function line(t) {
        var row = el("div", "cc-t");
        row.appendChild(el("span", "cc-tool", t));
        row.appendChild(tick(d.t[t].cpu, "CPU"));
        row.appendChild(tick(d.t[t].gpu, "GPU"));
        return row;
      }
      if (tool) {
        card.appendChild(line(tool));
        var notes = [];
        ["cpu", "gpu"].forEach(function (k) { if (notes.indexOf(d.t[tool][k].n) < 0) { notes.push(d.t[tool][k].n); } });
        card.appendChild(el("div", "cc-s", notes.join(" ")));
        if (!d.t[tool].cpu.c && !d.t[tool].gpu.c) { card.appendChild(el("div", "cc-s", "Not first-class for " + tool + ".")); }
      } else {
        data.tools.forEach(function (t) { if (d.t[t].cpu.c || d.t[t].gpu.c) { card.appendChild(line(t)); } });
      }
    }
    function closeCard() {
      if (openIc) { openIc.classList.remove("is-open"); openIc.setAttribute("aria-expanded", "false"); openIc = null; }
      if (card) { card.hidden = true; }
    }
    function openCard(g) {
      if (!card || openIc === g) { return; }
      closeCard();
      openIc = g;
      g.classList.add("is-open");
      g.setAttribute("aria-expanded", "true");
      fill(g.getAttribute("data-block"));
      card.hidden = false;
      var cr = chartBox.getBoundingClientRect(), ir = g.getBoundingClientRect();
      var w = card.offsetWidth, h = card.offsetHeight;
      var cx = ir.left + ir.width / 2 - cr.left;
      var left = Math.max(0, Math.min(cr.width - w, cx - w / 2));
      var below = ir.bottom - cr.top + 8;
      var top = (ir.top - cr.top > h + 16 && ir.top - cr.top > cr.height * 0.55) ? ir.top - cr.top - h - 8 : below;
      card.style.left = left + "px";
      card.style.top = top + "px";
    }
    icons.forEach(function (g) {
      g.addEventListener("click", function (ev) { ev.stopPropagation(); openCard(g); });
      g.addEventListener("focus", function () { openCard(g); });
      g.addEventListener("keydown", function (ev) {
        if (ev.key === "Enter" || ev.key === " ") { ev.preventDefault(); openCard(g); }
        if (ev.key === "Escape") { closeCard(); }
      });
    });
    root.addEventListener("keydown", function (ev) { if (ev.key === "Escape") { closeCard(); } });
    document.addEventListener("pointerdown", function (ev) {
      if (!ev.target.closest || !ev.target.closest(".cq-ic, .cq-card")) { closeCard(); }
    });
    function choose(tool, focus) {
      closeCard();
      root.setAttribute("data-tool", tool);
      root.classList.toggle("is-neutral", !tool);
      chips.forEach(function (c) {
        var on = c.getAttribute("data-tool") === tool;
        c.classList.toggle("is-on", on);
        c.setAttribute("aria-pressed", on ? "true" : "false");
        if (on) { root.style.setProperty("--c", c.style.getPropertyValue("--c")); }
        if (on && focus) { c.focus(); }
      });
      place(tool, false);
      for (var j = 0; j < panes.length; j++) { panes[j].hidden = panes[j].getAttribute("data-tool") !== tool; }
      if (summary) { summary.textContent = tool ? data.summaries[tool] : data.none; }
    }
    chips.forEach(function (c, idx) {
      c.addEventListener("click", function () {
        var t = c.getAttribute("data-tool");
        choose(root.getAttribute("data-tool") === t ? "" : t, false);
      });
      c.addEventListener("keydown", function (ev) {
        var d = ev.key === "ArrowRight" || ev.key === "ArrowDown" ? 1 : ev.key === "ArrowLeft" || ev.key === "ArrowUp" ? -1 : 0;
        if (!d) { return; }
        ev.preventDefault();
        choose(chips[(idx + d + chips.length) % chips.length].getAttribute("data-tool"), true);
      });
    });
    root.classList.add("cm-js");
    /* entrance: the first time the chart is seen, every icon starts on the middle band (no tool) and slides to the chosen tool, once */
    var reduce = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    var chart = root.querySelector(".cm-chart");
    if (reduce || !chart || !("IntersectionObserver" in window)) { return; }
    var io = new IntersectionObserver(function (es) {
      if (!es.some(function (e) { return e.isIntersecting; })) { return; }
      io.disconnect();
      var tool = root.getAttribute("data-tool");
      if (!tool) { return; }
      root.classList.add("is-neutral", "cm-enter", "cm-snap");
      place("", false);
      void root.offsetWidth;
      root.classList.remove("cm-snap", "is-neutral");
      place(tool, true);
      setTimeout(function () {
        root.classList.remove("cm-enter");
        icons.forEach(function (g) { g.style.transitionDelay = ""; });
      }, 1400);
    }, { threshold: 0.35 });
    io.observe(chart);
  }

  function start() {
    var maps = document.querySelectorAll(".capmap");
    for (var m = 0; m < maps.length; m++) { mountCapmap(maps[m]); }
  }
  if (document.readyState === "loading") { document.addEventListener("DOMContentLoaded", start); }
  else { start(); }
})();
