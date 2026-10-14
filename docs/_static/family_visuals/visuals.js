/* The two coverage visuals of the site: the two-worlds chart (.worlds) and the rosette (.rosette), plus the
 * cross-highlight every visual shares.
 *
 * Both are drawn on the server (tools/visuals.py, tools/visuals_markup.py) for every device, with the data tables behind
 * them; this script only shows the device the page's switch selects and handles three interactions:
 *   1. a tool, by its legend chip or by its blob (the same handler): the tool is isolated (the others fade), its outline
 *      thickens and, on the wide chart, its name shows inside its blob. Tap = focus = hover; Escape clears.
 *   2. a block (worlds) or a feature (rosette): an evidence popover with the features it needs, every tool's reading and
 *      who covers it, built from the JSON that mirrors the tables.
 *   3. a rosette sector: focus it (the others fade); on a phone-width column the sector picker shows one sector fanned.
 * Cross-highlight: anything carrying data-feature (a rosette node or label, a popover row, a radar axis, a table row)
 * lights every other element of the same feature on the page while it is hovered or focused.
 * Nothing is hover-only: every fact is in the tables, and a tap does what hover does. Vanilla JS, no library. */
(function () {
  "use strict";

  function mk(tag, cls, text) {
    var e = document.createElement(tag);
    if (cls) { e.className = cls; }
    if (text !== undefined) { e.textContent = text; }
    return e;
  }

  /* ---- cross-highlight: delegated, so the radar's own elements take part without knowing about this file ---- */
  var xhKey = null;
  function setXh(key) {
    if (key === xhKey) { return; }
    var old = document.querySelectorAll(".xh");
    for (var i = 0; i < old.length; i++) { old[i].classList.remove("xh"); }
    xhKey = key;
    if (!key) { return; }
    var hit = document.querySelectorAll('[data-feature="' + key + '"], [data-axis="' + key + '"]');
    for (var j = 0; j < hit.length; j++) { hit[j].classList.add("xh"); }
  }
  function featureOf(target) {
    var n = target.closest ? target.closest("[data-feature], .radar-axis[data-axis], .radar-icon[data-axis]") : null;
    return n ? (n.getAttribute("data-feature") || n.getAttribute("data-axis")) : null;
  }
  document.addEventListener("mouseover", function (ev) { setXh(featureOf(ev.target)); });
  document.addEventListener("mouseout", function (ev) { if (!ev.relatedTarget || !featureOf(ev.relatedTarget)) { setXh(null); } });
  document.addEventListener("focusin", function (ev) { setXh(featureOf(ev.target)); });
  document.addEventListener("focusout", function () { setXh(null); });

  function mount(root) {
    var data = JSON.parse(root.querySelector(".worlds-json").textContent);
    var side = data.side;
    var rosette = root.classList.contains("rosette");
    var stage = root.querySelector(".worlds-stage");
    var pop = root.querySelector(".worlds-pop");
    var status = root.querySelector(".worlds-status");
    var legend = root.querySelectorAll(".worlds-leg");
    var devSlug = null, hl = null, pinned = null;

    /* ---- the device ---- */
    function showDevice(slug) {
      var devs = root.querySelectorAll(".worlds-dev");
      var any = false;
      for (var i = 0; i < devs.length; i++) {
        var on = devs[i].getAttribute("data-slug") === slug;
        devs[i].hidden = !on;
        any = any || on;
      }
      if (!any && devs.length) { devs[0].hidden = false; slug = devs[0].getAttribute("data-slug"); }
      devSlug = slug;
      closePop();
      setHl(null);
    }
    function followSwitch() {
      var r = document.querySelector('.perf-switch input[name="perf-' + side + '"]:checked');
      var slug = r ? r.getAttribute("data-slug") : null;
      if (slug && slug !== devSlug) { showDevice(slug); }
    }
    var sw = document.querySelector(".perf-switch");
    if (sw) { sw.addEventListener("perf-device-change", followSwitch); }
    var first = root.querySelector(".worlds-dev:not([hidden])");
    devSlug = first ? first.getAttribute("data-slug") : null;
    followSwitch();

    /* ---- a tool: chip and blob share one handler ---- */
    function setHl(tool) {
      if (hl === tool) { return; }
      hl = tool;
      root.classList.toggle("has-hl", !!tool);
      var els = root.querySelectorAll("[data-tool]");
      for (var i = 0; i < els.length; i++) { els[i].classList.toggle("is-hl", !!tool && els[i].getAttribute("data-tool") === tool); }
      for (var j = 0; j < legend.length; j++) { legend[j].setAttribute("aria-pressed", tool && legend[j].getAttribute("data-tool") === tool ? "true" : "false"); }
      if (status) { status.textContent = tool ? tool : ""; }
    }
    function bindTool(el) {
      var tool = el.getAttribute("data-tool");
      el.addEventListener("mouseenter", function () { setHl(tool); });
      el.addEventListener("mouseleave", function () { if (pinned !== tool) { setHl(null); } });
      el.addEventListener("focus", function () { setHl(tool); });
      el.addEventListener("blur", function () { if (pinned !== tool) { setHl(null); } });
      el.addEventListener("click", function (ev) {
        ev.stopPropagation();
        pinned = pinned === tool ? null : tool;
        setHl(pinned);
      });
    }
    var tools = root.querySelectorAll(".worlds-leg, path.blob[data-tool]");
    for (var t = 0; t < tools.length; t++) { bindTool(tools[t]); }

    /* ---- the evidence popover of a block or a feature ---- */
    function closePop() { if (pop) { pop.hidden = true; pop.textContent = ""; } }
    function evidence(key, kind) {
      var d = data.devices[devSlug];
      var feats, title, who = null, dropped = [];
      if (kind === "block") {
        var b = d.blocks[key];
        title = b.l; feats = b.f.map(function (f) { return f[0]; }); who = b.w; dropped = b.x || [];
      } else {
        title = null; feats = [key];
      }
      var box = mk("div", "wp");
      var toolNames = Object.keys(d.vals[feats[0]] || {});
      var labelOf = function (f) { var r = root.querySelector('.worlds-features tr[data-feature="' + f + '"] th, .worlds-dev:not([hidden]) tr[data-feature="' + f + '"] th'); return r ? r.textContent : f; };
      if (title) { box.appendChild(mk("div", "wp-h", title)); }
      else { box.appendChild(mk("div", "wp-h", labelOf(key))); }
      if (kind === "block") { box.appendChild(mk("div", "wp-s", "Needs " + feats.map(labelOf).join(", ") + ".")); }
      var tb = mk("table", "wp-t");
      var head = mk("tr");
      head.appendChild(mk("th", "", ""));
      toolNames.forEach(function (n) { head.appendChild(mk("th", "", n)); });
      tb.appendChild(head);
      feats.forEach(function (f) {
        var tr = mk("tr");
        var th = mk("th", "", labelOf(f));
        th.setAttribute("data-feature", f);
        tr.appendChild(th);
        toolNames.forEach(function (n) {
          var v = d.vals[f] && d.vals[f][n], td = mk("td");
          if (!v) { td.className = "miss"; td.textContent = "no reading"; }
          else {
            td.className = v[2] ? "ok" : "no";
            td.appendChild(mk("span", "v", v[0] + (v[2] ? " ✓" : "")));
            if (v[1]) { td.appendChild(mk("span", "r", v[1].replace(/ the best$/, ""))); }
          }
          tr.appendChild(td);
        });
        tb.appendChild(tr);
      });
      box.appendChild(tb);
      if (who) { box.appendChild(mk("div", "wp-s", who.length ? "Covered by " + who.join(", ") + "." : "No tool covers this block on this device.")); }
      if (dropped.length) { box.appendChild(mk("div", "wp-s", "No reading on this device for " + dropped.map(labelOf).join(", ") + ": left out of the block's needs.")); }
      return box;
    }
    function placePop(anchor) {
      pop.hidden = false;
      var narrow = root.clientWidth < 640;
      pop.classList.toggle("inflow", narrow);
      if (narrow) { pop.style.left = ""; pop.style.top = ""; return; }
      var st = stage.getBoundingClientRect(), ar = anchor.getBoundingClientRect();
      var w = pop.offsetWidth, h = pop.offsetHeight;
      var left = Math.max(0, Math.min(st.width - w, ar.left - st.left + ar.width + 10));
      if (left + w > ar.left - st.left && ar.left - st.left > w + 10 && ar.left - st.left + ar.width + 10 + w > st.width) { left = ar.left - st.left - w - 10; }
      var top = Math.max(0, Math.min(st.height - h, ar.top - st.top - 8));
      pop.style.left = left + "px";
      pop.style.top = top + "px";
    }
    function showNode(n) {
      if (!pop) { return; }
      var block = n.getAttribute("data-block"), feature = n.getAttribute("data-feature");
      pop.textContent = "";
      pop.appendChild(evidence(block || feature, block ? "block" : "feature"));
      placePop(n);
    }
    var nodes = root.querySelectorAll(".wnode, .rnode");
    for (var k = 0; k < nodes.length; k++) {
      (function (n) {
        n.addEventListener("mouseenter", function () { showNode(n); });
        n.addEventListener("mouseleave", closePop);
        n.addEventListener("focus", function () { showNode(n); });
        n.addEventListener("blur", closePop);
        n.addEventListener("click", function (ev) { ev.stopPropagation(); showNode(n); });
        n.addEventListener("keydown", function (ev) { if (ev.key === "Enter" || ev.key === " ") { ev.preventDefault(); showNode(n); } });
      })(nodes[k]);
    }
    var keys = root.querySelectorAll(".worlds-key li");
    for (var q = 0; q < keys.length; q++) {
      (function (li) {
        li.tabIndex = 0;
        li.addEventListener("click", function () { if (pop) { pop.textContent = ""; pop.appendChild(evidence(li.getAttribute("data-block"), "block")); placePop(li); } });
      })(keys[q]);
    }

    /* ---- rosette: a sector is focus (wide) or the picker's choice (phone width) ---- */
    function focusSector(sk) {
      root.classList.toggle("has-focus", !!sk);
      var els = root.querySelectorAll("[data-sector]");
      for (var i = 0; i < els.length; i++) { els[i].classList.toggle("is-focus", !!sk && els[i].getAttribute("data-sector") === sk); }
    }
    var sectors = root.querySelectorAll(".rsector");
    for (var s = 0; s < sectors.length; s++) {
      (function (g) {
        var sk = g.getAttribute("data-sector");
        g.addEventListener("mouseenter", function () { focusSector(sk); });
        g.addEventListener("mouseleave", function () { focusSector(null); });
        g.addEventListener("focus", function () { focusSector(sk); });
        g.addEventListener("blur", function () { focusSector(null); });
        g.addEventListener("click", function (ev) { ev.stopPropagation(); focusSector(sk); });
      })(sectors[s]);
    }
    var picks = root.querySelectorAll(".pick");
    function pick(sk) {
      for (var i = 0; i < picks.length; i++) {
        var on = picks[i].getAttribute("data-sector") === sk;
        picks[i].classList.toggle("on", on);
        picks[i].setAttribute("aria-pressed", on ? "true" : "false");
      }
      var panes = root.querySelectorAll(".rosette-sector");
      for (var j = 0; j < panes.length; j++) { panes[j].hidden = panes[j].getAttribute("data-sector") !== sk; }
      closePop();
    }
    for (var p = 0; p < picks.length; p++) {
      (function (b) { b.addEventListener("click", function () { pick(b.getAttribute("data-sector")); }); })(picks[p]);
    }

    root.addEventListener("keydown", function (ev) { if (ev.key === "Escape") { pinned = null; setHl(null); closePop(); focusSector(null); } });
    document.addEventListener("pointerdown", function (ev) {
      if (!ev.target.closest || !ev.target.closest(".worlds-leg, .blob, .wnode, .rnode, .rsector, .worlds-pop, .worlds-key, .pick")) {
        pinned = null; setHl(null); closePop(); focusSector(null);
      }
    });
  }

  /* ---- the Radar | Rosette toggle of the detailed view ---- */
  var VIEW_KEY = "raptor-detail-view";
  function initToggle() {
    var roots = document.querySelectorAll(".perf-detailed");
    for (var r = 0; r < roots.length; r++) {
      (function (root) {
        var opts = root.querySelectorAll(".view-opt");
        if (!opts.length) { return; }
        function show(view) {
          root.setAttribute("data-view", view);
          for (var i = 0; i < opts.length; i++) { opts[i].setAttribute("aria-pressed", opts[i].getAttribute("data-view") === view ? "true" : "false"); }
        }
        var saved = null;
        try { saved = window.localStorage.getItem(VIEW_KEY); } catch (e) { /* private window */ }
        root.classList.add("view-js");
        show(saved === "radar" ? "radar" : "rosette");
        root.addEventListener("click", function (ev) {
          var b = ev.target.closest ? ev.target.closest(".view-opt") : null;
          if (!b) { return; }
          var v = b.getAttribute("data-view") === "radar" ? "radar" : "rosette";
          show(v);
          try { window.localStorage.setItem(VIEW_KEY, v); } catch (e) { /* not remembered */ }
        });
      })(roots[r]);
    }
  }

  function start() {
    var roots = document.querySelectorAll(".worlds, .rosette");
    for (var i = 0; i < roots.length; i++) { mount(roots[i]); }
    initToggle();
  }
  if (document.readyState === "loading") { document.addEventListener("DOMContentLoaded", start); }
  else { start(); }
})();
