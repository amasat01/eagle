/* "How it compares": the page's one device state, and the runtime bars' info lines.
 *
 * - Device state: the hardware pill (GPU | CPU) and the device chips are radio inputs;
 *   this script keeps them in the URL hash (#hw=gpu&dev=<device slug>) so a link opens
 *   on the same device, and announces every change as a "perf-device-change" event the
 *   radar listens to. Without script the radios and CSS still switch the bars.
 * - Bars: a row with an info line is focusable; hover or focus shows the line as a
 *   popover, a tap or Enter pins it under the row (.open), Escape or a tap elsewhere
 *   closes it. No layout change on hover (the bar grows by transform only; see custom.css).
 * - Bars view: the "Wall time" | "Uneven-batch efficiency" | "Early-finish saving" buttons (aria-pressed) set
 *   data-bars on the switch; the choice is shared by every device panel and remembered per
 *   viewer in localStorage (a convenience only: reads and writes are guarded).
 * Vanilla JS, no library.
 */
(function () {
  "use strict";

  function switchRoot() { return document.querySelector(".perf-switch"); }

  function readState(root) {
    var hw = root.querySelector('input[name="perf-hw"]:checked');
    var side = hw ? hw.getAttribute("data-side") : null;
    var dev = side ? root.querySelector('input[name="perf-' + side + '"]:checked') : null;
    return { side: side, slug: dev ? dev.getAttribute("data-slug") : null };
  }

  function announce(root) {
    var st = readState(root);
    root.dispatchEvent(new CustomEvent("perf-device-change", { bubbles: true, detail: st }));
    return st;
  }

  function writeHash(st) {
    if (!st.side) { return; }
    var h = "#hw=" + st.side + (st.slug ? "&dev=" + encodeURIComponent(st.slug) : "");
    try { history.replaceState(null, "", h); } catch (e) { /* file:// and sandboxed frames */ }
  }

  function applyHash(root) {
    var m = /^#hw=(gpu|cpu)(?:&dev=([^&]*))?$/.exec(location.hash);
    if (!m) { return; }
    var hw = root.querySelector('input[name="perf-hw"][data-side="' + m[1] + '"]');
    if (!hw) { return; }
    hw.checked = true;
    if (m[2]) {
      var slug = decodeURIComponent(m[2]);
      var devs = root.querySelectorAll('input[name="perf-' + m[1] + '"]');
      for (var i = 0; i < devs.length; i++) {
        if (devs[i].getAttribute("data-slug") === slug) { devs[i].checked = true; }
      }
    }
    announce(root);
  }

  function initState() {
    var root = switchRoot();
    if (!root) { return; }
    root.addEventListener("change", function (ev) {
      var t = ev.target;
      if (t && t.type === "radio" && /^perf-/.test(t.name)) { writeHash(announce(root)); }
    });
    window.addEventListener("hashchange", function () { applyHash(root); });
    applyHash(root);
  }

  function closeRows(except) {
    var open = document.querySelectorAll(".chart .row.open");
    for (var i = 0; i < open.length; i++) {
      if (open[i] !== except) { open[i].classList.remove("open"); open[i].setAttribute("aria-expanded", "false"); }
    }
  }

  function toggleRow(row) {
    var on = !row.classList.contains("open");
    closeRows(row);
    row.classList.toggle("open", on);
    row.setAttribute("aria-expanded", on ? "true" : "false");
  }

  function initBars() {
    document.addEventListener("click", function (ev) {
      var row = ev.target.closest ? ev.target.closest(".chart .row[tabindex]") : null;
      if (row) { toggleRow(row); } else { closeRows(null); }
    });
    document.addEventListener("keydown", function (ev) {
      var row = ev.target.closest ? ev.target.closest(".chart .row[tabindex]") : null;
      if (!row || ev.target !== row) { return; }
      if (ev.key === "Enter" || ev.key === " ") { ev.preventDefault(); toggleRow(row); }
      else if (ev.key === "Escape") { closeRows(null); }
    });
  }

  var VIEW_KEY = "raptor-bars-view";

  function initView() {
    var root = switchRoot();
    if (!root || !root.querySelector(".bars-toggle")) { return; }
    function show(view) {
      root.setAttribute("data-bars", view);
      var opts = root.querySelectorAll(".bars-opt");
      for (var i = 0; i < opts.length; i++) {
        opts[i].setAttribute("aria-pressed", opts[i].getAttribute("data-view") === view ? "true" : "false");
      }
      closeRows(null);
    }
    var saved = null;
    try { saved = window.localStorage.getItem(VIEW_KEY); } catch (e) { /* private window, blocked storage */ }
    root.classList.add("bars-js");
    show(saved === "eff" || saved === "early" ? saved : "wall");
    root.addEventListener("click", function (ev) {
      var b = ev.target.closest ? ev.target.closest(".bars-opt") : null;
      if (!b) { return; }
      var view = b.getAttribute("data-view");
      view = view === "eff" || view === "early" ? view : "wall";
      show(view);
      try { window.localStorage.setItem(VIEW_KEY, view); } catch (e) { /* the choice just is not remembered */ }
    });
  }

  function start() { initState(); initBars(); initView(); }
  if (document.readyState === "loading") { document.addEventListener("DOMContentLoaded", start); }
  else { start(); }
})();
