/* The landing radar: where each tool shines.
 *
 * The page carries, per side (GPU, CPU), one SVG for the default device plus a JSON
 * block with every device's vertex positions and tooltip texts, all computed by
 * tools/sync_cards.py. This script only draws and interacts:
 *   - redraws the tool shapes for the device the page's switch selects, morphing the
 *     shape over ~250 ms (no morph under prefers-reduced-motion);
 *   - hover or focus of a legend item or a shape brings that tool to the front and dims
 *     the others; a click on a legend item toggles the tool (at least one stays on);
 *   - hover or focus of an axis label, or hover of a vertex, shows a tooltip (in the detailed
 *     mode: fills the side panel with what the axis measures, its source and every tool's value);
 *   - on the landing radar an axis is a major: its tooltip is a compact table of its minors for every
 *     visible tool (value, ratio to the best, arm, "no reading"), and the minors' radii are drawn as
 *     small ticks on the spoke, so the plotted mean is visibly the mean of them;
 *   - touch: a tap acts as hover, a tap outside resets.
 * Vanilla JS, no library. Without script the server-drawn SVG and the data table remain.
 */
(function () {
  "use strict";
  var NS = "http://www.w3.org/2000/svg";
  var MORPH_MS = 250;

  function reduced() {
    return !!(window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches);
  }
  function el(name, attrs, parent) {
    var e = document.createElementNS(NS, name);
    for (var k in attrs) { e.setAttribute(k, attrs[k]); }
    if (parent) { parent.appendChild(e); }
    return e;
  }

  function mount(root) {
    var data = JSON.parse(root.querySelector(".radar-json").textContent);
    var side = data.side;
    var svg = root.querySelector(".radar-svg");
    var layer = svg.querySelector(".radar-tools");
    var tip = root.querySelector(".radar-tip");
    var stage = root.querySelector(".radar-stage");
    var legend = root.querySelectorAll(".radar-leg");
    var panel = root.querySelector(".radar-panel");
    var detailed = data.mode === "detailed";
    var nAx = data.axes.length;
    var centre = data.centre;
    var groups = {}, order = [], shown = {};   // shown[tool] = [[x,y]|null per axis]
    var devSlug = null, raf = 0, hl = null;
    var on = {};

    while (layer.firstChild) { layer.removeChild(layer.firstChild); }
    data.tools.forEach(function (t) {
      var g = el("g", { "class": "radar-tool", "data-tool": t.id }, layer);
      g.style.setProperty("--c", "var(" + t.color + ")");
      var poly = el("polygon", { "class": "radar-area" }, g);
      var dots = [];
      for (var i = 0; i < nAx; i++) {
        var q = data.axes[i].qual;
        var c = el("circle", { "class": "radar-vertex" + (q ? " q" : ""), r: q ? 5 : 4, tabindex: -1,
                               "data-axis": data.axes[i].key, "data-i": i }, g);
        dots.push(c);
      }
      groups[t.id] = { g: g, poly: poly, dots: dots };
      order.push(t.id);
      on[t.id] = true;
      for (var b = 0; b < legend.length; b++) {
        if (legend[b].getAttribute("data-tool") === t.id && legend[b].getAttribute("aria-pressed") === "false") { on[t.id] = false; }
      }
      g.classList.toggle("is-off", !on[t.id]);
      shown[t.id] = [];
    });

    function applyHollow(slug) {
      var hol = data.devices[slug].hol || {};
      data.tools.forEach(function (t) {
        for (var i = 0; i < nAx; i++) {
          groups[t.id].dots[i].classList.toggle("q", !!data.axes[i].qual || (hol[t.id] || []).indexOf(i) >= 0);
        }
      });
    }

    function paint(tool, pos) {
      var gr = groups[tool], pts = [];
      for (var i = 0; i < nAx; i++) {
        var p = pos[i];
        if (p) {
          pts.push(p[0].toFixed(1) + "," + p[1].toFixed(1));
          gr.dots[i].setAttribute("cx", p[0]); gr.dots[i].setAttribute("cy", p[1]);
          gr.dots[i].style.display = "";
        } else { gr.dots[i].style.display = "none"; }
      }
      gr.poly.setAttribute("points", pts.join(" "));
      gr.poly.style.display = pts.length >= 2 ? "" : "none";
    }

    function target(slug, tool) {
      var src = (data.devices[slug].pts[tool]) || {}, out = [];
      for (var i = 0; i < nAx; i++) { out.push(src[i] || null); }
      return out;
    }

    function setDevice(slug, animate) {
      if (!data.devices[slug] || slug === devSlug) { return; }
      var from = {}, to = {};
      data.tools.forEach(function (t) { from[t.id] = shown[t.id].slice(); to[t.id] = target(slug, t.id); });
      devSlug = slug;
      var devs = root.querySelectorAll(".radar-dev");
      for (var d = 0; d < devs.length; d++) { devs[d].hidden = devs[d].getAttribute("data-slug") !== slug; }
      svg.setAttribute("aria-label", data.devices[slug].aria);
      applyHollow(slug);
      hideTip();
      if (raf) { cancelAnimationFrame(raf); raf = 0; }
      var finish = function () {
        data.tools.forEach(function (t) { shown[t.id] = to[t.id]; paint(t.id, to[t.id]); });
      };
      if (!animate || reduced() || !from[data.tools[0].id].length) { finish(); return; }
      var t0 = null;
      var step = function (now) {
        if (t0 === null) { t0 = now; }
        var k = Math.min(1, (now - t0) / MORPH_MS);
        k = k * k * (3 - 2 * k);
        data.tools.forEach(function (t) {
          var pos = [];
          for (var i = 0; i < nAx; i++) {
            var a = from[t.id][i], b = to[t.id][i];
            if (!a && !b) { pos.push(null); continue; }
            a = a || centre; b = b || centre;
            pos.push([a[0] + (b[0] - a[0]) * k, a[1] + (b[1] - a[1]) * k]);
          }
          paint(t.id, pos);
        });
        if (k < 1) { raf = requestAnimationFrame(step); } else { raf = 0; finish(); }
      };
      raf = requestAnimationFrame(step);
    }

    /* ring units, drawn along the spoke of the axis being read (never on every axis at rest) */
    var ringLayer = el("g", { "class": "radar-hover-rings" }, svg);
    var RINGS = [1.0, 0.6, 0.2];
    function showBands(i) {
      var ax = data.axes[i], g = data.geom, a = ax.ang;
      if (!ax.bands || !ax.bands.length) { return; }
      var ca = Math.cos(a), sa = Math.sin(a), px = -sa, py = ca;
      if (py < 0) { px = -px; py = -py; }               // the lower side of the spoke: rings use the upper
      ax.bands.forEach(function (b) {
        var rr = g[4] * b.r, d = 0.09;
        var p0 = [g[2] + rr * Math.cos(a - d), g[3] + rr * Math.sin(a - d)];
        var p1 = [g[2] + rr * Math.cos(a + d), g[3] + rr * Math.sin(a + d)];
        el("path", { "class": "radar-band", d: "M" + p0[0].toFixed(1) + "," + p0[1].toFixed(1) + " A" + rr.toFixed(1) + "," + rr.toFixed(1) +
                     " 0 0 1 " + p1[0].toFixed(1) + "," + p1[1].toFixed(1) }, ringLayer);
        var t = el("text", { "class": "radar-hring radar-bandlabel", x: (g[2] + rr * ca + px * 14).toFixed(1), y: (g[3] + rr * sa + py * 14).toFixed(1),
                             "text-anchor": px >= 0 ? "start" : "end", "dominant-baseline": "central" }, ringLayer);
        t.textContent = b.t;
        var bb = t.getBBox();
        var pill = el("rect", { "class": "radar-pill", x: (bb.x - 4).toFixed(1), y: (bb.y - 2).toFixed(1), width: (bb.width + 8).toFixed(1),
                                height: (bb.height + 4).toFixed(1), rx: 4 }, ringLayer);
        ringLayer.insertBefore(pill, t);
      });
    }
    function showRings(key) {
      clearRings();
      var i = axisIndex(key), dev = data.devices[devSlug];
      if (i < 0 || !dev.rl[key]) { return; }
      showBands(i);
      var parts = dev.rl[key].split(" · "), g = data.geom, a = data.axes[i].ang;
      var ca = Math.cos(a), sa = Math.sin(a), px = -sa, py = ca;
      if (py > 0) { px = -px; py = -py; }               // offset to the upper side of the spoke
      for (var k = 0; k < RINGS.length && k < parts.length; k++) {
        var x = g[2] + g[4] * RINGS[k] * ca + px * 12, y = g[3] + g[4] * RINGS[k] * sa + py * 12;
        var t = el("text", { "class": "radar-hring", x: x.toFixed(1), y: y.toFixed(1), "text-anchor": "middle",
                             "dominant-baseline": "central" }, ringLayer);
        t.textContent = parts[k] + (k === 0 && data.axes[i].edge ? "" : "");
      }
    }
    function clearRings() { while (ringLayer.firstChild) { ringLayer.removeChild(ringLayer.firstChild); } }

    /* ---- a major's minors: ticks on its spoke and a table in the tooltip ---- */
    function minorsOf(key) { var ax = data.axes[axisIndex(key)]; return (ax && ax.minors) || []; }
    function visibleTools(only) {
      return data.tools.filter(function (t) { return only ? t.id === only : on[t.id]; });
    }
    function showTicks(key, only) {
      var i = axisIndex(key), mins = minorsOf(key), dev = data.devices[devSlug];
      if (i < 0 || !mins.length) { return; }
      var g = data.geom, a = data.axes[i].ang, ca = Math.cos(a), sa = Math.sin(a);
      var vis = visibleTools(only);
      vis.forEach(function (t, ti) {
        var m = dev.mn && dev.mn[t.id] && dev.mn[t.id][key];
        if (!m) { return; }
        var off = (ti - (vis.length - 1) / 2) * 6;
        mins.forEach(function (mi) {
          var v = m[mi.key];
          if (!v) { return; }
          var x = g[2] + g[4] * v[3] * ca - sa * off, y = g[3] + g[4] * v[3] * sa + ca * off;
          var c = el("circle", { "class": "radar-mtick" + (mi.qual ? " q" : ""), cx: x.toFixed(1), cy: y.toFixed(1), r: 3.4 }, ringLayer);
          c.style.setProperty("--c", "var(" + t.color + ")");
        });
      });
    }
    /* a wide tooltip sits on the half of the chart the hovered spoke is not in, so the spoke and its ticks stay in view */
    function narrow() { return root.clientWidth < (detailed ? 560 : 720); }   /* the CSS container-query widths */
    function placeWide(key) {
      tip.classList.toggle("inflow", narrow());   /* a phone-width column: the tip sits under the chart, which stays in view */
      if (narrow()) { tip.style.left = ""; tip.style.top = ""; tip.hidden = false; return; }
      var i = axisIndex(key), g = data.geom, a = data.axes[i].ang, vb = svg.viewBox.baseVal;
      var sr = svg.getBoundingClientRect(), st = stage.getBoundingClientRect(), k = sr.width / vb.width;
      var spokeY = sr.top - st.top + (g[3] + g[4] * 0.6 * Math.sin(a) - vb.y) * k;
      tip.hidden = false;
      var h = tip.offsetHeight, H = stage.clientHeight;
      tip.style.left = "0px";
      tip.style.top = Math.max(0, spokeY < H / 2 ? H - h : 0) + "px";
    }
    function cellText(div, text, cls) {
      var d = document.createElement("div");
      if (cls) { d.className = cls; }
      d.textContent = text;
      div.appendChild(d);
    }
    /* narrow columns: one block per tool, a line per minor, instead of the table */
    function majorList(key, only) {
      var dev = data.devices[devSlug], mins = minorsOf(key);
      var list = document.createElement("div");
      list.className = "radar-mlist";
      visibleTools(only).forEach(function (t) {
        var box = document.createElement("div"), head = document.createElement("div"), chip = document.createElement("span");
        box.className = "mtool";
        head.className = "mhead";
        chip.className = "chip";
        chip.style.setProperty("--c", "var(" + t.color + ")");
        head.appendChild(chip);
        var tt = dev.tips[t.id] && dev.tips[t.id][key];
        var strong = document.createElement("strong");
        strong.textContent = t.id;
        head.appendChild(strong);
        head.appendChild(document.createTextNode(tt ? " · " + (mins.length > 1 ? "mean " : "") + tt.v : " · no reading"));
        box.appendChild(head);
        var m = (dev.mn && dev.mn[t.id] && dev.mn[t.id][key]) || {};
        mins.forEach(function (mi) {
          var v = m[mi.key], row = document.createElement("div");
          row.className = "mrow";
          var name = document.createElement("span");
          name.className = "mn";
          name.textContent = mi.label + ": ";
          row.appendChild(name);
          if (!v) {
            var miss = document.createElement("em");
            miss.textContent = "no reading";
            row.appendChild(miss);
          } else {
            row.appendChild(document.createTextNode(v[0]));
            var more = [v[1] ? v[1].replace(/ the best$/, "") : "", v[2]].filter(Boolean).join(" · ");
            if (more) {
              var sm = document.createElement("span");
              sm.className = "mn";
              sm.textContent = " · " + more;
              row.appendChild(sm);
            }
          }
          box.appendChild(row);
        });
        list.appendChild(box);
      });
      return list;
    }
    function majorTable(key, only) {
      if (narrow()) { return majorList(key, only); }
      var dev = data.devices[devSlug], mins = minorsOf(key);
      var wrap = document.createElement("div");
      wrap.className = "radar-mwrap";
      var tb = document.createElement("table");
      tb.className = "radar-mtable";
      var head = document.createElement("tr");
      var corner = document.createElement("th");
      corner.textContent = "Tool";
      head.appendChild(corner);
      mins.forEach(function (mi) {
        var th = document.createElement("th");
        th.scope = "col";
        th.textContent = mi.label;
        if (mi.qual) { th.className = "q"; }
        head.appendChild(th);
      });
      var mean = document.createElement("th");
      mean.scope = "col";
      mean.textContent = mins.length > 1 ? "Mean, as drawn" : "As drawn";
      head.appendChild(mean);
      tb.appendChild(head);
      visibleTools(only).forEach(function (t) {
        var tr = document.createElement("tr");
        var th = document.createElement("th");
        th.scope = "row";
        var chip = document.createElement("span");
        chip.className = "chip";
        chip.style.setProperty("--c", "var(" + t.color + ")");
        th.appendChild(chip);
        th.appendChild(document.createTextNode(t.id));
        tr.appendChild(th);
        var m = (dev.mn && dev.mn[t.id] && dev.mn[t.id][key]) || {};
        mins.forEach(function (mi) {
          var td = document.createElement("td"), v = m[mi.key];
          if (!v) {
            td.className = "miss";
            td.textContent = "no reading";
          } else {
            cellText(td, v[0], "v");
            if (v[1]) { cellText(td, v[1].replace(/ the best$/, ""), "r"); }
            if (v[2]) { cellText(td, v[2].replace(/ \([^()]*\)$/, ""), "a"); }
          }
          tr.appendChild(td);
        });
        var tt = dev.tips[t.id] && dev.tips[t.id][key], tm = document.createElement("td");
        if (tt) { cellText(tm, tt.v, "v"); if (tt.d) { cellText(tm, tt.d, "a"); } } else { tm.className = "miss"; tm.textContent = "no reading"; }
        tr.appendChild(tm);
        tb.appendChild(tr);
      });
      wrap.appendChild(tb);
      return wrap;
    }

    /* ---- the fan: a major opens into its minors ----
     * On hover, focus or tap of a major that has several minors, the chart grows (about 12 %) and the major's spike splits into one spoke
     * per minor, clustered round the major's direction in a 120 degree sector; the other majors share the rest of the circle evenly and a
     * dashed spoke keeps the mean. The tools' shapes animate from the single spike (each minor starts on its major's spoke) to the fan.
     * Escape, a tap elsewhere or leaving the chart folds it back. The layer is built here from the same JSON as the rest chart. */
    var FAN_MS = 320, FAN_SECTOR = 120, FAN_GROW = 1.12;
    var fanKey = null, fanLayer = null, fanT = 0, fanRaf = 0, fanSpec = null, fanTimer = 0;
    var majorKeys = data.axes.map(function (a) { return a.key; });
    function wrapPi(a) { return ((a + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) - Math.PI; }
    function nameLines(name) {
      var w = name.split(" ");
      if (name.length <= 16 || w.length < 2) { return [name]; }
      var best = 1, bd = 1e9;
      for (var k = 1; k < w.length; k++) { var d = Math.abs(w.slice(0, k).join(" ").length - w.slice(k).join(" ").length); if (d < bd) { bd = d; best = k; } }
      return [w.slice(0, best).join(" "), w.slice(best).join(" ")];
    }
    function fanAxes(key) {
      var ia = axisIndex(key), mins = minorsOf(key), g = data.geom;
      var base = data.axes[ia].ang, sec = FAN_SECTOR * Math.PI / 180, m = mins.length;
      var out = [], gap = (2 * Math.PI - sec) / (nAx - 1);
      for (var j = 0; j < m; j++) {
        out.push({ key: mins[j].key, label: mins[j].label, qual: mins[j].qual, minor: true, parent: ia, a1: base - sec / 2 + sec * (j + 0.5) / m, a0: base });
      }
      for (var o = 0; o < nAx - 1; o++) {
        var i = (ia + 1 + o) % nAx;
        out.push({ key: data.axes[i].key, label: data.axes[i].label, qual: data.axes[i].qual, minor: false, parent: i, a1: base + sec / 2 + gap * (o + 0.5), a0: data.axes[i].ang,
                   dir: data.axes[i].dir });
      }
      return { key: key, ia: ia, base: base, sec: sec, axes: out, mins: mins };
    }
    function restPoint(tool, i) {          /* the tool's single spike on major i: [angle, radius fraction] or null */
      var p = (data.devices[devSlug].pts[tool] || {})[i];
      if (!p) { return null; }
      return [data.axes[i].ang, Math.hypot(p[0] - centre[0], p[1] - centre[1]) / data.geom[4]];
    }
    function fanEnd(tool, ax, spec) {      /* [angle, radius fraction] of the tool on a fan axis at the end of the animation */
      var dev = data.devices[devSlug];
      if (ax.minor) {
        var v = dev.mn && dev.mn[tool] && dev.mn[tool][spec.key] && dev.mn[tool][spec.key][ax.key];
        return v ? [ax.a1, v[3]] : null;
      }
      var r = restPoint(tool, ax.parent);
      return r ? [ax.a1, r[1]] : null;
    }
    function fanStart(tool, ax) {          /* a minor starts on its major's spike; another major starts where it rests */
      var r = restPoint(tool, ax.parent);
      return r ? [ax.a0, r[1]] : null;
    }
    function polar(c, ang, rf, Rt) { return [c[0] + Rt * rf * Math.cos(ang), c[1] + Rt * rf * Math.sin(ang)]; }
    function buildFan(key) {
      clearFan();
      fanSpec = fanAxes(key);
      var g = data.geom, Rf = g[4] * FAN_GROW;
      fanLayer = el("g", { "class": "radar-fan" }, svg);
      var sp = fanSpec;
      /* the sector wash, the rings and the spokes, which fade in with t */
      var back = el("g", { "class": "radar-fan-back" }, fanLayer);
      sp.back = back;
      sp.rings = RINGS.map(function () { return el("polygon", { "class": "radar-ring" }, back); });
      sp.wash = el("path", { "class": "radar-fan-wash" }, back);
      sp.spokes = sp.axes.map(function () { return el("line", { "class": "radar-spoke" }, back); });
      sp.mean = el("line", { "class": "radar-spoke radar-spoke-mean" }, back);
      sp.meanLabel = el("text", { "class": "radar-dir radar-mean-label" }, fanLayer);
      sp.meanLabel.textContent = "mean";
      sp.labels = sp.axes.map(function (ax, n) {
        var gg = el("g", { "class": "radar-fan-label" + (ax.minor ? " minor" : " major") }, fanLayer);
        var t = el("text", { "class": "radar-axis radar-fan-axis" + (ax.qual ? " q" : ""), "data-axis": ax.key, "data-fan": ax.minor ? "minor" : "major", tabindex: 0, role: "button" }, gg);
        t.setAttribute("aria-label", ax.label.replace(" (qualitative)", "") + (ax.minor ? ": a minor of " + data.axes[sp.ia].label.replace(" (qualitative)", "") : ": show its minors"));
        if (ax.minor) { t.setAttribute("data-feature", ax.key); }
        var nm = ax.label.replace(" (qualitative)", "");
        var lines = nameLines(nm);
        var sub = ax.minor ? "" : (ax.dir || (ax.qual ? "(qualitative)" : ""));
        gg.__lines = lines; gg.__sub = sub; gg.__t = t;
        var icon = el("use", { href: "#rico-" + side + "-" + data.mode + "-" + ax.key, width: 20, height: 20 }, el("g", { "class": "radar-icon radar-fan-icon", "aria-hidden": "true" }, gg));
        gg.__icon = icon.parentNode;
        return gg;
      });
      /* the tools */
      sp.tools = {};
      data.tools.forEach(function (t) {
        var g2 = el("g", { "class": "radar-tool radar-fan-tool" + (on[t.id] ? "" : " is-off") + (hl === t.id ? " is-hl" : ""), "data-tool": t.id }, fanLayer);
        g2.style.setProperty("--c", "var(" + t.color + ")");
        var poly = el("polygon", { "class": "radar-area" }, g2);
        var dots = sp.axes.map(function (ax) {
          var hol = (data.devices[devSlug].hol || {})[t.id] || [];
          var q = ax.qual || (!ax.minor && hol.indexOf(ax.parent) >= 0);
          var c = el("circle", { "class": "radar-vertex" + (q ? " q" : ""), r: q ? 5 : 4, tabindex: -1, "data-axis": ax.key, "data-fan-v": ax.minor ? "minor" : "major" }, g2);
          c.addEventListener("mouseenter", function () { setHl(t.id); showFanVertex(c, t.id, ax); });
          c.addEventListener("mouseleave", function () { setHl(null); hideTip(); });
          c.addEventListener("focus", function () { setHl(t.id); showFanVertex(c, t.id, ax); });
          c.addEventListener("blur", function () { setHl(null); hideTip(); });
          return c;
        });
        var mean = null, rp = restPoint(t.id, sp.ia);
        if (rp) { mean = el("circle", { "class": "radar-mean", r: 5.5 }, g2); }
        sp.tools[t.id] = { g: g2, poly: poly, dots: dots, mean: mean };
        g2.addEventListener("mouseenter", function () { if (on[t.id]) { setHl(t.id); } });
        g2.addEventListener("mouseleave", function () { setHl(null); });
      });
      sp.axes.forEach(function (ax, n) {
        var lab = sp.labels[n];
        var tt = lab.__t;
        var open = function () { if (!ax.minor) { fan(ax.key); } else { showFanLabel(lab, ax); } };
        tt.addEventListener("mouseenter", open);
        tt.addEventListener("focus", open);
        tt.addEventListener("mouseleave", function () { if (ax.minor) { hideTip(); } });
        tt.addEventListener("blur", function () { if (ax.minor) { hideTip(); } });
        tt.addEventListener("click", function (ev) { ev.stopPropagation(); if (!ax.minor) { fan(ax.key); } else { showFanLabel(lab, ax); } });
        tt.addEventListener("keydown", function (ev) { if (ev.key === "Enter" || ev.key === " ") { ev.preventDefault(); tt.dispatchEvent(new Event("click")); } });
        var ic = lab.__icon;
        ic.addEventListener("mouseenter", open); ic.addEventListener("focus", open);
        ic.addEventListener("mouseleave", function () { if (ax.minor) { hideTip(); } });
        ic.addEventListener("click", function (ev) { ev.stopPropagation(); if (!ax.minor) { fan(ax.key); } else { showFanLabel(lab, ax); } });
        ic.addEventListener("keydown", function (ev) { if (ev.key === "Enter" || ev.key === " ") { ev.preventDefault(); ic.dispatchEvent(new Event("click")); } });
      });
    }
    function clearFan() {
      if (fanLayer && fanLayer.parentNode) { fanLayer.parentNode.removeChild(fanLayer); }
      fanLayer = null; fanSpec = null;
    }
    function layoutFanLabels() {            /* names and icons at the open fan's size, once (and again when the column changes width) */
      var sp = fanSpec; if (!sp) { return; }
      var g = data.geom, Rt = g[4] * FAN_GROW, isNarrow = narrow();
      sp.labels.forEach(function (gg, n) {
        var ax = sp.axes[n], a = ax.a1, ca = Math.cos(a), sa = Math.sin(a);
        var t2 = gg.__t, ic = gg.__icon, isz = 22;
        if (isNarrow) {                       /* a phone-width column: the icon stands in for the name, on the rim */
          var dr = Rt + 28, ds = 28;
          ic.setAttribute("transform", "translate(" + (centre[0] + dr * ca - ds / 2).toFixed(1) + " " + (centre[1] + dr * sa - ds / 2).toFixed(1) + ")");
          ic.firstChild.setAttribute("width", ds); ic.firstChild.setAttribute("height", ds);
          ic.setAttribute("tabindex", "0"); ic.setAttribute("role", "button"); ic.setAttribute("aria-label", t2.getAttribute("aria-label"));
          return;
        }
        var lr = Rt + 24, anchor = ca > 0.1 ? "start" : (ca < -0.1 ? "end" : "middle");
        var lines = gg.__lines, sub = gg.__sub, fs = ax.minor ? 14 : 16, rows = lines.length + (sub ? 1 : 0), total = rows * (fs + 4);
        var lx = centre[0] + lr * ca, ly = centre[1] + lr * sa, shift = anchor === "start" ? isz + 6 : (anchor === "middle" ? (isz + 6) / 2 : 0);
        var y = sa < -0.5 ? ly - total + fs : (sa > 0.9 ? ly + fs : ly - total / 2 + fs);
        while (t2.firstChild) { t2.removeChild(t2.firstChild); }
        var yy = y;
        lines.forEach(function (ln) { var ts = el("tspan", { "class": "radar-name" + (ax.minor ? " radar-minor-name" : ""), x: (lx + shift).toFixed(1), y: yy.toFixed(1) }, t2); ts.textContent = ln; yy += fs + 4; });
        if (sub) { var td = el("tspan", { "class": "radar-dir", x: (lx + shift).toFixed(1), y: yy.toFixed(1) }, t2); td.textContent = sub; }
        t2.setAttribute("text-anchor", anchor);
        ic.setAttribute("tabindex", "-1");
        var bb = null;
        try { bb = t2.getBBox(); } catch (e) { bb = null; }
        var ix = bb && bb.width ? bb.x - 6 - isz : lx, iy = bb && bb.width ? bb.y + bb.height / 2 - isz / 2 : y - fs;
        ic.setAttribute("transform", "translate(" + ix.toFixed(1) + " " + iy.toFixed(1) + ")");
        ic.firstChild.setAttribute("width", isz); ic.firstChild.setAttribute("height", isz);
      });
      var pm = polar(centre, sp.base, 1, Rt + 16);
      sp.meanLabel.setAttribute("x", pm[0].toFixed(1)); sp.meanLabel.setAttribute("y", (pm[1] + 4).toFixed(1));
      sp.meanLabel.setAttribute("text-anchor", "middle");
    }
    function drawFan(t) {                  /* t in [0, 1]: 0 = the rest chart, 1 = the fan */
      var sp = fanSpec; if (!sp) { return; }
      var g = data.geom, Rt = g[4] * (1 + (FAN_GROW - 1) * t), e = t;
      sp.back.style.opacity = Math.min(1, t * 1.4).toFixed(3);
      var angs = sp.axes.map(function (ax) { return ax.a0 + wrapPi(ax.a1 - ax.a0) * e; });
      /* rings as polygons through the (moving) spoke angles, in angle order */
      var ord = angs.map(function (a, n) { return n; }).sort(function (x, y) { return wrapPi(angs[x] - sp.base) - wrapPi(angs[y] - sp.base); });
      RINGS.forEach(function (rf, k) {
        sp.rings[k].setAttribute("points", ord.map(function (n) { var p = polar(centre, angs[n], rf, Rt); return p[0].toFixed(1) + "," + p[1].toFixed(1); }).join(" "));
      });
      sp.axes.forEach(function (ax, n) {
        var p = polar(centre, angs[n], 1, Rt);
        sp.spokes[n].setAttribute("x1", centre[0]); sp.spokes[n].setAttribute("y1", centre[1]);
        sp.spokes[n].setAttribute("x2", p[0].toFixed(1)); sp.spokes[n].setAttribute("y2", p[1].toFixed(1));
      });
      var pm = polar(centre, sp.base, 1, Rt);
      sp.mean.setAttribute("x1", centre[0]); sp.mean.setAttribute("y1", centre[1]); sp.mean.setAttribute("x2", pm[0].toFixed(1)); sp.mean.setAttribute("y2", pm[1].toFixed(1));
      var a0 = sp.base - sp.sec / 2 * e, a1 = sp.base + sp.sec / 2 * e, Rw = Rt + 10;
      sp.wash.setAttribute("d", "M" + centre[0] + " " + centre[1] + "L" + polar(centre, a0, 1, Rw)[0].toFixed(1) + " " + polar(centre, a0, 1, Rw)[1].toFixed(1) +
        "A" + Rw.toFixed(1) + " " + Rw.toFixed(1) + " 0 0 1 " + polar(centre, a1, 1, Rw)[0].toFixed(1) + " " + polar(centre, a1, 1, Rw)[1].toFixed(1) + "Z");
      data.tools.forEach(function (tl) {
        var tg = sp.tools[tl.id], pts = [];
        sp.axes.forEach(function (ax, n) {
          var s0 = fanStart(tl.id, ax), s1 = fanEnd(tl.id, ax, sp), dot = tg.dots[n];
          if (!s1) { dot.style.display = "none"; return; }
          var r = (s0 ? s0[1] : s1[1]) + (s1[1] - (s0 ? s0[1] : s1[1])) * e;
          var a = s0 ? s0[0] + wrapPi(s1[0] - s0[0]) * e : s1[0];
          var p = polar(centre, a, r, Rt);
          dot.setAttribute("cx", p[0].toFixed(1)); dot.setAttribute("cy", p[1].toFixed(1)); dot.style.display = "";
          pts.push([a, p]);
        });
        pts.sort(function (x, y) { return wrapPi(x[0] - sp.base) - wrapPi(y[0] - sp.base); });
        tg.poly.setAttribute("points", pts.map(function (q) { return q[1][0].toFixed(1) + "," + q[1][1].toFixed(1); }).join(" "));
        tg.poly.style.display = pts.length >= 2 ? "" : "none";
        if (tg.mean) {
          var rp = restPoint(tl.id, sp.ia), pmn = polar(centre, sp.base, rp[1], Rt);
          tg.mean.setAttribute("cx", pmn[0].toFixed(1)); tg.mean.setAttribute("cy", pmn[1].toFixed(1));
        }
      });
      /* the labels were laid out once at the open fan's size; they fade in as the fan opens */
      var op = Math.max(0, (t - 0.55) / 0.45).toFixed(3);
      sp.labels.forEach(function (gg) { gg.style.opacity = op; });
      sp.meanLabel.style.opacity = op;
      fanT = t;
    }
    function fitFan() {
      /* the chart grows: widen the viewBox to the fan's labels and the stage with it, so the type keeps its size */
      var g = data.geom, narrowNow = root.clientWidth < (detailed ? 560 : 720);
      if (narrowNow) {
        var Rf = g[4] * FAN_GROW, pad = 48;
        svg.setAttribute("viewBox", [g[2] - Rf - pad, g[3] - Rf - pad, 2 * (Rf + pad), 2 * (Rf + pad)].join(" "));
        return;
      }
      var grow = (fanKey ? 1 : 0), vb = full.split(" ").map(Number), extra = 100 * grow;
      svg.setAttribute("viewBox", [vb[0] - extra / 2, vb[1], vb[2] + extra, vb[3]].join(" "));
      stage.style.maxWidth = (780 + extra) + "px";
    }
    function animateFan(from, to, done) {
      if (fanRaf) { cancelAnimationFrame(fanRaf); fanRaf = 0; }
      if (reduced()) { drawFan(to); if (done) { done(); } return; }
      var t0 = null;
      var step = function (now) {
        if (t0 === null) { t0 = now; }
        var k = Math.min(1, (now - t0) / FAN_MS);
        k = k * k * (3 - 2 * k);
        drawFan(from + (to - from) * k);
        if (now - t0 < FAN_MS) { fanRaf = requestAnimationFrame(step); } else { fanRaf = 0; drawFan(to); if (done) { done(); } }
      };
      fanRaf = requestAnimationFrame(step);
    }
    function fan(key) {
      if (detailed || !devSlug || minorsOf(key).length < 2) { return false; }
      if (fanTimer) { clearTimeout(fanTimer); fanTimer = 0; }
      if (fanKey === key) { return true; }
      var from = fanKey ? 1 : 0;
      fanKey = key;
      hideTip();
      buildFan(key);
      svg.classList.add("is-fan");
      root.classList.add("is-fanned");
      fitFan();
      layoutFanLabels();
      drawFan(from);
      animateFan(from, 1);
      return true;
    }
    function unfan() {
      if (!fanKey) { return; }
      if (fanTimer) { clearTimeout(fanTimer); fanTimer = 0; }
      fanKey = null;
      hideTip();
      animateFan(fanT, 0, function () {
        if (fanKey) { return; }
        clearFan();
        svg.classList.remove("is-fan");
        root.classList.remove("is-fanned");
        fitFan();
        fit();
      });
      stage.style.maxWidth = "";
    }
    function unfanSoon() { if (fanKey && !fanTimer) { fanTimer = setTimeout(function () { fanTimer = 0; unfan(); }, 260); } }
    function minorTip(tool, ax, anchor) {
      var dev = data.devices[devSlug], mn = dev.mn && dev.mn[tool] && dev.mn[tool][fanSpec.key], v = mn && mn[ax.key];
      tip.textContent = "";
      tip.classList.remove("wide");
      line(tip, tool + " · " + ax.label.replace(" (qualitative)", ""), "radar-tip-h");
      if (!v) { line(tip, "no reading"); } else { line(tip, v[0]); if (v[1]) { line(tip, v[1]); } if (v[2]) { line(tip, v[2], "radar-tip-s"); } }
      var d = dev.mdesc && dev.mdesc[ax.key]; if (d) { line(tip, d, "radar-tip-s"); }
      line(tip, dev.card, "radar-tip-s");
      var sr = svg.getBoundingClientRect(), st = stage.getBoundingClientRect(), r = anchor.getBoundingClientRect();
      placeTip(r.right - st.left, r.top - st.top);
    }
    function showFanVertex(c, tool, ax) {
      if (ax.minor) { minorTip(tool, ax, c); return; }
      var save = fanKey; void save;
      showVertex(c, tool);
    }
    function showFanLabel(gg, ax) {
      var dev = data.devices[devSlug];
      tip.textContent = "";
      tip.classList.remove("wide");
      line(tip, ax.label.replace(" (qualitative)", ""), "radar-tip-h");
      var d = dev.mdesc && dev.mdesc[ax.key]; if (d) { line(tip, d, "radar-tip-s"); }
      var rl = dev.mrl && dev.mrl[ax.key]; if (rl) { line(tip, "Scale, edge to centre: " + rl, "radar-tip-s"); }
      visibleTools(null).forEach(function (t) {
        var mn = dev.mn && dev.mn[t.id] && dev.mn[t.id][fanSpec.key], v = mn && mn[ax.key];
        line(tip, t.id + ": " + (v ? v[0] + (v[1] ? " (" + v[1].replace(/ the best$/, "") + ")" : "") : "no reading"));
      });
      line(tip, dev.card, "radar-tip-s");
      var st = stage.getBoundingClientRect(), r = gg.getBoundingClientRect();
      placeTip(r.right - st.left, r.top - st.top);
    }

    /* ---- highlight and toggle ---- */
    function applyHl() {
      root.classList.toggle("has-hl", !!hl);
      data.tools.forEach(function (t) {
        groups[t.id].g.classList.toggle("is-hl", hl === t.id);
      });
      for (var i = 0; i < legend.length; i++) {
        legend[i].classList.toggle("is-hl", legend[i].getAttribute("data-tool") === hl);
      }
      order.forEach(function (id) { layer.appendChild(groups[id].g); });
      if (hl) { layer.appendChild(groups[hl].g); }
      if (fanSpec) {
        data.tools.forEach(function (t) { fanSpec.tools[t.id].g.classList.toggle("is-hl", hl === t.id); });
        if (hl) { fanLayer.appendChild(fanSpec.tools[hl].g); }
      }
    }
    function setHl(tool) { if (hl !== tool) { hl = tool; applyHl(); } }
    function toggle(tool) {
      var n = 0;
      for (var k in on) { if (on[k]) { n++; } }
      if (on[tool] && n === 1) { return; }
      on[tool] = !on[tool];
      groups[tool].g.classList.toggle("is-off", !on[tool]);
      if (fanSpec) { fanSpec.tools[tool].g.classList.toggle("is-off", !on[tool]); }
      for (var i = 0; i < legend.length; i++) {
        if (legend[i].getAttribute("data-tool") === tool) { legend[i].setAttribute("aria-pressed", on[tool] ? "true" : "false"); }
      }
      if (!on[tool] && hl === tool) { setHl(null); }
    }

    /* ---- tooltip ---- */
    function line(parent, text, cls) {
      var d = document.createElement("div");
      if (cls) { d.className = cls; }
      d.textContent = text;
      parent.appendChild(d);
    }
    function placeTip(x, y) {
      var w = stage.clientWidth;
      tip.hidden = false;
      var tw = tip.offsetWidth;
      var left = Math.max(0, Math.min(w - tw, x + 12));
      tip.style.left = left + "px";
      tip.style.top = Math.max(0, y - 8) + "px";
    }
    function hideTip() { tip.hidden = true; tip.classList.remove("wide"); clearRings(); }
    function axisIndex(key) {
      for (var i = 0; i < nAx; i++) { if (data.axes[i].key === key) { return i; } }
      return -1;
    }
    function toolLabel(id) { return id; }
    function showPanel(key) {
      var dev = data.devices[devSlug], ax = data.axes[axisIndex(key)];
      panel.textContent = "";
      line(panel, ax.label, "radar-panel-h");
      if (ax.what) { line(panel, ax.what); }
      if (ax.major) { line(panel, "One of the minors of " + ax.major + " on the landing radar.", "radar-tip-s"); }
      if (ax.edge) { line(panel, "Edge of the chart: " + ax.edge + " (" + ax.dir + ").", "radar-tip-s"); }
      line(panel, (ax.qual ? "Rubric: " : "Source: " + ax.src + "; ") + dev.desc[key], "radar-tip-s");
      showRings(key);
      var list = document.createElement("div");
      list.className = "radar-panel-list";
      panel.appendChild(list);
      data.tools.forEach(function (t) {
        var tt = dev.tips[t.id] && dev.tips[t.id][key];
        var row = document.createElement("div");
        row.className = "radar-panel-row" + (on[t.id] ? "" : " is-off");
        var chip = document.createElement("span");
        chip.className = "chip";
        chip.style.setProperty("--c", "var(" + t.color + ")");
        row.appendChild(chip);
        var txt = document.createElement("span");
        txt.textContent = t.id + ": " + (tt ? tt.v + (ax.qual ? "" : " (" + tt.r + ")") : "no reading");
        row.appendChild(txt);
        list.appendChild(row);
        if (tt && ax.qual) { line(list, tt.r, "radar-tip-s radar-panel-why"); }
        else if (tt && tt.a) { line(list, tt.a + (tt.d ? " · " + tt.d : ""), "radar-tip-s radar-panel-why"); }
      });
      if (!ax.qual) {
        var a = document.createElement("a");
        a.href = dev.link; a.textContent = "Open this device's card";
        panel.appendChild(a);
      }
      line(panel, dev.card, "radar-tip-s");
    }
    function showVertex(circle, tool) {
      var key = circle.getAttribute("data-axis"), dev = data.devices[devSlug];
      if (detailed) { showPanel(key); return; }
      if (!fanKey && minorsOf(key).length > 1) { fan(key); return; }
      var tt = dev.tips[tool] && dev.tips[tool][key];
      if (!tt) { return; }
      tip.textContent = "";
      tip.classList.remove("wide");
      if (minorsOf(key).length) {
        tip.classList.add("wide");
        line(tip, toolLabel(tool) + " · " + data.axes[axisIndex(key)].label, "radar-tip-h");
        tip.appendChild(majorTable(key, tool));
        line(tip, dev.card, "radar-tip-s");
        showRings(key);
        showTicks(key, tool);
        placeWide(key);
        return;
      }
      line(tip, toolLabel(tool) + " · " + data.axes[axisIndex(key)].label, "radar-tip-h");
      line(tip, tt.v);
      line(tip, tt.r);
      if (tt.a) { line(tip, tt.a, "radar-tip-s"); }
      if (tt.d) { line(tip, tt.d, "radar-tip-s"); }
      line(tip, dev.desc[key], "radar-tip-s");
      var ax0 = data.axes[axisIndex(key)];
      if (ax0.edge) { line(tip, "Edge of the chart: " + ax0.edge + " (" + ax0.dir + ")", "radar-tip-s"); }
      line(tip, dev.card, "radar-tip-s");
      showRings(key);
      var sr = svg.getBoundingClientRect(), st = stage.getBoundingClientRect();
      var k = sr.width / svg.viewBox.baseVal.width;
      placeTip(sr.left - st.left + parseFloat(circle.getAttribute("cx")) * k,
               sr.top - st.top + parseFloat(circle.getAttribute("cy")) * k);
    }
    function showAxis(label) {
      var key = label.getAttribute("data-axis"), dev = data.devices[devSlug];
      if (detailed) { showPanel(key); return; }
      if (minorsOf(key).length > 1) { fan(key); return; }
      tip.textContent = "";
      tip.classList.remove("wide");
      line(tip, data.axes[axisIndex(key)].label, "radar-tip-h");
      var axx = data.axes[axisIndex(key)];
      if (!axx.minors || !axx.minors.length) { line(tip, dev.desc[key], "radar-tip-s"); }
      if (axx.what) { line(tip, axx.what, "radar-tip-s"); }
      if (axx.edge) { line(tip, "Edge of the chart: " + axx.edge + " (" + axx.dir + ")", "radar-tip-s"); }
      showRings(key);
      if (minorsOf(key).length) {
        tip.classList.add("wide");
        tip.appendChild(majorTable(key, null));
        showTicks(key, null);
        line(tip, dev.card, "radar-tip-s");
        placeWide(key);
        return;
      } else {
        data.tools.forEach(function (t) {
          var tt = dev.tips[t.id] && dev.tips[t.id][key];
          if (on[t.id]) { line(tip, t.id + ": " + (tt ? tt.v + " (" + tt.r + ")" : "no reading")); }
        });
      }
      line(tip, dev.card, "radar-tip-s");
      var anchor = label;
      if (label.tagName === "LI" || label.getBoundingClientRect().width === 0) {
        anchor = svg.querySelector('.radar-icon.disc[data-axis="' + key + '"]') || label;
      }
      var lr = anchor.getBoundingClientRect(), st = stage.getBoundingClientRect();
      placeTip(lr.right - st.left, lr.top - st.top);
    }

    /* ---- events ---- */
    for (var i = 0; i < legend.length; i++) {
      (function (b) {
        var tool = b.getAttribute("data-tool");
        b.addEventListener("mouseenter", function () { setHl(tool); });
        b.addEventListener("mouseleave", function () { setHl(null); });
        b.addEventListener("focus", function () { setHl(tool); });
        b.addEventListener("blur", function () { setHl(null); });
        b.addEventListener("click", function () { toggle(tool); });
      })(legend[i]);
    }
    data.tools.forEach(function (t) {
      var gr = groups[t.id];
      gr.poly.addEventListener("mouseenter", function () { if (on[t.id]) { setHl(t.id); } });
      gr.poly.addEventListener("mouseleave", function () { setHl(null); });
      gr.dots.forEach(function (c) {
        c.addEventListener("mouseenter", function () { setHl(t.id); showVertex(c, t.id); });
        c.addEventListener("mouseleave", function () { setHl(null); hideTip(); });
        c.addEventListener("focus", function () { setHl(t.id); showVertex(c, t.id); });
        c.addEventListener("blur", function () { setHl(null); hideTip(); });
      });
    });
    var icons = svg.querySelectorAll(".radar-icon.disc");
    for (var ic = 0; ic < icons.length; ic++) {
      (function (n) {
        n.addEventListener("mouseenter", function () { showAxis(n); });
        n.addEventListener("mouseleave", hideTip);
        n.addEventListener("focus", function () { showAxis(n); });
        n.addEventListener("blur", hideTip);
        n.addEventListener("click", function () { showAxis(n); });
        n.addEventListener("keydown", function (ev) { if (ev.key === "Enter" || ev.key === " ") { ev.preventDefault(); showAxis(n); } });
      })(icons[ic]);
    }
    var labels = svg.querySelectorAll(".radar-axis");
    for (var j = 0; j < labels.length; j++) {
      (function (l) {
        l.addEventListener("mouseenter", function () { showAxis(l); });
        l.addEventListener("mouseleave", hideTip);
        l.addEventListener("focus", function () { showAxis(l); });
        l.addEventListener("blur", hideTip);
      })(labels[j]);
    }
    var items = root.querySelectorAll(".radar-axlist li");
    for (var m = 0; m < items.length; m++) {
      (function (li) {
        li.addEventListener("mouseenter", function () { showAxis(li); });
        li.addEventListener("focus", function () { showAxis(li); });
        li.addEventListener("click", function (ev) { ev.stopPropagation(); showAxis(li); });
        li.addEventListener("blur", hideTip);
      })(items[m]);
    }
    root.addEventListener("keydown", function (ev) { if (ev.key === "Escape") { hideTip(); setHl(null); unfan(); } });
    document.addEventListener("pointerdown", function (ev) {
      if (!ev.target.closest || !ev.target.closest(".radar-tool, .radar-leg, .radar-axis, .radar-axlist, .radar-icon, .radar-fan")) { hideTip(); setHl(null); unfan(); }
    });
    stage.addEventListener("mouseleave", unfanSoon);
    stage.addEventListener("mouseenter", function () { if (fanTimer) { clearTimeout(fanTimer); fanTimer = 0; } });

    /* ---- icons beside the axis names: the name is placed by the server; the icon is set against the measured width ---- */
    var wideIcons = svg.querySelectorAll(".radar-icon.wide");
    function placeIcons() {
      for (var k = 0; k < wideIcons.length; k++) {
        var g = wideIcons[k], t = svg.querySelector('.radar-axis[data-axis="' + g.getAttribute("data-axis") + '"]'), bb = null;
        try { bb = t ? t.getBBox() : null; } catch (e) { bb = null; }
        if (!bb || !bb.width) { continue; }
        var u = g.firstChild, size = parseFloat(u.getAttribute("width")), gap = 6;
        var x = bb.x - gap - size;
        g.setAttribute("transform", "translate(" + x.toFixed(1) + " " + (bb.y + bb.height / 2 - size / 2).toFixed(1) + ")");
      }
    }
    placeIcons();
    if (document.fonts && document.fonts.ready) { document.fonts.ready.then(placeIcons); }

    /* ---- narrow column: crop the chart to its disc (the names give way to numbers) ---- */
    var full = svg.getAttribute("viewBox");
    function fit() {
      if (fanKey) { fitFan(); layoutFanLabels(); placeIcons(); return; }
      var g = data.geom, pad = 48;
      var narrow = root.clientWidth < (detailed ? 560 : 720);   /* the same widths as the CSS container queries */
      svg.setAttribute("viewBox", narrow ? [g[2] - g[4] - pad, g[3] - g[4] - pad, 2 * (g[4] + pad), 2 * (g[4] + pad)].join(" ") : full);
      placeIcons();
    }
    fit();
    if (window.ResizeObserver) { new ResizeObserver(fit).observe(root); } else { window.addEventListener("resize", fit); }

    /* ---- the page's device state ---- */
    function followSwitch(animate) {
      var r = document.querySelector('.perf-switch input[name="perf-' + side + '"]:checked');
      var slug = r ? r.getAttribute("data-slug") : null;
      if (slug) { setDevice(slug, animate); }
    }
    var sw = document.querySelector(".perf-switch");
    if (sw) { sw.addEventListener("perf-device-change", function () { followSwitch(true); }); }
    var first = root.querySelector(".radar-dev:not([hidden])");
    devSlug = first ? first.getAttribute("data-slug") : null;
    if (devSlug) {
      data.tools.forEach(function (t) { shown[t.id] = target(devSlug, t.id); paint(t.id, shown[t.id]); });
      svg.setAttribute("aria-label", data.devices[devSlug].aria);
      applyHollow(devSlug);
    }
    followSwitch(false);
  }

  function start() {
    var roots = document.querySelectorAll(".radar");
    for (var i = 0; i < roots.length; i++) { mount(roots[i]); }
  }
  if (document.readyState === "loading") { document.addEventListener("DOMContentLoaded", start); }
  else { start(); }
})();
