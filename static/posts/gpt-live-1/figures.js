/* Animated figures for the GPT-Live-1 post.
 * Vanilla JS + inline SVG. Each figure renders into a <div class="glf" id="glf-*">.
 */
(function () {
  "use strict";

  var NS = "http://www.w3.org/2000/svg";
  var C = {
    user: "#f59e0b",
    agent: "#2563eb",
    backend: "#7c3aed",
    tool: "#10b981",
    text: "#1e293b",
    muted: "#64748b",
    faint: "#94a3b8",
    border: "#e2e8f0",
    bg: "#f8fafc",
    think: "#a78bfa",
    silence: "#cbd5e1",
    userText: "#fbbf24",
    agentText: "#60a5fa"
  };
  var reduceMotion = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  // ---------- tiny helpers ----------
  function el(tag, attrs, parent) {
    var e = document.createElement(tag);
    if (attrs) for (var k in attrs) {
      if (k === "text") e.textContent = attrs[k];
      else if (k === "html") e.innerHTML = attrs[k];
      else e.setAttribute(k, attrs[k]);
    }
    if (parent) parent.appendChild(e);
    return e;
  }
  function svgEl(tag, attrs, parent) {
    var e = document.createElementNS(NS, tag);
    if (attrs) for (var k in attrs) {
      if (k === "text") e.textContent = attrs[k];
      else e.setAttribute(k, attrs[k]);
    }
    if (parent) parent.appendChild(e);
    return e;
  }
  function svgRoot(w, h, parent) {
    var s = svgEl("svg", { viewBox: "0 0 " + w + " " + h, role: "img" }, parent);
    return s;
  }
  function box(parent, x, y, w, h, fill, stroke, r) {
    return svgEl("rect", { x: x, y: y, width: w, height: h, rx: r == null ? 8 : r, fill: fill, stroke: stroke || "none", "stroke-width": 1.5, "class": "glf-box" }, parent);
  }
  function label(parent, x, y, txt, size, anchor, weight, fill) {
    return svgEl("text", { x: x, y: y, "font-size": size || 12, "text-anchor": anchor || "middle", "font-weight": weight || 400, fill: fill || C.text, text: txt }, parent);
  }
  function arrow(parent, x1, y1, x2, y2, color, id, dashed) {
    var a = svgEl("line", { x1: x1, y1: y1, x2: x2, y2: y2, stroke: color, "stroke-width": 2, "marker-end": "url(#" + id + ")", "class": "glf-arrow" }, parent);
    if (dashed) a.setAttribute("stroke-dasharray", "5 4");
    return a;
  }
  function marker(defs, id, color) {
    var m = svgEl("marker", { id: id, viewBox: "0 0 10 10", refX: 9, refY: 5, markerWidth: 7, markerHeight: 7, orient: "auto-start-reverse" }, defs);
    svgEl("path", { d: "M0,0 L10,5 L0,10 z", fill: color }, m);
  }
  function fmtMs(ms) { return ms >= 1000 ? (ms / 1000).toFixed(2) + " s" : Math.round(ms) + " ms"; }
  function fmtUsd(x, d) { return "$" + x.toFixed(d == null ? 3 : d); }

  // Simple step player used by the scripted figures.
  function player(steps, onStep, opts) {
    opts = opts || {};
    var i = -1, timer = null, playing = false;
    var api = {};
    function show(n) {
      i = (n + steps.length) % steps.length;
      onStep(i, steps[i]);
      if (api.onIndex) api.onIndex(i);
    }
    function tick() {
      show(i + 1);
      if (playing) timer = setTimeout(tick, steps[i].dur || opts.dur || 2200);
    }
    api.play = function () { if (playing) return; playing = true; if (api.onPlay) api.onPlay(true); tick(); };
    api.pause = function () { playing = false; clearTimeout(timer); if (api.onPlay) api.onPlay(false); };
    api.toggle = function () { playing ? api.pause() : api.play(); };
    api.next = function () { api.pause(); show(i + 1); };
    api.prev = function () { api.pause(); show(i - 1); };
    api.goto = function (n) { api.pause(); show(n); };
    api.index = function () { return i; };
    show(0);
    return api;
  }

  function controls(parent, p, nSteps) {
    var bar = el("div", { "class": "glf-controls" }, parent);
    var play = el("button", { "class": "glf-primary", text: "Play" }, bar);
    var prev = el("button", { text: "◀" }, bar);
    var next = el("button", { text: "▶" }, bar);
    var dots = el("div", { "class": "glf-steps" }, bar);
    var ds = [];
    for (var k = 0; k < nSteps; k++) {
      var d = el("span", null, dots);
      (function (idx) { d.addEventListener("click", function () { p.goto(idx); }); })(k);
      ds.push(d);
    }
    play.addEventListener("click", p.toggle);
    prev.addEventListener("click", p.prev);
    next.addEventListener("click", p.next);
    p.onPlay = function (on) { play.textContent = on ? "Pause" : "Play"; };
    p.onIndex = function (idx) { ds.forEach(function (d, j) { d.className = j === idx ? "on" : ""; }); };
    p.onIndex(p.index());
    return bar;
  }

  // Auto-play when scrolled into view, pause when out of view.
  function autoplay(node, p) {
    if (reduceMotion || !("IntersectionObserver" in window)) return;
    var started = false;
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) {
        if (en.isIntersecting && !started) { started = true; p.play(); }
        else if (!en.isIntersecting && started) { p.pause(); started = false; }
      });
    }, { threshold: 0.35 });
    io.observe(node);
  }

  // =====================================================================
  // Figure 1: Realtime vs Live architecture
  // =====================================================================
  function figArch(root) {
    el("p", { "class": "glf-title", text: "Figure 1. GPT-Realtime does everything in one model. GPT-Live-1 delegates reasoning and tools." }, root);
    var W = 760, H = 330;
    var svg = svgRoot(W, H, root);
    var defs = svgEl("defs", null, svg);
    marker(defs, "ar-user", C.user); marker(defs, "ar-agent", C.agent); marker(defs, "ar-backend", C.backend); marker(defs, "ar-tool", C.tool); marker(defs, "ar-faint", C.faint);

    // ---- Left panel: GPT-Realtime ----
    var L = svgEl("g", null, svg);
    label(L, 165, 22, "GPT-Realtime (one model)", 13, "middle", 600, C.muted);
    // user
    var lUser = box(L, 20, 130, 70, 46, "#fff7ed", C.user);
    label(L, 55, 150, "User", 12, "middle", 600);
    var lUserWave = svgEl("g", null, L);
    for (var i = 0; i < 5; i++) svgEl("rect", { x: 36 + i * 8, y: 158, width: 4, height: 10, rx: 2, fill: C.user, "class": "glf-wave", style: "animation-delay:" + (i * 0.12) + "s" }, lUserWave);
    // model
    var lModel = box(L, 130, 95, 110, 120, "#eff6ff", C.agent);
    label(L, 185, 118, "gpt-realtime", 12, "middle", 600);
    label(L, 185, 138, "listen", 10.5, "middle", 400, C.muted);
    label(L, 185, 156, "reason", 10.5, "middle", 400, C.muted);
    label(L, 185, 174, "call tools", 10.5, "middle", 400, C.muted);
    label(L, 185, 192, "speak", 10.5, "middle", 400, C.muted);
    // tools
    var lTools = box(L, 280, 130, 60, 46, "#ecfdf5", C.tool);
    label(L, 310, 150, "Tools", 12, "middle", 600);
    label(L, 310, 166, "DB / API", 10, "middle", 400, C.muted);
    var lA1 = arrow(L, 92, 140, 128, 140, C.user, "ar-user");
    var lA2 = arrow(L, 128, 170, 92, 170, C.agent, "ar-agent");
    var lA3 = arrow(L, 242, 140, 278, 140, C.tool, "ar-tool");
    var lA4 = arrow(L, 278, 168, 242, 168, C.tool, "ar-tool");
    var lNote = label(L, 165, 250, "", 11, "middle", 500, C.muted);
    var lNote2 = label(L, 165, 268, "", 11, "middle", 400, C.muted);
    var lState = label(L, 185, 82, "", 10.5, "middle", 600, C.agent);

    // divider
    svgEl("line", { x1: 375, y1: 10, x2: 375, y2: H - 30, stroke: C.border, "stroke-width": 1.5, "stroke-dasharray": "4 4" }, svg);

    // ---- Right panel: GPT-Live ----
    var R = svgEl("g", null, svg);
    label(R, 570, 22, "GPT-Live-1 (frontend + backend)", 13, "middle", 600, C.muted);
    var rUser = box(R, 395, 130, 70, 46, "#fff7ed", C.user);
    label(R, 430, 150, "User", 12, "middle", 600);
    var rUserWave = svgEl("g", null, R);
    for (i = 0; i < 5; i++) svgEl("rect", { x: 411 + i * 8, y: 158, width: 4, height: 10, rx: 2, fill: C.user, "class": "glf-wave", style: "animation-delay:" + (i * 0.12) + "s" }, rUserWave);
    var rFront = box(R, 500, 95, 110, 120, "#eff6ff", C.agent);
    label(R, 555, 118, "frontend", 12, "middle", 600);
    label(R, 555, 134, "GPT-Live-1", 10.5, "middle", 400, C.muted);
    label(R, 555, 156, "listen + speak", 10.5, "middle", 400, C.muted);
    label(R, 555, 174, "at the same time", 10.5, "middle", 400, C.muted);
    label(R, 555, 196, "decide when to", 10.5, "middle", 400, C.muted);
    label(R, 555, 208, "delegate", 10.5, "middle", 400, C.muted);
    var rBack = box(R, 650, 95, 95, 74, "#f5f3ff", C.backend);
    label(R, 697, 118, "backend", 12, "middle", 600);
    label(R, 697, 134, "GPT-6 Astra", 10.5, "middle", 400, C.muted);
    label(R, 697, 150, "or your agents", 10.5, "middle", 400, C.muted);
    var rTools = box(R, 665, 200, 65, 40, "#ecfdf5", C.tool);
    label(R, 697, 218, "Tools", 12, "middle", 600);
    label(R, 697, 232, "DB / API", 10, "middle", 400, C.muted);
    var rA1 = arrow(R, 467, 140, 498, 140, C.user, "ar-user");
    var rA2 = arrow(R, 498, 170, 467, 170, C.agent, "ar-agent");
    var rDel = arrow(R, 612, 118, 648, 118, C.backend, "ar-backend");
    var rDelLbl = label(R, 630, 108, "delegate", 9.5, "middle", 500, C.backend);
    var rRet = arrow(R, 648, 150, 612, 150, C.backend, "ar-backend");
    var rRetLbl = label(R, 630, 166, "", 9, "middle", 500, C.backend);
    var rT1 = arrow(R, 697, 171, 697, 198, C.tool, "ar-tool");
    var rT2 = arrow(R, 712, 198, 712, 171, C.tool, "ar-tool");
    var rState = label(R, 555, 82, "", 10.5, "middle", 600, C.agent);
    var rNote = label(R, 570, 265, "", 11, "middle", 500, C.muted);
    var rNote2 = label(R, 570, 283, "", 11, "middle", 400, C.muted);
    // moving packet
    var pkt = svgEl("circle", { r: 5, fill: C.backend, opacity: 0 }, R);

    var caption = el("p", { "class": "glf-caption" }, root);

    var steps = [
      { cap: "<b>User asks for something that needs a tool.</b> Both models listen. GPT-Live can also speak while listening (“mm-hm”).",
        l: { user: 1, a1: 1, a2: 0, a3: 0, a4: 0, state: "listening", note: "", note2: "" },
        r: { user: 1, a1: 1, a2: 1, del: 0, ret: 0, t: 0, state: "listening + backchannel", note: "“mm-hm”", note2: "", pkt: null } },
      { cap: "<b>Deciding what to do.</b> GPT-Realtime reasons about the tool call itself. GPT-Live emits a delegation with the transcript so far, and keeps talking.",
        l: { user: 0, a1: 0, a2: 0, a3: 0, a4: 0, state: "reasoning", note: "nothing spoken meanwhile", note2: "" },
        r: { user: 0, a1: 0, a2: 1, del: 1, ret: 0, t: 0, state: "speaking + delegating", note: "“Let me check that.”", note2: "", pkt: "out" } },
      { cap: "<b>Tool call.</b> GPT-Realtime calls the tool. GPT-Live's backend calls the tool while the frontend holds the floor.",
        l: { user: 0, a1: 0, a2: 0, a3: 1, a4: 1, state: "tool call", note: "nothing spoken meanwhile", note2: "" },
        r: { user: 0, a1: 0, a2: 1, del: 0, ret: 0, t: 1, state: "holding the floor", note: "“Just a moment…” or silence", note2: "(measured as ‘silence during delegation’)", pkt: null } },
      { cap: "<b>Result.</b> GPT-Realtime speaks it. For GPT-Live, the backend result is appended to the frontend's context as <i>commentary</i> (speak this), <i>thinking</i> (know this), or <i>instructions</i> (behave like this), at most 500 tokens, and the frontend speaks it.",
        l: { user: 0, a1: 0, a2: 1, a3: 0, a4: 0, state: "speaking", note: "reason, call, speak in one model", note2: "" },
        r: { user: 0, a1: 0, a2: 1, del: 0, ret: 1, t: 0, state: "speaking result", note: "session.commentary.append", note2: "“Thursday at 2 pm is confirmed.”", pkt: "in" } }
    ];

    function setA(a, v) { a.style.opacity = v; }
    function apply(i, s) {
      caption.innerHTML = s.cap;
      var l = s.l, r = s.r;
      lUserWave.style.opacity = l.user ? 1 : 0.15;
      setA(lA1, l.a1 || 0.15); setA(lA2, l.a2 || 0.15); setA(lA3, l.a3 || 0.15); setA(lA4, l.a4 || 0.15);
      lTools.setAttribute("fill", l.a3 ? "#d1fae5" : "#ecfdf5");
      lModel.setAttribute("fill", l.a2 || l.state.indexOf("reason") === 0 || l.a3 ? "#dbeafe" : "#eff6ff");
      lState.textContent = l.state; lNote.textContent = l.note; lNote2.textContent = l.note2;
      rUserWave.style.opacity = r.user ? 1 : 0.15;
      setA(rA1, r.a1 || 0.15); setA(rA2, r.a2 || 0.15); setA(rDel, r.del || 0.15); setA(rRet, r.ret || 0.15);
      setA(rDelLbl, r.del || 0.3); setA(rT1, r.t || 0.15); setA(rT2, r.t || 0.15);
      rRetLbl.textContent = r.ret ? "append" : "";
      rTools.setAttribute("fill", r.t ? "#d1fae5" : "#ecfdf5");
      rBack.setAttribute("fill", r.del || r.t || r.ret ? "#ede9fe" : "#f5f3ff");
      rFront.setAttribute("fill", "#dbeafe");
      rState.textContent = r.state; rNote.textContent = r.note; rNote2.textContent = r.note2;
      // packet animation
      pkt.style.transition = "none"; pkt.setAttribute("opacity", 0);
      if (r.pkt && !reduceMotion) {
        var from = r.pkt === "out" ? [614, 118] : [646, 150], to = r.pkt === "out" ? [646, 118] : [614, 150];
        pkt.setAttribute("cx", from[0]); pkt.setAttribute("cy", from[1]);
        requestAnimationFrame(function () {
          pkt.style.transition = "cx 900ms ease-in-out, opacity 200ms";
          pkt.setAttribute("opacity", 1); pkt.setAttribute("cx", to[0]);
        });
      }
    }
    var p = player(steps, apply, { dur: 3200 });
    controls(root, p, steps.length);
    autoplay(root, p);
  }

  // =====================================================================
  // Figure 2: the frontend's interleaved token stream, with a delegation
  // =====================================================================
  function figStream(root) {
    el("p", { "class": "glf-title", text: "Figure 2. A guess at the frontend's sequence, including a delegation" }, root);
    var W = 760, H = 290;
    var svg = svgRoot(W, H, root);
    var g = svgEl("g", null, svg);

    var rows = [
      { key: "uaudio", y: 36, name: "user audio (in)", color: C.user },
      { key: "utext", y: 70, name: "user text (ASR)", color: C.userText },
      { key: "think", y: 104, name: "thinking", color: C.think },
      { key: "backend", y: 138, name: "backend", color: C.backend },
      { key: "atext", y: 172, name: "agent text", color: C.agentText },
      { key: "aaudio", y: 206, name: "agent audio (out)", color: C.agent }
    ];
    var X0 = 130, X1 = W - 15, ROWH = 20;
    rows.forEach(function (r) {
      label(g, X0 - 10, r.y + 14, r.name, 11, "end", 500, C.muted);
      svgEl("line", { x1: X0, y1: r.y + ROWH + 4, x2: X1, y2: r.y + ROWH + 4, stroke: C.border }, g);
    });
    // The tape scrolls left once it fills the width.
    var defs = svgEl("defs", null, svg);
    var cp = svgEl("clipPath", { id: "glf-stream-clip" }, defs);
    svgEl("rect", { x: X0 - 4, y: 26, width: X1 - X0 + 8, height: 250 }, cp);
    var tapeOuter = svgEl("g", { "clip-path": "url(#glf-stream-clip)" }, svg);
    var tokens = svgEl("g", null, tapeOuter);
    tokens.style.transition = "transform 250ms ease-out";
    var cursor = svgEl("line", { x1: X0, y1: 28, x2: X0, y2: 236, stroke: C.text, "stroke-width": 1.5, "stroke-dasharray": "3 3", opacity: 0.6 }, svg);
    var stepLbl = label(svg, X0, 22, "", 10.5, "start", 500, C.muted);
    var scaleLbl = label(svg, X1, 22, "", 10, "end", 400, C.faint);

    var ctrls = el("div", { "class": "glf-controls" }, root);
    var slid = el("label", { "class": "glf-slider", html: "user chunk <input type='range' min='1' max='5' step='1' value='5'> <output></output>" }, ctrls);
    var rng = slid.querySelector("input"), out = slid.querySelector("output");
    el("span", { text: "frame = 80 ms (12.5 Hz)" }, ctrls);
    var play = el("button", { "class": "glf-primary", text: "Pause" }, ctrls);
    var prev = el("button", { text: "◀", title: "previous step" }, ctrls);
    var next = el("button", { text: "▶", title: "next step" }, ctrls);
    var reset = el("button", { text: "Replay" }, ctrls);

    var legend = el("div", { "class": "glf-legend" }, root);
    [["user audio", C.user], ["user text", C.userText], ["thinking", C.think], ["delegation / injected result", C.backend], ["agent text", C.agentText], ["agent audio", C.agent], ["silence", C.silence]].forEach(function (it) {
      el("span", { html: "<i style='background:" + it[1] + "'></i>" + it[0] }, legend);
    });

    var readout = el("div", { "class": "glf-readout" }, root);
    function ro(lbl) { var d = el("div", null, readout); el("small", { text: lbl }, d); return el("strong", { text: "" }, d); }
    var roChunk = ro("chunk duration"), roFloor = ro("reaction floor"), roPre = ro("prefills / min"), roDec = ro("decodes / min");

    el("p", { "class": "glf-caption", html: "Each square is one token. User audio arrives in <b>chunks</b> of <i>k</i> frames and is prefilled at once (solid outline). Between chunks the model decodes user text, thinking, and agent tokens. When the request needs the backend, the frontend emits a <b>delegation</b> event and can say a filler. The backend works asynchronously while the frontend keeps prefilling user chunks and decoding (silent) agent audio. The result comes back as a <b>prefilled block</b> via <code>session.commentary.append</code>, and the agent speaks it. The streams are drawn as separate rows for clarity; the model sees one interleaved sequence with special tokens marking the boundaries. Drag the slider to change the chunk size, or step through with ◀ ▶." }, root);

    var FRAME_MS = 80, TOK_W = 9, TOK_GAP = 2;
    var k = 5, timer = null, running = true, seq = [], pos = 0;

    function buildSeq(k) {
      var s = [];
      var userFrames = 15, c = 0;
      while (c < userFrames) {
        var n = Math.min(k, userFrames - c);
        s.push({ row: "uaudio", kind: "prefill", n: n, label: "prefill " + n + " user frames (" + n * FRAME_MS + " ms)" });
        s.push({ row: "utext", kind: "decode", n: 1, label: "decode user text" });
        c += n;
      }
      s.push({ row: "uaudio", kind: "prefill", n: k, silent: true, mark: "user stops", markColor: C.user, label: "prefill " + k + " frames of silence" });
      s.push({ row: "think", kind: "decode", n: 2, label: "decode thinking" });
      s.push({ row: "backend", kind: "decode", n: 1, delegate: true, mark: "delegate", markColor: C.backend, label: "emit delegation event (transcript goes to backend)" });
      s.push({ row: "atext", kind: "decode", n: 2, label: "decode agent text" });
      s.push({ row: "aaudio", kind: "decode", n: k, mark: "filler (“let me check”)", markColor: C.agent, label: "decode " + k + " agent audio frames (filler)" });
      for (var j = 0; j < 3; j++) {
        s.push({ row: "uaudio", kind: "prefill", n: k, silent: true, label: "prefill " + k + " frames of user silence (backend still working)" });
        s.push({ row: "aaudio", kind: "decode", n: k, silent: true, label: "decode " + k + " frames of agent silence (backend still working)" });
      }
      s.push({ row: "backend", kind: "prefill", n: 6, inject: true, mark: "result injected", markColor: C.backend, label: "prefill backend result: session.commentary.append" });
      s.push({ row: "think", kind: "decode", n: 1, label: "decode thinking" });
      s.push({ row: "atext", kind: "decode", n: 3, label: "decode agent text" });
      for (var a = 0; a < 2; a++) {
        s.push({ row: "aaudio", kind: "decode", n: k, mark: a === 0 ? "agent speaks result" : null, markColor: C.agent, label: "decode " + k + " agent audio frames" });
        s.push({ row: "uaudio", kind: "prefill", n: k, silent: true, label: "prefill " + k + " frames of user silence" });
      }
      return s;
    }

    function rowY(key) { for (var i = 0; i < rows.length; i++) if (rows[i].key === key) return rows[i].y; return 0; }
    function rowColor(key) { for (var i = 0; i < rows.length; i++) if (rows[i].key === key) return rows[i].color; return C.text; }

    var x = X0, delegateX = null, markIdx = 0, history = [];
    function setCursor() {
      var shift = Math.min(0, X1 - x);
      tokens.style.transform = "translateX(" + shift + "px)";
      var cx = Math.min(x, X1) - 2;
      cursor.setAttribute("x1", cx); cursor.setAttribute("x2", cx);
    }
    function pause() { running = false; play.textContent = "Play"; clearTimeout(timer); }
    function reset_() {
      clearTimeout(timer);
      while (tokens.firstChild) tokens.removeChild(tokens.firstChild);
      seq = buildSeq(k); pos = 0; x = X0; delegateX = null; markIdx = 0; history = [];
      tokens.style.transition = "none"; setCursor();
      requestAnimationFrame(function () { tokens.style.transition = "transform 250ms ease-out"; });
      stepLbl.textContent = "";
      update();
      if (running) timer = setTimeout(step, 600);
    }
    function markLine(parent, mx, txt, col) {
      var ly = (markIdx++ % 2 === 0) ? 252 : 266;
      svgEl("line", { x1: mx - 3, y1: 30, x2: mx - 3, y2: ly - 10, stroke: col, "stroke-width": 1, "stroke-dasharray": "2 3" }, parent);
      label(parent, mx - 3, ly, txt, 9.5, "middle", 500, col);
    }
    // Draw one event. manual=true means a user click: no auto-advance.
    function step(manual) {
      clearTimeout(timer);
      if (pos >= seq.length) { if (!manual) timer = setTimeout(reset_, 4000); return; }
      var ev = seq[pos++];
      var w = ev.n * TOK_W + Math.max(0, ev.n - 1) * TOK_GAP;
      var y = rowY(ev.row), col = rowColor(ev.row);
      var evg = svgEl("g", null, tokens);
      history.push({ g: evg, x: x, delegateX: delegateX, markIdx: markIdx, label: stepLbl.textContent });
      if (ev.inject && delegateX !== null) {
        // the backend has been working asynchronously since the delegation event
        svgEl("rect", { x: delegateX + 2, y: y + 4, width: x - delegateX - 6, height: ROWH - 8, rx: 4, fill: C.backend, opacity: 0.08, stroke: C.backend, "stroke-width": 1, "stroke-dasharray": "3 3" }, evg);
        label(evg, (delegateX + x) / 2, y + 14, "backend working (async)", 9, "middle", 500, C.backend);
      }
      if (ev.kind === "prefill") {
        svgEl("rect", { x: x - 2, y: y - 2, width: w + 4, height: ROWH + 4, rx: 4, fill: "none", stroke: col, "stroke-width": 1.5, opacity: 0.9, "stroke-dasharray": ev.silent ? "3 3" : "none" }, evg);
      }
      var perTok = manual || reduceMotion ? 0 : 110;
      for (var i = 0; i < ev.n; i++) {
        var t = svgEl("rect", { x: x + i * (TOK_W + TOK_GAP), y: y, width: TOK_W, height: ROWH, rx: ev.delegate ? 6 : 2, fill: ev.silent ? C.silence : col, "class": "glf-token", opacity: perTok ? 0 : 1 }, evg);
        if (perTok) (function (t, d) { setTimeout(function () { t.setAttribute("opacity", 1); }, ev.kind === "prefill" ? 0 : d * perTok); })(t, i);
      }
      if (ev.mark) markLine(evg, x, ev.mark, ev.markColor || col);
      if (ev.delegate) delegateX = x + w;
      x += w + 4;
      setCursor();
      stepLbl.textContent = ev.label || "";
      if (manual || !running) return;
      var delay = ev.kind === "prefill" ? 1100 : 700 + ev.n * 110;
      timer = setTimeout(step, reduceMotion ? 0 : delay);
    }
    function undo() {
      if (!history.length) return;
      var h = history.pop();
      tokens.removeChild(h.g);
      pos--; x = h.x; delegateX = h.delegateX; markIdx = h.markIdx;
      stepLbl.textContent = h.label;
      setCursor();
    }
    function update() {
      out.textContent = k + " frame" + (k === 1 ? "" : "s") + " = " + (k * FRAME_MS) + " ms";
      roChunk.textContent = fmtMs(k * FRAME_MS);
      roFloor.textContent = "≈ " + fmtMs(k * FRAME_MS + FRAME_MS);
      var framesPerMin = 60 * 1000 / FRAME_MS;
      roPre.textContent = Math.round(framesPerMin / k) + " × " + k + " tok";
      roDec.textContent = "≈ " + Math.round(framesPerMin * 1.4);
      scaleLbl.textContent = "one square = one token = one " + FRAME_MS + " ms frame (audio rows)";
    }
    rng.addEventListener("input", function () { k = parseInt(rng.value, 10); reset_(); });
    play.addEventListener("click", function () { if (running) { pause(); } else { running = true; play.textContent = "Pause"; step(); } });
    prev.addEventListener("click", function () { pause(); undo(); });
    next.addEventListener("click", function () { pause(); step(true); });
    reset.addEventListener("click", reset_);
    update();
    reset_();
    if (reduceMotion) pause();
  }

  // =====================================================================
  // Figure 3: cost calculator
  // =====================================================================
  function figCost(root) {
    el("p", { "class": "glf-title", text: "Figure 3. What would each OpenAI model charge for GPT-Live's token budget?" }, root);

    var MODELS = [
      { name: "GPT-5.5 Pro", inp: 30, out: 180 },
      { name: "o3-pro", inp: 20, out: 80 },
      { name: "GPT-6 Astra", inp: 10, out: 50 },
      { name: "GPT-5.5 / GPT-5.6 Sol", inp: 5, out: 30 },
      { name: "GPT Realtime 2 (audio)", inp: 4, out: 24 },
      { name: "GPT-5.4", inp: 2.5, out: 15 },
      { name: "GPT-5.6 Terra", inp: 2, out: 12 },
      { name: "GPT-5.4 mini", inp: 0.75, out: 4.5 }
    ];
    var LIVE_PER_MIN = 0.05;

    var ctrls = el("div", { "class": "glf-controls" }, root);
    function slider(labelTxt, min, max, step, val) {
      var l = el("label", { "class": "glf-slider", html: labelTxt + " <input type='range' min='" + min + "' max='" + max + "' step='" + step + "' value='" + val + "'> <output></output>" }, ctrls);
      return { input: l.querySelector("input"), out: l.querySelector("output") };
    }
    var sHz = slider("frame rate", 0, 3, 1, 1);        // index into HZ
    var HZ = [6.25, 12.5, 25, 50];
    var sChunk = slider("user chunk", 1, 10, 1, 5);
    var sText = slider("text : audio", 0, 50, 5, 20);    // percent
    var sSplit = slider("bill prefill as", 0, 1, 1, 0);  // 0 = input price, 1 = output price

    var readout = el("div", { "class": "glf-readout" }, root);
    function ro(lbl) { var d = el("div", null, readout); el("small", { text: lbl }, d); return el("strong", { text: "" }, d); }
    var roIn = ro("prefilled tokens / min"), roOut = ro("decoded tokens / min"), roPer = ro("implied $ / M tokens"), roNear = ro("closest list price");

    var W = 760, H = 30 + MODELS.length * 30 + 40;
    var svg = svgRoot(W, H, root);
    var X0 = 190, X1 = W - 70, BAR_H = 18;
    var maxUsd = 0.22;
    function xOf(v) { return X0 + Math.min(1, v / maxUsd) * (X1 - X0); }
    // axis
    var axisY = 20 + MODELS.length * 30 + 8;
    svgEl("line", { x1: X0, y1: axisY, x2: X1, y2: axisY, stroke: C.border }, svg);
    [0, 0.05, 0.1, 0.15, 0.2].forEach(function (v) {
      svgEl("line", { x1: xOf(v), y1: axisY, x2: xOf(v), y2: axisY + 4, stroke: C.faint }, svg);
      label(svg, xOf(v), axisY + 16, fmtUsd(v, 2), 9.5, "middle", 400, C.muted);
    });
    label(svg, (X0 + X1) / 2, axisY + 30, "implied cost per minute at the same token mix", 10, "middle", 500, C.muted);
    var bars = MODELS.map(function (m, i) {
      var y = 20 + i * 30;
      label(svg, X0 - 8, y + 13, m.name, 11, "end", 500);
      var r = svgEl("rect", { x: X0, y: y, width: 0, height: BAR_H, rx: 3, fill: C.faint, "class": "glf-bar" }, svg);
      var t = label(svg, X0 + 4, y + 13, "", 10, "start", 600, C.text);
      return { rect: r, txt: t, y: y };
    });
    // live line
    var liveX = xOf(LIVE_PER_MIN);
    svgEl("line", { x1: liveX, y1: 10, x2: liveX, y2: axisY, stroke: C.agent, "stroke-width": 2, "stroke-dasharray": "5 4" }, svg);
    label(svg, liveX + 5, 14, "GPT-Live-1: $0.05 / min", 10.5, "start", 600, C.agent);

    var caption = el("p", { "class": "glf-caption" }, root);

    function compute() {
      var hz = HZ[parseInt(sHz.input.value, 10)];
      var k = parseInt(sChunk.input.value, 10);
      var textRatio = parseInt(sText.input.value, 10) / 100;
      var asOut = sSplit.input.value === "1";
      var frames = hz * 60;                 // per stream per minute
      var inTok = frames;                   // user audio frames, prefilled
      var textTok = Math.round(2 * frames * textRatio); // text over both streams
      var outTok = frames + textTok;        // agent audio + all text
      sHz.out.textContent = hz + " Hz (" + Math.round(1000 / hz) + " ms)";
      sChunk.out.textContent = k + " fr = " + Math.round(k * 1000 / hz) + " ms";
      sText.out.textContent = "1 : " + (textRatio === 0 ? "∞" : (1 / textRatio).toFixed(textRatio >= 0.34 ? 1 : 0));
      sSplit.out.textContent = asOut ? "output (decode)" : "input";
      roIn.textContent = Math.round(inTok).toLocaleString() + "  (" + Math.round(frames / k) + " × " + k + ")";
      roOut.textContent = Math.round(outTok).toLocaleString();
      var total = inTok + outTok;
      roPer.textContent = fmtUsd(LIVE_PER_MIN / total * 1e6, 1) + " blended";
      var best = null;
      MODELS.forEach(function (m, i) {
        var perMin = (asOut ? m.out : m.inp) * inTok / 1e6 + m.out * outTok / 1e6;
        var b = bars[i];
        b.rect.setAttribute("width", Math.max(2, xOf(perMin) - X0));
        var over = perMin > LIVE_PER_MIN;
        b.rect.setAttribute("fill", over ? "#c4b5fd" : "#93c5fd");
        b.txt.textContent = fmtUsd(perMin, 3) + (perMin > maxUsd ? " →" : "");
        b.txt.setAttribute("x", Math.min(xOf(perMin), X1) + 6);
        var d = Math.abs(Math.log(perMin / LIVE_PER_MIN));
        if (!best || d < best.d) best = { m: m, d: d, perMin: perMin };
      });
      roNear.textContent = best.m.name;
      caption.innerHTML = "At <b>" + hz + " Hz</b> with <b>" + k + "-frame</b> chunks and <b>" + Math.round(textRatio * 100) + "%</b> text overhead, the frontend handles about <b>" + Math.round(total).toLocaleString() + "</b> tokens per minute, so $0.05 is roughly <b>" + fmtUsd(LIVE_PER_MIN / total * 1e6, 0) + " per million tokens</b>. Bars show what each model's list price would charge for that same mix" + (asOut ? ", pricing the tiny prefills as decode (closer to the GPU's view)" : "") + ". Closest: <b>" + best.m.name + "</b> at " + fmtUsd(best.perMin, 3) + "/min.";
    }
    [sHz, sChunk, sText, sSplit].forEach(function (s) { s.input.addEventListener("input", compute); });
    compute();
  }

  // =====================================================================
  // Figure 4: evaluation modes
  // =====================================================================
  function figEvals(root) {
    el("p", { "class": "glf-title", text: "Figure 6. Three harnesses, three things isolated" }, root);
    var W = 760, H = 320;
    var svg = svgRoot(W, H, root);
    var defs = svgEl("defs", null, svg);
    marker(defs, "ev-user", C.user); marker(defs, "ev-agent", C.agent); marker(defs, "ev-faint", C.faint);

    var X0 = 95, XM = 470, XE = W - 20;
    var STRIP0 = X0 + 140;            // strip start for CRAWL / WALK
    var RUN_STRIP0 = STRIP0 + 105;    // RUN leaves room for the caller box
    var lanes = [
      { name: "CRAWL", sub: "synthetic TTS, one request", y: 30, tests: "reasoning + task completion", stripX: STRIP0 },
      { name: "WALK", sub: "recorded WAV + noise / telephony / loss", y: 120, tests: "robustness to real audio", stripX: STRIP0 },
      { name: "RUN", sub: "second GPT-Live plays the caller", y: 210, tests: "the whole full-duplex system", stripX: RUN_STRIP0 }
    ];
    var laneG = lanes.map(function (ln) {
      var g = svgEl("g", null, svg);
      label(g, 12, ln.y + 22, ln.name, 13, "start", 700, C.text);
      label(g, 12, ln.y + 38, ln.sub, 9.5, "start", 400, C.muted);
      label(g, 12, ln.y + 52, "isolates: " + ln.tests, 9.5, "start", 500, C.faint);
      // timeline strip
      svgEl("rect", { x: ln.stripX, y: ln.y + 8, width: XM - ln.stripX, height: 34, rx: 6, fill: C.bg, stroke: C.border }, g);
      // model box
      box(g, XM + 20, ln.y + 4, 110, 44, "#eff6ff", C.agent);
      label(g, XM + 75, ln.y + 22, "GPT-Live-1", 11.5, "middle", 600);
      label(g, XM + 75, ln.y + 38, "assistant under test", 9, "middle", 400, C.muted);
      arrow(g, XM + 2, ln.y + 25, XM + 18, ln.y + 25, C.faint, "ev-faint");
      // out arrow
      arrow(g, XM + 132, ln.y + 25, XM + 165, ln.y + 25, C.agent, "ev-agent");
      box(g, XM + 168, ln.y + 4, XE - (XM + 168), 44, "#f8fafc", C.border);
      label(g, (XM + 168 + XE) / 2, ln.y + 21, "grade", 11, "middle", 600, C.muted);
      label(g, (XM + 168 + XE) / 2, ln.y + 36, ln.name === "RUN" ? "state + turns + latency" : "app state + tools", 8.5, "middle", 400, C.faint);
      return g;
    });

    // Lane 1 & 2 content: speech chunks then endless silence
    function speechThenSilence(g, y, jaggy) {
      var x = STRIP0 + 8, sx = x;
      var frames = svgEl("g", null, g);
      for (var i = 0; i < 14; i++) {
        var h = jaggy ? 8 + Math.round(12 * Math.abs(Math.sin(i * 1.3 + 0.4))) : 10 + Math.round(10 * Math.abs(Math.sin(i * 0.9)));
        svgEl("rect", { x: x, y: y + 25 - h / 2, width: 5, height: h, rx: 1.5, fill: C.user }, frames);
        x += 7;
      }
      label(g, sx, y + 2, "recording, 20 ms PCM appends", 8.5, "start", 500, C.user);
      var silX = x + 6;
      var sil = svgEl("g", null, g);
      for (var j = 0; silX + j * 7 < XM - 10; j++) {
        svgEl("rect", { x: silX + j * 7, y: y + 23, width: 5, height: 4, rx: 1, fill: C.silence, "class": "glf-token" }, sil);
      }
      label(g, XM - 8, y + 53, "then zero-valued silence until the quiet tail elapses", 8.5, "end", 500, C.muted);
      if (jaggy) {
        label(g, XM - 8, y + 65, "+ acoustic simulator: noise, 8 kHz μ-law, packet loss", 8.5, "end", 400, C.faint);
      }
      return { sil: sil };
    }
    var c1 = speechThenSilence(laneG[0], lanes[0].y, false);
    var c2 = speechThenSilence(laneG[1], lanes[1].y, true);

    // Lane 3: two live participants
    (function () {
      var g = laneG[2], y = lanes[2].y;
      var cbx = STRIP0, cbw = 95, cbc = cbx + cbw / 2;
      box(g, cbx, y + 4, cbw, 44, "#fff7ed", C.user);
      label(g, cbc, y + 22, "GPT-Live-1", 11.5, "middle", 600);
      label(g, cbc, y + 38, "simulated caller", 9, "middle", 400, C.muted);
      arrow(g, cbx + cbw + 2, y + 25, RUN_STRIP0 - 2, y + 25, C.user, "ev-user");
      // two overlapping waveforms, bounded to the strip
      var x = RUN_STRIP0 + 8;
      for (var i = 0; x + 5 < XM - 8; i++) {
        var hu = 4 + Math.round(9 * Math.abs(Math.sin(i * 0.7))) * (i % 11 < 6 ? 1 : 0.15);
        var ha = 4 + Math.round(9 * Math.abs(Math.cos(i * 0.5))) * (i % 11 >= 5 ? 1 : 0.15);
        svgEl("rect", { x: x, y: y + 17 - hu / 2, width: 5, height: hu, rx: 1.5, fill: C.user, opacity: 0.9 }, g);
        svgEl("rect", { x: x, y: y + 33 - ha / 2, width: 5, height: ha, rx: 1.5, fill: C.agent, opacity: 0.9 }, g);
        x += 7;
      }
      label(g, (RUN_STRIP0 + XM) / 2, y + 2, "continuous full duplex, 20 ms frames each way", 8.5, "middle", 500, C.muted);
      label(g, XM - 8, y + 94, "relay holds ≤ 400 ms of output audio (jitter buffer), so latency is measured through it", 8.5, "end", 400, C.faint);
      // return arrow: assistant audio back to the caller
      svgEl("path", { d: "M" + (XM + 75) + "," + (y + 50) + " C" + (XM + 75) + "," + (y + 72) + " " + cbc + "," + (y + 72) + " " + cbc + "," + (y + 50), fill: "none", stroke: C.agent, "stroke-width": 1.5, "class": "glf-dash", "marker-end": "url(#ev-agent)" }, g);
      label(g, (XM + 75 + cbc) / 2, y + 80, "assistant audio", 8.5, "middle", 500, C.agent);
    })();

    // Animate the silence rows: a moving highlight that marches to the right forever.
    if (!reduceMotion) {
      var t = 0;
      function tick() {
        t++;
        [c1, c2].forEach(function (c) {
          var kids = c.sil.childNodes;
          for (var i = 0; i < kids.length; i++) kids[i].setAttribute("opacity", ((i + t) % 6 === 0) ? 1 : 0.45);
        });
        setTimeout(tick, 220);
      }
      tick();
    }

    el("p", { "class": "glf-caption", html: "<b>CRAWL</b> and <b>WALK</b> stream a recording, then silence, until a 600 ms quiet tail with no pending delegation. <b>RUN</b> uses a second GPT-Live session as the caller, so overlap, interruptions, and backchannels happen for real." }, root);
  }

  // =====================================================================
  // Generic horizontal bar panel used for the benchmark figures
  // =====================================================================
  var BAR_FILL = { live: C.agent, openai: "#93c5fd", other: "#cbd5e1" };
  function barPanel(root, opts) {
    var wrap = el("div", { "class": "glf-panel" }, root);
    var sub = el("p", { "class": "glf-subtitle" }, wrap);
    sub.appendChild(document.createTextNode(opts.title + " "));
    if (opts.sourceUrl) {
      var a = el("a", { href: opts.sourceUrl, target: "_blank", rel: "noopener", text: opts.sourceText || "source" }, sub);
      a.className = "glf-source";
    }
    var items = opts.items, RH = 22, W = 760, LABEL_W = opts.labelWidth || 250;
    var H = items.length * RH + 30;
    var svg = svgRoot(W, H, wrap);
    var X0 = LABEL_W, X1 = W - 56, max = opts.max || 100;
    function xOf(v) { return X0 + (v / max) * (X1 - X0); }
    // gridlines
    [0, 25, 50, 75, 100].forEach(function (v) {
      if (v > max) return;
      svgEl("line", { x1: xOf(v), y1: 6, x2: xOf(v), y2: H - 22, stroke: C.border, "stroke-dasharray": v === 0 ? "none" : "2 3" }, svg);
      label(svg, xOf(v), H - 8, v + (opts.unit || "%"), 9, "middle", 400, C.faint);
    });
    var bars = items.map(function (it, i) {
      var y = 8 + i * RH;
      label(svg, X0 - 8, y + 14, it.name, 10.5, "end", it.cls === "live" ? 700 : 400, it.cls === "live" ? C.text : C.muted);
      var r = svgEl("rect", { x: X0, y: y + 2, width: 0, height: RH - 6, rx: 3, fill: BAR_FILL[it.cls || "other"], "class": "glf-bar" }, svg);
      var t = label(svg, X0 + 4, y + 14, (opts.fmt || function (v) { return v.toFixed(1) + "%"; })(it.value), 10, "start", 600, C.text);
      t.setAttribute("opacity", 0);
      return { r: r, t: t, v: it.value };
    });
    function reveal() {
      bars.forEach(function (b, i) {
        setTimeout(function () {
          b.r.setAttribute("width", Math.max(1, xOf(b.v) - X0));
          b.t.setAttribute("x", xOf(b.v) + 5);
          b.t.style.transition = "opacity 300ms";
          b.t.setAttribute("opacity", 1);
        }, reduceMotion ? 0 : i * 40);
      });
    }
    if (reduceMotion || !("IntersectionObserver" in window)) reveal();
    else {
      var io = new IntersectionObserver(function (en) { if (en[0].isIntersecting) { reveal(); io.disconnect(); } }, { threshold: 0.15 });
      io.observe(wrap);
    }
  }
  function barLegend(root) {
    var lg = el("div", { "class": "glf-legend" }, root);
    [["GPT-Live-1", BAR_FILL.live], ["other OpenAI models", BAR_FILL.openai], ["other providers", BAR_FILL.other]].forEach(function (it) {
      el("span", { html: "<i style='background:" + it[1] + "'></i>" + it[0] }, lg);
    });
  }

  // =====================================================================
  // Figure 4: τ-Voice results
  // =====================================================================
  function figTau(root) {
    el("p", { "class": "glf-title", text: "Figure 4. τ-Voice task completion (pass^1), as reported by two sources" }, root);
    barLegend(root);
    barPanel(root, {
      title: "τ-bench leaderboard, τ³-Voice, as of Sep 11, 2026.",
      sourceText: "taubench.com ↗", sourceUrl: "https://taubench.com/leaderboard?benchmark=voice",
      items: [
        { name: "gpt-live-1 (backend: gpt-6 astra, medium)", value: 81.7, cls: "live" },
        { name: "Pine Voice Preview", value: 75.4 },
        { name: "grok-voice-think-fast-1.0", value: 67.3 },
        { name: "grok-voice-think-fast-2.0 (high)", value: 62.5 },
        { name: "qwen3.5-omni-plus-realtime", value: 53.7 },
        { name: "gemini-3.1-flash-live-preview (thinking high)", value: 43.8 },
        { name: "gpt-realtime-2 (xhigh)", value: 42.4, cls: "openai" },
        { name: "gpt-realtime-2 (minimal)", value: 38.5, cls: "openai" },
        { name: "grok-voice-fast-1.0", value: 38.3 },
        { name: "gpt-realtime-1.5", value: 35.3, cls: "openai" }
      ]
    });
    barPanel(root, {
      title: "Artificial Analysis, agentic performance (τ-Voice), average of 3 trials.",
      sourceText: "artificialanalysis.ai ↗", sourceUrl: "https://artificialanalysis.ai/speech-to-speech#agentic-performance-voice",
      items: [
        { name: "GPT-Live-1 (Astra, medium)", value: 67.9, cls: "live" },
        { name: "GPT-Live-1 (Sol, low)", value: 59.3, cls: "live" },
        { name: "Grok Voice Think Fast 2.0 High", value: 56.5 },
        { name: "Qwen Audio 3.0 Realtime Plus", value: 54.6 },
        { name: "Grok Voice Think Fast 1.0", value: 52.1 },
        { name: "GPT-Realtime-2.1 High", value: 45.7, cls: "openai" },
        { name: "GPT-Realtime-2 (High)", value: 39.8, cls: "openai" },
        { name: "GPT-Realtime-1.5", value: 38.8, cls: "openai" },
        { name: "Gemini 3.1 Flash Live High", value: 37.7 },
        { name: "Qwen Audio 3.0 Realtime Flash", value: 35.9 },
        { name: "GPT-Realtime-2 (Minimal)", value: 30.8, cls: "openai" },
        { name: "GPT-Realtime-2.1 Mini High", value: 29.4, cls: "openai" },
        { name: "Grok Voice Fast 1.0", value: 27.4 },
        { name: "Gemini 3.1 Flash Live Minimal", value: 26.2 },
        { name: "GPT-Realtime-2.1 Mini Minimal", value: 22.5, cls: "openai" },
        { name: "Higgs Realtime", value: 18.6 },
        { name: "Deepslate Opal", value: 17.5 },
        { name: "GPT Realtime Mini (Oct '25)", value: 15.1, cls: "openai" }
      ]
    });
    el("p", { "class": "glf-caption", html: "Same benchmark, three numbers for GPT-Live-1: 86.2% in OpenAI's launch post (Sierra run, custom user simulator), 81.7% on the leaderboard (GPT-4.1 user simulator), 67.9% on Artificial Analysis (a harness OpenAI says had bugs). The user simulator and the backend model are both part of the system under test." }, root);
  }

  // =====================================================================
  // Figure 5: Full-Duplex-Bench results
  // =====================================================================
  function figFdb(root) {
    el("p", { "class": "glf-title", text: "Figure 5. Full-Duplex-Bench: conversational dynamics and interactivity" }, root);
    barLegend(root);
    barPanel(root, {
      title: "Artificial Analysis, conversational dynamics (weighted average of pause handling, turn-taking, interruption handling, backchannel handling on FDB v1 + v1.5).",
      sourceText: "artificialanalysis.ai ↗", sourceUrl: "https://artificialanalysis.ai/speech-to-speech#sts-quality-index",
      items: [
        { name: "Qwen Audio 3.0 Realtime Plus", value: 98.4 },
        { name: "GPT-Live-1 (Sol, low)", value: 97.3, cls: "live" },
        { name: "Qwen Audio 3.0 Realtime Flash", value: 96.9 },
        { name: "GPT-Realtime-2 (Minimal)", value: 96.1, cls: "openai" },
        { name: "GPT-Realtime-2.1 High", value: 95.7, cls: "openai" },
        { name: "GPT-Realtime-1.5", value: 95.7, cls: "openai" },
        { name: "GPT Realtime Mini (Oct '25)", value: 95.7, cls: "openai" },
        { name: "GPT-Realtime-2 (High)", value: 95.3, cls: "openai" },
        { name: "Grok Voice Think Fast 2.0 High", value: 95.1 },
        { name: "GPT-Live-1 (Astra, medium)", value: 94.9, cls: "live" },
        { name: "Higgs Realtime", value: 92.8 },
        { name: "GPT-Realtime-2.1 Mini Minimal", value: 91.8, cls: "openai" },
        { name: "GPT-Realtime-2.1 Mini High", value: 91.7, cls: "openai" },
        { name: "PersonaPlex", value: 91.0 },
        { name: "Deepslate Opal", value: 85.7 },
        { name: "Grok Voice Think Fast 1.0", value: 77.8 },
        { name: "Gemini 3.1 Flash Live High", value: 74.3 },
        { name: "Qwen3 Omni Flash", value: 72.7 },
        { name: "Gemini 3.1 Flash Live Minimal", value: 72.3 },
        { name: "Grok Voice Fast 1.0", value: 71.6 },
        { name: "FLM-Audio", value: 62.0 },
        { name: "Moshi", value: 61.0 }
      ]
    });
    barPanel(root, {
      title: "OpenAI, Full-Duplex-Bench v1.5 interactivity (reactions to background speech, speech to another person, listener backchannels, and interruptions).",
      sourceText: "launch post ↗", sourceUrl: "https://openai.com/index/introducing-gpt-live-1-in-the-api/",
      items: [
        { name: "gpt-live-1", value: 80.1, cls: "live" },
        { name: "gpt-realtime-2", value: 47.8, cls: "openai" },
        { name: "gpt-realtime-2.1", value: 45.4, cls: "openai" }
      ]
    });
    el("p", { "class": "glf-caption", html: "Conversational dynamics is saturated: every serious model is above 90%. Interactivity separates them, and it is the metric that tests whether the model knows when speech is directed at it." }, root);
  }

  // =====================================================================
  // Sidenotes: lay guesses out in the right margin on wide screens
  // =====================================================================
  function layoutSidenotes() {
    var article = document.querySelector(".blog-post");
    if (!article) return;
    var notes = Array.prototype.slice.call(article.querySelectorAll(".sn"));
    if (!notes.length) return;
    var wide = window.matchMedia("(min-width: 1200px)").matches;
    if (!wide) {
      notes.forEach(function (n) { var b = n.querySelector(".sn-body"); b.style.top = ""; b.classList.remove("laid"); });
      return;
    }
    var aTop = article.getBoundingClientRect().top;
    var figs = Array.prototype.map.call(article.querySelectorAll(".glf, table"), function (f) {
      var r = f.getBoundingClientRect(); return { top: r.top - aTop, bottom: r.bottom - aTop };
    });
    var cursor = 0;
    notes.forEach(function (n) {
      var ref = n.querySelector(".sn-ref"), body = n.querySelector(".sn-body");
      var top = Math.max(ref.getBoundingClientRect().top - aTop - 6, cursor);
      var h = body.offsetHeight;
      figs.forEach(function (f) { if (top < f.bottom + 8 && top + h > f.top - 8) top = f.bottom + 14; });
      body.style.top = top + "px";
      body.classList.add("laid");
      cursor = top + h + 16;
    });
  }
  function initSidenotes() {
    var notes = document.querySelectorAll(".sn");
    if (!notes.length) return;
    Array.prototype.forEach.call(notes, function (n) {
      var ref = n.querySelector(".sn-ref");
      if (ref) ref.addEventListener("click", function () {
        if (window.matchMedia("(min-width: 1200px)").matches) return;
        n.classList.toggle("open");
      });
    });
    layoutSidenotes();
    var t = null;
    window.addEventListener("resize", function () { clearTimeout(t); t = setTimeout(layoutSidenotes, 100); });
    window.addEventListener("load", layoutSidenotes);
    if (document.fonts && document.fonts.ready) document.fonts.ready.then(layoutSidenotes);
    setTimeout(layoutSidenotes, 800);
  }

  // ---------- boot ----------
  function boot() {
    var map = { "glf-arch": figArch, "glf-stream": figStream, "glf-cost": figCost, "glf-tau": figTau, "glf-fdb": figFdb, "glf-evals": figEvals };
    Object.keys(map).forEach(function (id) {
      var node = document.getElementById(id);
      if (!node) return;
      try {
        var ns = node.querySelector("noscript"); if (ns) node.removeChild(ns);
        map[id](node);
      } catch (e) {
        node.textContent = "Figure failed to render: " + e.message;
        if (window.console) console.error(e);
      }
    });
    initSidenotes();
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot); else boot();
})();
