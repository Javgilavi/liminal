"use strict";
// Operator console. Plain DOM, no build step, no network beyond this server.
const $ = (id) => document.getElementById(id);
const S = { opts: null, proposal: null, rows: [], extraction: null, runId: null, since: 0, es: null,
            poll: null, status: "idle", memTick: 0, ground: null };

function h(tag, attrs, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (k === "class") e.className = v;
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else if (v !== false && v != null) e.setAttribute(k, v);
  }
  for (const k of kids.flat()) if (k != null) e.append(k.nodeType ? k : document.createTextNode(String(k)));
  return e;
}
async function api(path, body, method) {
  const r = await fetch(path, body === undefined ? {} : {
    method: method || "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) });
  let data = null;
  try { data = await r.json(); } catch (e) { /* non-json */ }
  if (!r.ok) {
    let d = data && data.detail;
    if (Array.isArray(d)) d = d.map((x) => x.msg || JSON.stringify(x)).join("; ");
    throw new Error(d || r.statusText);
  }
  return data;
}
const fill = (sel, items) => { sel.replaceChildren(...items.map((v) => h("option", { value: v }, v))); };
const armKind = (a) => (a === "left" ? "L" : a === "right" ? "R" : "X");
const norm = (s) => (s === "completed" ? "verified" : s === "open" || s === "started" ? "pending" : s);
const badge = (a) => h("span", { class: "badge " + armKind(a) }, (a || "auto").toUpperCase());

// ---- options -------------------------------------------------------------------------------
async function loadOptions() {
  const o = (S.opts = await api("/api/options"));
  fill($("o-scenario"), o.scenarios);
  const wb = o.scenarios.find((s) => s.includes("workbench"));
  if (wb) $("o-scenario").value = wb;
  fill($("o-mode"), o.modes); fill($("o-model"), o.models);
  fill($("v-demo"), o.demos.length ? o.demos : ["(no demos in configs/demos)"]);
  $("b-video").disabled = !o.features.video;
  $("b-video").title = o.features.video ? "" : "video input not wired yet";
  if (o.banner) $("banner").textContent = o.banner;
  renderMemoryOptions(o.memory);
}
function renderMemoryOptions(mem) {
  const box = $("o-memory"); box.replaceChildren();
  S.memOpts = {};
  if (!mem) return;
  box.append(h("h2", {}, "Memory options"));
  const entries = Array.isArray(mem) ? mem.map((k) => [k, { type: "boolean" }]) : Object.entries(mem);
  for (const [k, spec] of entries) {
    const type = typeof spec === "object" && spec ? spec.type : typeof spec;
    const choices = typeof spec === "object" && spec ? spec.choices : null;
    let input;
    if (choices) input = h("select", {}, choices.map((c) => h("option", { value: c }, c)));
    else if (type === "boolean") input = h("input", { type: "checkbox" });
    else input = h("input", { type: "number", value: spec && spec.default != null ? spec.default : "" });
    input.addEventListener("change", () => {
      const v = input.type === "checkbox" ? input.checked : input.value === "" ? null : (choices ? input.value : Number(input.value));
      if (v === null || v === false) delete S.memOpts[k]; else S.memOpts[k] = v;
    });
    box.append(h("label", { class: input.type === "checkbox" ? "chk" : "" }, k.replace(/_/g, " "), input));
  }
}
function runOptions() {
  const num = (id) => ($(id).value === "" ? null : Number($(id).value));
  return { scenario: $("o-scenario").value, mode: $("o-mode").value, model: $("o-model").value,
           max_attempts: num("o-max"), interrupt_after: num("o-int"),
           fail_first: $("o-fail").checked, lie_first: $("o-lie").checked, memory: S.memOpts || {} };
}

// ---- input / grounding ----------------------------------------------------------------------
document.querySelectorAll(".tab").forEach((b) => b.addEventListener("click", () => {
  document.querySelectorAll(".tab").forEach((x) => x.classList.toggle("on", x === b));
  $("tab-text").hidden = b.dataset.tab !== "text"; $("tab-video").hidden = b.dataset.tab !== "video";
}));
function showError(el, msg) { el.hidden = !msg; el.textContent = msg || ""; }
function setPlan(res) {
  S.proposal = res.proposal; S.rows = res.rows; S.extraction = res.extraction || null;
  showError($("g-error"), ""); showError($("run-error"), "");
  renderExtraction(res); renderPlan();
}
async function guarded(btn, fn) {
  btn.disabled = true;
  try { await fn(); } catch (e) { showError($("g-error"), String(e.message || e)); }
  finally { btn.disabled = false; }
}
$("b-ground").addEventListener("click", () => guarded($("b-ground"), async () => {
  const o = runOptions();
  setPlan(await api("/api/ground", { scenario: o.scenario, goal: $("goal").value, model: o.model }));
}));
$("b-video").addEventListener("click", () => guarded($("b-video"), async () => {
  const o = runOptions(); const f = $("v-file").files[0];
  let res;
  if (f) {
    const r = await fetch(`/api/ground_video?scenario=${encodeURIComponent(o.scenario)}&model=${encodeURIComponent(o.model)}&filename=${encodeURIComponent(f.name)}`,
      { method: "POST", headers: { "content-type": "application/octet-stream" }, body: f });
    const d = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(d.detail || r.statusText);
    res = d;
  } else res = await api("/api/ground_video", { scenario: o.scenario, demo: $("v-demo").value, model: o.model });
  setPlan(res);
}));
function renderExtraction(res) {
  const ex = res.extraction;
  $("v-extract").replaceChildren(); $("v-frames").replaceChildren();
  if (!ex) return;
  const frames = ex.frames || ex.keyframes || [];
  $("v-extract").append(h("pre", {}, JSON.stringify(ex, (k, v) => (k === "frames" || k === "keyframes" ? undefined : v), 1)));
  for (const f of frames) {
    let src = typeof f === "string" ? f : f.url || f.path;
    const m = typeof src === "string" && src.match(/^demo:([\w-]+):([\w-]+)$/);
    if (m) src = `/api/demo_frames/${m[1]}/${m[2]}`;
    $("v-frames").append(h("div", { class: "thumb" }, src ? h("img", { src }) : null, typeof f === "object" ? (f.reason || f.t || f.id || "") : ""));
  }
}

// ---- plan view + editing --------------------------------------------------------------------
function rowOf(id) { return S.rows.find((r) => r.id === id); }
function stepOf(id) { return S.proposal.steps.find((s) => s.id === id); }
function removeStep(id) {
  const st = stepOf(id); if (!st) return;
  S.proposal.steps = S.proposal.steps.filter((s) => s.id !== id);
  S.rows = S.rows.filter((r) => r.id !== id);
  for (const s of S.proposal.steps) s.depends_on = s.depends_on.filter((d) => d !== id);
  for (const r of S.rows) r.depends_on = r.depends_on.filter((d) => d !== id);
  if (!S.proposal.steps.some((s) => s.effect === st.effect))
    S.proposal.obligations = S.proposal.obligations.filter((o) => o !== st.effect);
  renderPlan();
}
function moveStep(id, d) {
  const a = S.proposal.steps, i = a.findIndex((s) => s.id === id), j = i + d;
  if (j < 0 || j >= a.length) return;
  [a[i], a[j]] = [a[j], a[i]];
  const r = S.rows, ri = r.findIndex((x) => x.id === id), rj = ri + d;
  [r[ri], r[rj]] = [r[rj], r[ri]];
  renderPlan();
}
function setArm(id, arm) { stepOf(id).arm = arm; rowOf(id).arm = arm; renderPlan(); }
function renderPlan() {
  const p = S.proposal;
  const has = p && p.steps.length > 0;
  $("plan-empty").hidden = !!p; $("lanes").hidden = !p; $("steps-h").hidden = !p;
  $("b-accept").disabled = !has || S.status === "running";
  const warn = $("plan-warn"); warn.replaceChildren();
  if (!p) { $("steps").replaceChildren(); return; }
  $("plan-meta").textContent = `${p.source_type} · “${p.goal_text}”`;
  const w = [...p.unsupported.map((x) => ["Unsupported", x]), ...p.unresolved.map((x) => ["Unresolved", x])];
  for (const [k, t] of w) warn.append(h("div", { class: "callout warn" }, h("b", {}, k + ": "), t));
  if (!has) warn.append(h("div", { class: "callout warn" }, "No runnable steps were produced."));
  const lanes = { left: $("lane-left"), right: $("lane-right") }; $("lane-other").replaceChildren();
  lanes.left.replaceChildren(); lanes.right.replaceChildren();
  const other = [];
  S.rows.forEach((r, i) => {
    const mini = h("div", { class: "mini" }, h("b", {}, `${i + 1}.`), h("b", {}, r.skill), `${r.object} → ${r.destination || "·"}`,
      r.depends_on.length ? h("span", { class: "muted" }, "  ⟵ " + r.depends_on.join(",")) : null);
    (lanes[r.arm] || { append: (m) => other.push(m) }).append(mini);
  });
  if (other.length) $("lane-other").append(h("div", { class: "lane" }, h("h3", {}, h("span", { class: "badge X" }, "GROUP/AUTO")), ...other));
  const box = $("steps"); box.replaceChildren();
  const arms = ["left", "right", ...(S.opts.arms_extra || [])];
  S.rows.forEach((r, i) => {
    const sel = h("select", { title: "arm (server rejects arms the skill can't use)", onchange: (e) => setArm(r.id, e.target.value) },
      [...new Set([...arms, r.arm].filter(Boolean))].map((a) => h("option", { value: a, selected: a === r.arm }, a)));
    const conf = r.confidence;
    box.append(h("div", { class: "step " + armKind(r.arm) },
      h("div", { class: "n" }, i + 1),
      h("div", {}, badge(r.arm), " ", h("b", {}, r.skill), "  ", h("span", {}, `${r.object} → ${r.destination || "—"}`), " ", h("span", { class: "muted" }, `[${r.id}]`)),
      h("div", { class: "tools" }, sel,
        h("button", { class: "sm", title: "move up", onclick: () => moveStep(r.id, -1) }, "↑"),
        h("button", { class: "sm", title: "move down", onclick: () => moveStep(r.id, 1) }, "↓"),
        h("button", { class: "sm", title: "remove step", onclick: () => removeStep(r.id) }, "✕")),
      h("div", { class: "prompt" }, r.flux_prompt),
      h("div", { class: "meta" },
        r.depends_on.length ? r.depends_on.map((d) => h("span", { class: "tag dep" }, "after " + d)) : h("span", { class: "tag" }, "no deps"),
        r.maintain.map((m) => h("span", { class: "tag keep" }, "while " + m)),
        r.obligation ? h("span", { class: "tag" }, "goal: " + r.effect) : h("span", { class: "tag" }, r.effect),
        r.sources.map((s) => h("span", { class: "tag src", title: s.note || "" }, `${s.kind}:${s.ref}`)),
        conf != null ? h("span", {}, "conf ", h("span", { class: "conf" }, h("i", { style: `width:${Math.round(conf * 100)}%` })), ` ${conf}`) : null,
        r.note ? h("span", {}, r.note) : null)));
  });
  $("plan-oblig").replaceChildren(h("span", { class: "muted" }, "Goal obligations: "),
    ...p.obligations.map((o) => h("span", { class: "chip pending" }, o)));
}

// ---- running ---------------------------------------------------------------------------------
$("b-accept").addEventListener("click", async () => {
  showError($("run-error"), "");
  try {
    stopWatching(); $("timeline").replaceChildren(); $("attempts").replaceChildren(); S.since = 0;
    const res = await api("/api/runs", { ...runOptions(), procedure: S.proposal });
    watch(res.run_id);
  } catch (e) { showError($("run-error"), "Server rejected the plan:\n" + (e.message || e)); }
});
$("b-resume").addEventListener("click", async () => {
  try { stopWatching(); const r = await api(`/api/runs/${S.runId}/resume`, {}); S.since = 0; $("timeline").replaceChildren(); watch(r.run_id); }
  catch (e) { showError($("run-error"), String(e.message || e)); }
});
$("estop").addEventListener("click", async () => {
  try { await api("/estop", {}); } catch (e) { showError($("run-error"), "E-STOP request failed: " + e.message); }
  refreshStatus();
});
function stopWatching() { if (S.es) S.es.close(); S.es = null; clearInterval(S.poll); S.poll = null; }
function watch(runId) {
  S.runId = runId; S.status = "running"; setPill("running");
  $("scene").hidden = false; $("b-accept").disabled = true; $("b-resume").disabled = true;
  if (window.EventSource) {
    S.es = new EventSource(`/api/runs/${runId}/events?since=0&stream=1`);
    S.es.onmessage = (m) => addEvent(JSON.parse(m.data));
    S.es.addEventListener("done", () => { S.es.close(); S.es = null; });
    S.es.onerror = () => { if (S.es) { S.es.close(); S.es = null; } };   // status poll below keeps working
  }
  S.poll = setInterval(tick, 1000); tick();
}
async function tick() {
  if (!S.runId) return;
  try {
    if (!S.es) {  // polling fallback
      const d = await api(`/api/runs/${S.runId}/events?since=${S.since}`);
      d.events.forEach(addEvent);
    }
    await refreshStatus();
    $("scene").src = `/api/runs/${S.runId}/scene.png?t=${Date.now()}`;
    if (S.status !== "running" || ++S.memTick % 2 === 0) refreshMemory();
    if (S.status !== "running" && !S.es) { stopWatching(); }
  } catch (e) { /* transient */ }
}
function setPill(s) { const p = $("runstate"); p.textContent = s; p.className = "pill " + s; }
async function refreshStatus() {
  if (!S.runId) return;
  const r = await api(`/api/runs/${S.runId}`);
  S.status = r.active ? "running" : r.status; setPill(r.estopped && !r.active ? "estopped" : S.status);
  $("exec-meta").textContent = `${r.run_id} · ${r.mode}`;
  $("oblig").replaceChildren(...r.obligations.map((o) => h("span", { class: "chip " + norm(o.status), title: (o.evidence_ids || []).join(", ") }, `${o.id}: ${norm(o.status)}`)));
  $("attempts").replaceChildren(...r.attempts.map((a) => h("div", { class: "att" },
    badge(a.arm || null), h("b", {}, a.skill || ""), h("span", { class: "chip " + (["verified","completed"].includes(a.status) ? "verified" : /fail|reject/.test(a.status) ? "failed" : norm(a.status) === "pending" ? "pending" : "unknown") }, a.status),
    h("span", { class: "muted" }, (a.obligation_ids || []).join(", ")))));
  if (r.error) showError($("run-error"), r.error);
  else if (!r.active && r.report) showError($("run-error"), "");
  $("b-accept").disabled = !!r.active || !S.proposal || !S.proposal.steps.length;
  $("b-resume").disabled = !!r.active || r.mode !== "planner" || !S.runId;
}
const EV_SKIP = new Set(["t", "kind", "i"]);
function addEvent(e) {
  if (e.i < S.since) return; S.since = e.i + 1;
  const k = String(e.kind || "");
  const txt = JSON.stringify(e, (key, v) => (EV_SKIP.has(key) ? undefined : v));
  let cls = "dim";
  if (/dropped|stale/.test(k + txt)) cls = "drop";
  else if (/fail|error|reject|violat|estop|lie/.test(k) || /"(status|decision|result)":"(failed|rejected|error)"/.test(txt)) cls = "bad";
  else if (/unknown/.test(k) || /"(status|decision)":"unknown"/.test(txt)) cls = "unk";
  else if (/verif|success|complete|done/.test(k) || /"(status|decision)":"verified"/.test(txt)) cls = "ok";
  else if (/^(episode_|skill_|verification)/.test(k)) cls = "";
  const tl = $("timeline"); const stick = tl.scrollTop + tl.clientHeight >= tl.scrollHeight - 30;
  tl.append(h("div", { class: "ev " + cls }, h("span", { class: "i" }, e.i), h("div", {}, h("span", { class: "k" }, k), " ", txt.length > 2 ? txt.slice(0, 400) : "")));
  if (stick) tl.scrollTop = tl.scrollHeight;
}

// ---- memory -----------------------------------------------------------------------------------
async function refreshMemory() {
  if (!S.runId) return;
  const m = await api(`/api/runs/${S.runId}/memory`);
  const done = m.obligations.filter((o) => norm(o.status) === "verified").length;
  $("mem-summary").textContent = m.has_ledger
    ? `ledger: ${done}/${m.obligations.length} obligations verified · ${m.attempts.length} attempts · ${m.frames.length} selected keyframes`
    : "authored run: no memory ledger";
  $("mem-frames").replaceChildren(...m.frames.map((f) => h("div", { class: "thumb" },
    h("img", { src: f.url, alt: f.frame_id }), h("div", {}, f.reason || f.event || f.frame_id))));
  const extra = $("mem-extra"); extra.replaceChildren();
  if (m.inspect) extra.append(h("h3", { class: "sub" }, "Inspector"), h("pre", {}, JSON.stringify(m.inspect, null, 1)));
  if (m.inspect_error) extra.append(h("div", { class: "callout warn" }, m.inspect_error));
}

loadOptions().catch((e) => showError($("g-error"), "Cannot load options: " + e.message));
