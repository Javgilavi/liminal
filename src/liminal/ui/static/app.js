const $ = (s, root = document) => root.querySelector(s);
let latest = null;
let busy = false;
let lastFrameAt = 0;
const cards = new Map();
const cameraCards = new Map();

function updateCameras(names, labels = {}) {
  for (const [name, camera] of cameraCards) {
    if (!names.includes(name)) {
      camera.active = false; clearTimeout(camera.timer); camera.node.remove(); cameraCards.delete(name);
    }
  }
  for (const name of names) {
    if (cameraCards.has(name)) continue;
    const node = document.createElement("figure");
    const caption = document.createElement("figcaption");
    const title = document.createElement("strong"); title.textContent = labels[name] || name;
    const status = document.createElement("span"); status.textContent = "Connecting…";
    const img = document.createElement("img"); img.alt = title.textContent; img.hidden = true;
    caption.append(title, status); node.append(caption, img); $("#cameras").append(node);
    const camera = {node, active: true, timer: null}; cameraCards.set(name, camera);
    function refresh() {
      if (!camera.active) return;
      if (document.hidden) { camera.timer = setTimeout(refresh, 500); return; }
      img.src = `/cameras/${encodeURIComponent(name)}.jpg?t=${Date.now()}`;
    }
    img.onload = () => {
      if (!camera.active) return;
      img.hidden = false; status.textContent = "Live"; status.dataset.state = "live";
      camera.timer = setTimeout(refresh, 200);
    };
    img.onerror = () => {
      if (!camera.active) return;
      img.hidden = true; status.textContent = "Feed unavailable · retrying"; status.dataset.state = "error";
      camera.timer = setTimeout(refresh, 2000);
    };
    refresh();
  }
}

async function command(path, body = {}) {
  const response = await fetch(path, {
    method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body),
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.detail || response.statusText);
  return data;
}
async function act(fn) {
  if (busy) return;
  busy = true;
  try { await fn(); $("#error").textContent = ""; }
  catch (error) { $("#error").textContent = error.message; }
  finally { busy = false; }
}
// Emergency release must never wait for another UI request.
$("#estop").onclick = () => command("/estop").catch(e => { $("#error").textContent = e.message; });
$("#reset").onclick = () => act(() => command("/reset"));
$("#task-form").onsubmit = e => {
  e.preventDefault(); act(() => command("/run", {task: $("#task").value}));
};

function makeCard(name, joints) {
  const node = $("#arm-template").content.firstElementChild.cloneNode(true);
  $("h2", node).textContent = name;
  const endpoint = `/arms/${encodeURIComponent(name)}`;
  $(".engage", node).onclick = () => act(() => command(`${endpoint}/torque`, {enabled: true}));
  $(".release", node).onclick = () => act(() => command(`${endpoint}/torque`, {enabled: false}));
  joints.forEach((joint, index) => {
    const row = document.createElement("tr");
    const title = document.createElement("th"); title.textContent = joint;
    const value = document.createElement("td"); value.className = "value";
    const controls = document.createElement("td");
    [-1, 1].forEach(delta => {
      const button = document.createElement("button");
      button.textContent = delta < 0 ? "−" : "+";
      button.setAttribute("aria-label", `${name} ${joint} ${delta < 0 ? "decrease" : "increase"}`);
      button.onclick = () => act(async () => {
        if (Date.now() - lastFrameAt > 1000) throw new Error("Telemetry stale; command refused");
        const positions = [...latest.arms[name].positions];
        positions[index] += delta;
        await command(`${endpoint}/goal`, {positions});
      });
      controls.append(button);
    });
    row.append(title, value, controls); $("tbody", node).append(row);
  });
  $("#arms").append(node); cards.set(name, node);
}

async function poll() {
  try {
    const response = await fetch("/state", {cache: "no-store"});
    if (!response.ok) throw new Error("State unavailable");
    latest = await response.json(); lastFrameAt = Date.now();
    const stale = latest.sample_age_s > 0.5;
    $("#link").textContent = stale ? "Telemetry stale" : latest.estopped ? "E-STOP active" : "Live";
    for (const [name, arm] of Object.entries(latest.arms)) {
      if (!cards.has(name)) makeCard(name, latest.joints);
      const card = cards.get(name);
      const fault = arm.error || arm.health?.fault;
      const temps = Object.values(arm.health?.temperature_c || {});
      $(".health", card).textContent = fault || `${arm.health.torque ? "Torque ON" : "Torque OFF"}` +
        (temps.length ? ` · max ${Math.max(...temps)} °C` : " · simulated");
      $(".owner", card).textContent = `Owner: ${latest.owners[name] || "operator"}`;
      $(".engage", card).disabled = stale || latest.estopped || latest.running || !!fault || arm.health?.torque;
      $(".release", card).disabled = latest.running || !!fault;
      $("tbody", card).querySelectorAll("tr").forEach((row, i) => {
        $(".value", row).textContent = arm.positions ? arm.positions[i].toFixed(1) : "—";
        row.querySelectorAll("button").forEach(button => {
          button.disabled = stale || latest.estopped || latest.running || !!fault || !arm.health?.torque;
        });
      });
    }
    $("#run").disabled = stale || latest.estopped || latest.running;
    $("#report").textContent = JSON.stringify(latest.last_report, null, 2);
    updateCameras(latest.cameras, latest.camera_labels);
  } catch (error) {
    $("#link").textContent = "Disconnected · controls disabled";
    for (const card of cards.values()) card.querySelectorAll("button").forEach(b => { b.disabled = true; });
    $("#run").disabled = true;
  }
  setTimeout(poll, 200);
}
fetch("/tasks").then(r => r.json()).then(data => {
  for (const task of data.tasks) { const option = document.createElement("option"); option.textContent = task; $("#task").append(option); }
}).catch(e => { $("#error").textContent = e.message; });
poll();
