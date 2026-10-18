import { TwinView, loadScene, drawViews } from "./viewer.js";
const $ = (s) => document.querySelector(s),
  $$ = (s) => [...document.querySelectorAll(s)];
let hero, replay, data;
loadScene()
  .then((scene) => {
    hero = new TwinView($("#hero-view"), scene);
    replay = new TwinView($("#replay-view"), scene);
    if (data) updateReplay();
  })
  .catch(() => {
    $$(".viewer").forEach(
      (e) =>
        (e.textContent =
          "The interactive scene could not load. The recorded videos remain available."),
    );
  });
fetch("./assets/replay.json")
  .then((r) => r.json())
  .then((r) => {
    data = r;
    $("#replay-time").max = r.duration;
    updateReplay();
  });
$("#reset-hero").onclick = () => hero?.reset();
$$(".preset").forEach(
  (b) =>
    (b.onclick = () => {
      $$(".preset").forEach((x) => x.classList.toggle("active", x === b));
      replay?.reset(b.dataset.camera);
    }),
);
let playing = false,
  start = 0,
  offset = 0;
const range = $("#replay-time");
function updateReplay() {
  if (!data) return;
  const t = +range.value;
  const frames = data.frames;
  const frame = frames[Math.min(frames.length - 1, Math.floor(t * data.fps))];
  replay?.setPoses(frame.poses ?? frame);
  $("#replay-clock").textContent =
    `${t.toFixed(1)} / ${data.duration.toFixed(1)} s`;
}
range.oninput = () => {
  offset = +range.value;
  start = performance.now();
  updateReplay();
};
$("#play-replay").onclick = () => {
  if (+range.value >= +range.max) range.value = 0;
  playing = !playing;
  offset = +range.value;
  start = performance.now();
  $("#play-replay").textContent = playing ? "Ⅱ" : "▶";
  $("#play-replay").setAttribute(
    "aria-label",
    playing ? "Pause simulation replay" : "Play simulation replay",
  );
};
function animate(now) {
  if (playing && data) {
    range.value = Math.min(data.duration, offset + (now - start) / 1000);
    updateReplay();
    if (+range.value >= data.duration) {
      playing = false;
      $("#play-replay").textContent = "▶";
    }
  }
  drawViews();
  requestAnimationFrame(animate);
}
requestAnimationFrame(animate);
const architecture = {
  tree: [
    "Behavior-tree harness",
    "A planner proposes task YAML. The compiler validates it. The tree enforces sequencing, support invariants, arm ownership and verified completion; a policy cannot mark its own task successful.",
  ],
  left: [
    "Left arm · manipulation",
    "The task arm receives scene and wrist observations plus its own joint state. Its six-channel SO-101 FLUX instance keeps a separate observation and command history, and loads the checkpoint bundle selected for the active object skill.",
  ],
  right: [
    "Right arm · support",
    "The helper arm has its own FLUX instance and history. A support skill establishes access, then the runtime can maintain a local holding condition while the other arm manipulates. Opening and some object-specific bundles still need staging and physical validation.",
  ],
  safety: [
    "Safety supervisor",
    "Every motor command passes through a supervisor. Arm ownership, calibrated limits, telemetry freshness, tracking and stop conditions constrain execution. The research page does not open a serial connection or take ownership of an arm.",
  ],
  perception: [
    "Independent scene inspection",
    "Qwen scene inspection produces schema-validated observations. The WorldState stores predicates with timestamps and expiry. A task advances from independent feedback rather than accepting the manipulation policy’s own success claim.",
  ],
};
$$(".arch-node").forEach(
  (b) =>
    (b.onclick = () => {
      $$(".arch-node").forEach((x) => x.classList.toggle("selected", x === b));
      [
        $("#arch-detail-label").textContent,
        $("#arch-detail-text").textContent,
      ] = architecture[b.dataset.node];
    }),
);
const stages = [
  [
    "Open the lid",
    "Wait for access",
    "pending",
    "pending",
    "AFTER: establish access",
    "The helper arm changes the scene. The placement skill waits for a verified open-lid condition.",
  ],
  [
    "Maintain lid support",
    "Wait for support verification",
    "true",
    "pending",
    "WHILE: establish support",
    "An open lid is not enough. The runtime checks that support can be maintained before allowing the task arm to enter.",
  ],
  [
    "Hold position locally",
    "Pick and insert the object",
    "true",
    "pending",
    "WHILE: lid_open",
    "The task arm runs the object skill while the helper maintains the lid. Loss of the support invariant cancels dependent placement.",
  ],
  [
    "Keep the lid open",
    "Wait for independent verification",
    "true",
    "verify",
    "UNTIL: object_inserted",
    "The policy finishing its action chunk does not prove placement. The scene verifier must observe the object inside.",
  ],
  [
    "Release support",
    "Placement verified",
    "release",
    "true",
    "AFTER: insertion verified",
    "Only after independent verification may the helper release its support. This illustration covers one insertion; the showcase repeats for three objects.",
  ],
];
function taskStage() {
  const i = +$("#task-progress").value,
    loss = $("#support-loss").checked && i > 0 && i < 4;
  let [helper, task, lid, obj, runtime, explanation] = stages[i];
  if (loss) {
    helper = "Support lost";
    task = "Cancel dependent placement";
    lid = "false";
    runtime = "WHILE violated → cancel";
    explanation =
      "The supporting condition has disappeared. Liminal cancels the dependent skill and requires recovery or a verified new plan before continuing.";
  }
  $("#helper-action").textContent = helper;
  $("#task-action").textContent = task;
  $("#predicate-lid").textContent = `lid_open: ${lid}`;
  $("#predicate-object").textContent = `object_inserted: ${obj}`;
  $("#predicate-runtime").textContent = runtime;
  $("#task-explanation").textContent = explanation;
  $("#helper-bar").style.width = `${20 + i * 20}%`;
  $("#task-bar").style.width = `${Math.max(4, (i - 1) * 33)}%`;
  $(".task-explorer").classList.toggle("interrupted", loss);
  $$(".task-stages button").forEach((b) => {
    b.classList.toggle("active", +b.dataset.step === i);
    b.setAttribute("aria-pressed", String(+b.dataset.step === i));
  });
}
$("#task-progress").oninput = taskStage;
$("#support-loss").onchange = taskStage;
$$(".task-stages button").forEach(
  (b) =>
    (b.onclick = () => {
      $("#task-progress").value = b.dataset.step;
      taskStage();
    }),
);
taskStage();
const base = $("#compare-base"),
  lora = $("#compare-lora");
$("#play-comparison").onclick = async () => {
  if (!base.paused || !lora.paused) {
    base.pause();
    lora.pause();
    $("#play-comparison").textContent = "Play both";
    return;
  }
  if (base.ended || lora.ended) {
    base.currentTime = 0;
    lora.currentTime = 0;
  }
  await Promise.allSettled([base.play(), lora.play()]);
  $("#play-comparison").textContent = "Pause both";
};
$("#comparison-time").oninput = (e) => {
  base.pause();
  lora.pause();
  $("#play-comparison").textContent = "Play both";
  [base, lora].forEach((v) => {
    if (Number.isFinite(v.duration))
      v.currentTime = +e.target.value * v.duration;
  });
};
base.ontimeupdate = () => {
  if (!base.paused)
    $("#comparison-time").value = base.currentTime / base.duration;
};
const slides = [
  [
    "real-prediction.mp4",
    "Real recorded input / visual prediction",
    "What the policy imagines next.",
    "On held-out red-pen take 101, the RAW-5k LoRA’s imagined video appears beside recorded camera frames. A predicted future is not a measured outcome.",
  ],
  [
    "flux-session.mp4",
    "Learned policy · simulated arms",
    "The inference loop, running in MuJoCo.",
    "This archived interface session uses real FLUX inference with simulated arms. It demonstrates integration and policy behavior; it is not evidence of physical dual-arm task completion.",
  ],
  [
    "perception-session.mp4",
    "Perception · recorded human demonstration",
    "Observe the scene without commanding a robot.",
    "An archived Qwen analysis session inspects a recorded demonstration and reports scene observations. No real robot commands are issued by this session.",
  ],
  [
    "predicted-future.mp4",
    "Visual prediction diagnostic",
    "Observed frames and an imagined future.",
    "A recorded action chunk shows what the camera observed beside the future predicted by FLUX. The imagined video is a model output, not a measured outcome.",
  ],
];
let slide = 0;
function showSlide(i) {
  slide = (i + slides.length) % slides.length;
  const [file, type, title, description] = slides[slide];
  $("#gallery-video").pause();
  $("#gallery-video").src = `assets/${file}`;
  $("#gallery-type").textContent = type;
  $("#gallery-title").textContent = title;
  $("#gallery-description").textContent = description;
  $("#gallery-count").textContent = `${slide + 1} / ${slides.length}`;
  $$(".gallery-buttons button").forEach((b) => {
    b.classList.toggle("active", +b.dataset.slide === slide);
    b.setAttribute("aria-pressed", String(+b.dataset.slide === slide));
  });
}
$$(".gallery-buttons button").forEach(
  (b) => (b.onclick = () => showSlide(+b.dataset.slide)),
);
$("#gallery-prev").onclick = () => showSlide(slide - 1);
$("#gallery-next").onclick = () => showSlide(slide + 1);
let results;
fetch("./evidence/results.json")
  .then((r) => r.json())
  .then((r) => {
    results = r;
    renderResults();
  });
function renderResults() {
  if (!results) return;
  const key = $("#result-metric").value,
    n = results.simulation.n;
  $("#result-bars").replaceChildren(
    ...results.simulation.models.map((m) => {
      const row = document.createElement("div");
      row.className = "result-bar";
      const name = document.createElement("span");
      name.textContent = m.name;
      const track = document.createElement("div");
      track.className = "bar-track";
      const bar = document.createElement("div");
      bar.className = "bar-fill";
      bar.style.width = `${(m[key] / n) * 100}%`;
      track.append(bar);
      const count = document.createElement("strong");
      count.textContent = `${m[key]} / ${n}`;
      row.append(name, track, count);
      return row;
    }),
  );
}
$("#result-metric").onchange = renderResults;
const tabs = $$(".results-tabs button");
function activateTab(b) {
  tabs.forEach((x) => {
    x.setAttribute("aria-selected", String(x === b));
    x.tabIndex = x === b ? 0 : -1;
    $(`#results-${x.dataset.result}`).hidden = x !== b;
  });
}
tabs.forEach((b, i) => {
  b.id = `tab-${b.dataset.result}`;
  b.setAttribute("aria-controls", `results-${b.dataset.result}`);
  const panel = $(`#results-${b.dataset.result}`);
  panel.setAttribute("role", "tabpanel");
  panel.setAttribute("aria-labelledby", b.id);
  b.onclick = () => activateTab(b);
  b.onkeydown = (e) => {
    let index;
    if (e.key === "ArrowRight") index = (i + 1) % tabs.length;
    if (e.key === "ArrowLeft") index = (i + tabs.length - 1) % tabs.length;
    if (e.key === "Home") index = 0;
    if (e.key === "End") index = tabs.length - 1;
    if (index !== undefined) {
      e.preventDefault();
      activateTab(tabs[index]);
      tabs[index].focus();
    }
  };
});
activateTab(tabs[0]);
$("#scene-divider").oninput = (e) =>
  $("#scene-comparison").style.setProperty("--split", `${e.target.value}%`);

const physicalClips = [
  [
    "real-insert-C.mp4",
    "RAW 5k adapter · layout C · real head and wrist camera recordings.",
  ],
  [
    "real-insert-A.mp4",
    "RAW 5k adapter · layout A · real head and wrist camera recordings.",
  ],
  [
    "real-insert-B.mp4",
    "RAW 5k adapter · layout B · real head and wrist camera recordings.",
  ],
  [
    "real-morning-A.mp4",
    "07:40 session · layout A · real head and wrist camera recordings.",
  ],
  [
    "real-morning-C.mp4",
    "07:40 session · layout C · real head and wrist camera recordings.",
  ],
];
$$("[data-physical]").forEach(
  (b) =>
    (b.onclick = () => {
      const [file, caption] = physicalClips[+b.dataset.physical];
      $("#physical-video").pause();
      $("#physical-video").removeAttribute("poster");
      $("#physical-video").src = `assets/${file}`;
      $("#physical-caption").textContent = caption;
      $$("[data-physical]").forEach((x) => {
        x.classList.toggle("active", x === b);
        x.setAttribute("aria-pressed", String(x === b));
      });
    }),
);
