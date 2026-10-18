const $ = (s) => document.querySelector(s),
  $$ = (s) => [...document.querySelectorAll(s)];
// Video tabs activate on Enter/Space; arrow keys can browse without reloading media.
function bindTabs(list, panelFor, activate, automatic = false) {
  const buttons = [...list.querySelectorAll("button")];
  list.setAttribute("role", "tablist");
  const compact = matchMedia("(max-width: 600px)");
  const chooser = list.previousElementSibling?.querySelector("select");
  let selectedIndex = 0;
  function labelPanel() {
    const panel = panelFor(selectedIndex);
    panel.setAttribute(
      "role",
      chooser && compact.matches ? "region" : "tabpanel",
    );
    panel.setAttribute(
      "aria-labelledby",
      chooser && compact.matches ? chooser.id : buttons[selectedIndex].id,
    );
  }
  if (chooser) {
    if (compact.addEventListener)
      compact.addEventListener("change", labelPanel);
    else compact.addListener(labelPanel);
  }
  function select(index) {
    index = (index + buttons.length) % buttons.length;
    buttons.forEach((button, i) => {
      button.setAttribute("aria-selected", String(i === index));
      button.classList.toggle("active", i === index);
      button.tabIndex = i === index ? 0 : -1;
    });
    selectedIndex = index;
    labelPanel();
    activate(index);
  }
  buttons.forEach((button, i) => {
    button.id = `${list.classList[0]}-tab-${i}`;
    button.setAttribute("role", "tab");
    button.setAttribute("aria-controls", panelFor(i).id);
    const panel = panelFor(i);
    panel.setAttribute("role", "tabpanel");
    panel.tabIndex = 0;
    panel.setAttribute("aria-labelledby", button.id);
    button.onclick = () => select(i);
    button.onkeydown = (event) => {
      let next;
      if (event.key === "ArrowRight") next = (i + 1) % buttons.length;
      if (event.key === "ArrowLeft")
        next = (i + buttons.length - 1) % buttons.length;
      if (event.key === "Home") next = 0;
      if (event.key === "End") next = buttons.length - 1;
      if (next === undefined) return;
      event.preventDefault();
      buttons[next].focus({ preventScroll: true });
      if (automatic) select(next);
    };
  });
  select(0);
  return { select };
}
let hero, replay, data;
fetch("./assets/replay.json")
  .then((response) => {
    if (!response.ok) throw new Error("Replay unavailable");
    return response.json();
  })
  .then((recording) => {
    data = recording;
    $("#replay-time").max = recording.duration;
    updateReplay();
    if (replay) enableReplay();
  })
  .catch(() => {
    $("#replay-clock").textContent = "Replay unavailable";
  });
function enableReplay() {
  $("#play-replay").disabled = false;
  $("#replay-time").disabled = false;
}
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
  if (!data || !replay) return;
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
  requestAnimationFrame(animate);
}
requestAnimationFrame(animate);
const architecture = {
  tree: [
    "Behavior-tree harness",
    "The tree schedules skills, checks support conditions and waits for independent verification.",
  ],
  left: [
    "Left arm · manipulation",
    "Scene and wrist images plus measured joints feed the task arm’s FLUX instance. Each skill selects its task checkpoint.",
  ],
  right: [
    "Right arm · support",
    "The helper has a separate FLUX instance. It establishes access, then a local controller maintains the hold.",
  ],
  safety: [
    "Safety supervisor",
    "Commands pass through ownership, joint-limit, freshness, tracking and stop checks.",
  ],
  perception: [
    "Independent scene inspection",
    "Qwen updates timestamped scene predicates. The runtime uses these observations to verify progress.",
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
    "The helper opens the lid. Placement waits for verification.",
  ],
  [
    "Maintain lid support",
    "Wait for support verification",
    "true",
    "pending",
    "WHILE: establish support",
    "Verify lid support before the placement arm enters.",
  ],
  [
    "Hold position locally",
    "Pick and insert the object",
    "true",
    "pending",
    "WHILE: lid_open",
    "Place the object while the helper holds the lid. Lost support cancels placement.",
  ],
  [
    "Keep the lid open",
    "Wait for independent verification",
    "true",
    "verify",
    "UNTIL: object_inserted",
    "Wait for the scene verifier to observe the object inside.",
  ],
  [
    "Release support",
    "Placement verified",
    "release",
    "true",
    "AFTER: insertion verified",
    "Release support after insertion is verified. Repeat for the remaining objects.",
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
      "Lid support was lost. Cancel placement and recover before continuing.";
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
    if (+b.dataset.step === i) b.setAttribute("aria-current", "step");
    else b.removeAttribute("aria-current");
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
const comparisonStatus = document.createElement("p");
comparisonStatus.className = "inline-status";
comparisonStatus.setAttribute("role", "status");
comparisonStatus.hidden = true;
$(".comparison-controls").after(comparisonStatus);
[base, lora].forEach((video) =>
  ["pause", "ended", "play"].forEach((event) =>
    video.addEventListener(event, () => {
      $("#play-comparison").textContent =
        base.paused && lora.paused ? "Play both" : "Pause both";
    }),
  ),
);
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
  const outcomes = await Promise.allSettled([base.play(), lora.play()]);
  if (outcomes.some((outcome) => outcome.status === "rejected")) {
    base.pause();
    lora.pause();
    $("#play-comparison").textContent = "Play both";
    comparisonStatus.textContent =
      "Playback didn't start. Try the play button in each recording.";
    comparisonStatus.hidden = false;
  } else {
    $("#play-comparison").textContent = "Pause both";
    comparisonStatus.hidden = true;
  }
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
    "Held-out take 101: RAW-5k predicted frames beside recorded camera views.",
  ],
  [
    "real-cutter-trial.mp4",
    "Real robot · recorded cutter trial",
    "From the table toward the box.",
    "An outside view of Haller grasping and carrying the yellow cutter. The recording ends without a completed release.",
  ],
  [
    "liminal-overview.mp4",
    "Project film · 60 seconds",
    "Teaching, acting and imagining.",
    "The team’s film: VR teaching, real FLUX runs and held-out visual predictions. Speed changes and scripted replay are labeled in the film; session counts are team-reported.",
  ],
  [
    "predicted-future.mp4",
    "Visual prediction diagnostic",
    "Observed frames and an imagined future.",
    "Observed simulated frames beside FLUX’s predicted future.",
  ],
];
let slide = 0;
function showSlide(i) {
  slide = (i + slides.length) % slides.length;
  const [file, type, title, description] = slides[slide];
  const video = $("#gallery-video");
  video.poster = `assets/preview-${file.replace(".mp4", ".jpg")}`;
  if (video.getAttribute("src") !== `assets/${file}`) {
    video.pause();
    video.src = `assets/${file}`;
  }
  $("#gallery-type").textContent = type;
  $("#gallery-title").textContent = title;
  $("#gallery-description").textContent = description;
  $("#gallery-count").textContent = `${slide + 1} / ${slides.length}`;
  $("#gallery-select").value = String(slide);
}
const galleryTabs = bindTabs(
  $(".gallery-buttons"),
  () => $("#gallery-panel"),
  showSlide,
);
$("#gallery-select").onchange = (event) =>
  galleryTabs.select(+event.target.value);
$("#gallery-prev").onclick = () => galleryTabs.select(slide - 1);
$("#gallery-next").onclick = () => galleryTabs.select(slide + 1);
let results;
async function loadResults() {
  $("#result-metric").disabled = true;
  try {
    const response = await fetch("./evidence/results.json", {
      cache: "no-cache",
    });
    if (!response.ok) throw new Error("Results unavailable");
    results = await response.json();
    renderResults();
    $("#result-metric").disabled = false;
  } catch {
    const note = document.createElement("div");
    note.className = "inline-status";
    note.append("The chart couldn't load. ");
    const retry = document.createElement("button");
    retry.textContent = "Reload chart";
    retry.onclick = loadResults;
    note.append(retry);
    $("#result-bars").replaceChildren(note);
  }
}
loadResults();
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
const resultButtons = $$(".results-tabs button");
bindTabs(
  $(".results-tabs"),
  (index) => $(`#results-${resultButtons[index].dataset.result}`),
  (index) =>
    resultButtons.forEach((button, i) => {
      $(`#results-${button.dataset.result}`).hidden = i !== index;
    }),
  true,
);
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
const physicalTabs = bindTabs(
  $(".physical-tabs"),
  () => $("#physical-panel"),
  (index) => {
    const [file, caption] = physicalClips[index];
    const video = $("#physical-video");
    video.poster = `assets/preview-${file.replace(".mp4", ".jpg")}`;
    if (video.getAttribute("src") !== `assets/${file}`) {
      video.pause();
      video.src = `assets/${file}`;
    }
    $("#physical-caption").textContent = caption;
    $("#physical-select").value = String(index);
  },
);

$("#physical-select").onchange = (event) =>
  physicalTabs.select(+event.target.value);

// Core controls above are ready before any 3D module or mesh download starts.
import("./scene-preview.js?v=20261018-resilient")
  .then(({ createScenePreview }) => {
    const desktop =
      matchMedia("(min-width: 851px)").matches &&
      !navigator.connection?.saveData;
    createScenePreview($("#hero-view"), {
      auto: desktop,
      controls: [$("#reset-hero")],
      onReady(view) {
        hero = view;
        $(".hero-twin .tag").innerHTML = "<i></i> Interactive MuJoCo scene";
        $(".orbit-hint").textContent = "Drag to orbit · Scroll to zoom";
      },
      onError() {
        hero = null;
        $(".hero-twin .tag").innerHTML = "<i></i> Recorded scene preview";
        $(".orbit-hint").textContent = "Recorded scene preview";
      },
    });
    createScenePreview($("#replay-view"), {
      auto: desktop ? "visible" : false,
      controls: $$(".preset"),
      onReady(view) {
        replay = view;
        if (data) {
          updateReplay();
          enableReplay();
        }
      },
      onError() {
        replay = null;
        playing = false;
        $("#play-replay").textContent = "▶";
        $("#play-replay").setAttribute("aria-label", "Play simulation replay");
        $("#play-replay").disabled = true;
        $("#replay-time").disabled = true;
      },
    });
  })
  .catch(() => {
    $$(".scene-status").forEach((note) => {
      note.textContent =
        "The 3D preview is unavailable. Recordings and controls still work.";
    });
    $$(".scene-load").forEach((button) => {
      button.disabled = false;
      button.textContent = "Reload page";
      button.onclick = () => location.reload();
    });
  });
