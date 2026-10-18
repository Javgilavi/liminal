import * as T from "three";
import { OrbitControls } from "./vendor/OrbitControls.js";
const $ = (id) => document.getElementById(id),
  views = [],
  meshCache = new Map();
let playStartedAt = 0, playOffset = 0;
let replay,
  time = 0,
  playing = false,
  state = null,
  hero,
  recorded,
  live,
  target = null;
const video = $("demo-video");
const quaternion = (a) => new T.Quaternion(a[1], a[2], a[3], a[0]);
const unpack = (base64, ArrayType) =>
  new ArrayType(Uint8Array.from(atob(base64), (c) => c.charCodeAt(0)).buffer);
class TwinView {
  constructor(element, data, dark = false) {
    this.element = element;
    this.scene = new T.Scene();
    this.scene.background = new T.Color(dark ? "#1c3022" : "#edf1e7");
    this.camera = new T.PerspectiveCamera(34, 1, 0.01, 30);
    this.camera.up.set(0, 0, 1);
    this.renderer = new T.WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 1.6));
    this.renderer.outputColorSpace = T.SRGBColorSpace;
    this.renderer.toneMapping = T.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = dark ? 1.2 : 1.05;
    this.renderer.shadowMap.enabled = !dark;
    this.renderer.shadowMap.type = T.PCFSoftShadowMap;
    element.replaceChildren(this.renderer.domElement);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = !matchMedia(
      "(prefers-reduced-motion: reduce)",
    ).matches;
    this.dirty = true;
    this.controls.addEventListener("change", () => {
      this.dirty = true;
    });
    this.controls.minDistance = 0.38;
    this.controls.maxDistance = 2.8;
    this.controls.maxPolarAngle = Math.PI / 2 + 0.04;
    this.reset();
    this.scene.add(new T.AmbientLight(0xffffff, 0.7));
    const hemi = new T.HemisphereLight(0xf3f8e9, 0x849272, 1.4);
    hemi.position.set(0, 0, 2);
    this.scene.add(hemi);
    const light = new T.DirectionalLight(0xffffff, 2.3);
    light.position.set(-0.6, -0.8, 1.8);
    light.castShadow = true;
    light.shadow.mapSize.set(1024, 1024);
    Object.assign(light.shadow.camera, {
      left: -1,
      right: 1,
      top: 1,
      bottom: -1,
      near: 0.1,
      far: 4,
    });
    light.shadow.bias = -0.0004;
    this.scene.add(light);
    const fill = new T.DirectionalLight(0xe5f1d2, 0.8);
    fill.position.set(1, 0.5, 0.7);
    this.scene.add(fill);
    this.bodies = data.poses.map(() => {
      const group = new T.Group();
      this.scene.add(group);
      return group;
    });
    for (const item of data.geometries) {
      let geometry;
      const [x, y, z] = item.size;
      if (item.type === 7) {
        if (!meshCache.has(item.mesh)) {
          const m = data.meshes[item.mesh],
            g = new T.BufferGeometry();
          g.setAttribute(
            "position",
            new T.BufferAttribute(unpack(m.vertices_b64, Float32Array), 3),
          );
          g.setIndex(
            new T.BufferAttribute(unpack(m.faces_b64, Uint32Array), 1),
          );
          g.computeVertexNormals();
          meshCache.set(item.mesh, g);
        }
        geometry = meshCache.get(item.mesh);
      } else if (item.type === 6)
        geometry = new T.BoxGeometry(2 * x, 2 * y, 2 * z);
      else if (item.type === 5) {
        geometry = new T.CylinderGeometry(x, x, 2 * y, 36);
        geometry.rotateX(Math.PI / 2);
      } else if (item.type === 2) geometry = new T.SphereGeometry(x, 24, 16);
      else if (item.type === 3) {
        geometry = new T.CapsuleGeometry(x, 2 * y, 5, 16);
        geometry.rotateX(Math.PI / 2);
      } else if (item.type === 4)
        geometry = new T.SphereGeometry(1, 24, 16).scale(x, y, z);
      else if (item.type === 0) geometry = new T.PlaneGeometry(4, 4);
      else continue;
      const color = new T.Color().setRGB(
        ...item.rgba.slice(0, 3),
        T.SRGBColorSpace,
      );
      const material = new T.MeshStandardMaterial({
        color,
        roughness: 0.72,
        metalness: 0.06,
        side: T.DoubleSide,
        transparent: item.rgba[3] < 1,
        opacity: item.rgba[3],
      });
      const mesh = new T.Mesh(geometry, material);
      mesh.position.fromArray(item.position);
      mesh.quaternion.copy(quaternion(item.quaternion));
      mesh.castShadow = item.type !== 0;
      mesh.receiveShadow = true;
      this.bodies[item.body].add(mesh);
    }
    this.setPoses(data.poses);
    const resize = () => {
      const { width, height } = element.getBoundingClientRect();
      if (!width || !height) return;
      this.camera.aspect = width / height;
      this.camera.updateProjectionMatrix();
      this.renderer.setSize(width, height, false);
      this.dirty = true;
    };
    new ResizeObserver(resize).observe(element);
    resize();
    this.visible = true;
    new IntersectionObserver(([entry]) => {
      this.visible = entry.isIntersecting;
    }).observe(element);
    views.push(this);
  }
  reset(preset = "perspective") {
    this.controls.target.set(0, -0.15, 0.13);
    this.camera.position.set(
      ...{
        perspective: [0.93, -1.13, 0.85],
        top: [0.001, -0.15, 1.6],
        front: [0, -1.45, 0.58],
      }[preset],
    );
    this.controls.update();
  }
  setPoses(poses) {
    if (poses === this.previousPoses) return;
    this.previousPoses = poses;
    this.dirty = true;
    poses.forEach((p, i) => {
      const b = this.bodies[i];
      if (b) {
        b.position.set(...p.slice(0, 3));
        b.quaternion.copy(quaternion(p.slice(3)));
      }
    });
  }
  draw() {
    if (this.visible && !document.hidden) {
      this.controls.update();
      if (this.dirty) {
        this.renderer.render(this.scene, this.camera);
        this.dirty = false;
      }
    }
  }
}
function showReplay(seconds) {
  if (!replay || !recorded) return;
  const index = Math.min(
    replay.frames.length - 1,
    Math.max(0, Math.floor(seconds * replay.fps)),
  );
  recorded.setPoses(replay.frames[index].poses);
  hero.setPoses(replay.frames[index].poses);
  $("replay-time").value = seconds;
  $("replay-clock").textContent =
    `${seconds.toFixed(1).padStart(4, "0")} / ${replay.duration.toFixed(1)} s`;
}
function animate(now) {
  if (!playing && !video.paused && !video.ended) time = video.currentTime;
  if (playing) {
    time = playOffset + Math.max(0, (now - playStartedAt) / 1000);
    if (time >= replay.duration) {
      time = 0;
      playing = false;
      $("play-replay").textContent = "▶";
    }
  }
  if (replay) showReplay(time);
  if (live && target && state?.connected && target.progress !== 1) {
    live.dirty = true;
    const t = Math.min((now - target.started) / 65, 1);
    target.progress = t;
    target.poses.forEach((p, i) => {
      const b = live.bodies[i];
      if (b) {
        b.position.lerpVectors(
          target.from[i].p,
          new T.Vector3(...p.slice(0, 3)),
          t,
        );
        b.quaternion.slerpQuaternions(
          target.from[i].q,
          quaternion(p.slice(3)),
          t,
        );
      }
    });
  }
  views.forEach((v) => v.draw());
  requestAnimationFrame(animate);
}
function updateState(next) {
  state = next;
  $("connection-dot").classList.toggle("connected", state.connected);
  $("connection-label").textContent = state.connected
    ? "Measured telemetry connected"
    : "Waiting for controller";
  $("metric-controller").textContent = state.connected
    ? "Connected"
    : "Offline";
  $("metric-rate").textContent = state.connected
    ? `${state.observed_hz.toFixed(1)} Hz`
    : "— Hz";
  $("metric-age").textContent =
    state.sample_age_ms !== null ? `${state.sample_age_ms} ms` : "— ms";
  $("metric-arm").textContent = state.active_arm
    ? `${state.active_arm} arm`
    : "No profile";
  if (state.controller) $("controller-link").href = state.controller;
  const busy = ["starting", "running"].includes(state.job.state);
  $("run-button").disabled =
    !state.connected || !state.profile_configured || busy;
  $("stop-button").disabled = !busy;
  if (busy) $("run-message").textContent = `Running: ${state.job.task}`;
  else if (state.job.error) {
    $("run-message").textContent = state.job.error;
    $("run-message").classList.add("error");
  } else if (["finished", "stopped"].includes(state.job.state))
    $("run-message").textContent =
      `Trial ${state.job.state}. Logs saved locally. Task success needs verification.`;
  if (!state.connected) {
    $("real-camera").hidden = true;
    $("camera-empty").hidden = false;
    $("camera-status").textContent = "Disconnected";
  }
  if (state.connected && live) {
    const signature = JSON.stringify(state.poses);
    if (signature !== target?.signature)
      target = {
        signature,
        poses: state.poses,
        started: performance.now(),
        from: live.bodies.map((b) => ({
          p: b.position.clone(),
          q: b.quaternion.clone(),
        })),
      };
  }
  if (Object.keys(state.arms).length) {
    $("joint-values").replaceChildren();
    const names = [
      "Shoulder pan",
      "Shoulder lift",
      "Elbow flex",
      "Wrist flex",
      "Wrist roll",
      "Gripper",
    ];
    for (const [side, arm] of Object.entries(state.arms)) {
      const col = document.createElement("div"),
        heading = document.createElement("strong");
      heading.textContent = `${side.toUpperCase()} / ${arm.mode}`;
      col.append(heading);
      arm.positions.forEach((v, i) => {
        const row = document.createElement("div");
        row.className = "joint-row";
        const label = document.createElement("span"),
          number = document.createElement("span");
        label.textContent = names[i];
        number.textContent = `${v.toFixed(1)}${i === 5 ? "%" : "°"}`;
        row.append(label, number);
        col.append(row);
      });
      $("joint-values").append(col);
    }
  }
}
function connect() {
  const socket = new WebSocket(
    `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws/twin`,
  );
  socket.onmessage = (e) => {
    try {
      updateState(JSON.parse(e.data));
    } catch (error) {
      console.error(error);
    }
  };
  socket.onclose = () => {
    if (state) updateState({ ...state, connected: false });
    setTimeout(connect, 2000);
  };
  socket.onerror = () => socket.close();
}
let loadingCamera = false;
function cameraTick() {
  if (state?.connected && !document.hidden && !loadingCamera) {
    loadingCamera = true;
    const image = $("real-camera");
    image.onload = () => {
      loadingCamera = false;
      if (!state?.connected) return;
      image.hidden = false;
      $("camera-empty").hidden = true;
      $("camera-status").textContent = "Real camera / live snapshots";
    };
    image.onerror = () => {
      loadingCamera = false;
      image.hidden = true;
      $("camera-empty").hidden = false;
      $("camera-status").textContent = "Camera unavailable";
    };
    image.src = `/api/cameras/${encodeURIComponent($("camera-select").value)}?t=${Date.now()}`;
  }
  setTimeout(cameraTick, 150);
}
$("play-replay").onclick = () => {
  playing = !playing;
  if (playing) { playOffset = time; playStartedAt = performance.now(); }
  video.pause();
  $("play-replay").textContent = playing ? "Ⅱ" : "▶";
};
$("replay-time").oninput = (e) => {
  playing = false;
  video.pause();
  time = Number(e.target.value);
  video.currentTime = time;
  $("play-replay").textContent = "▶";
  showReplay(time);
};
video.onplay = () => {
  playing = false;
  $("play-replay").textContent = "Ⅱ";
};
video.onpause = () => {
  if (!playing) $("play-replay").textContent = "▶";
};
video.onseeked = () => {
  time = video.currentTime;
  showReplay(time);
};
document.querySelectorAll(".preset").forEach(
  (button) =>
    (button.onclick = () => {
      recorded?.reset(button.dataset.camera);
      document
        .querySelectorAll(".preset")
        .forEach((b) => b.classList.toggle("active", b === button));
    }),
);
document
  .querySelectorAll(".reset-view")
  .forEach(
    (button) =>
      (button.onclick = () => ({ hero, live })[button.dataset.view]?.reset()),
  );
$("execute").onchange = () => {
  $("engage-label").hidden = !$("execute").checked;
  $("run-button").textContent = $("execute").checked
    ? "Run on robot ↗"
    : "Predict actions ↗";
};
async function command(route, body) {
  const response = await fetch(route, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const result = await response.json();
  if (!response.ok) throw new Error(result.detail || "Request failed");
  return result;
}
$("prompt-form").onsubmit = async (event) => {
  event.preventDefault();
  $("run-button").disabled = true;
  $("run-message").classList.remove("error");
  try {
    await command("/api/run", {
      prompt: $("prompt").value.trim(),
      execute: $("execute").checked,
      engage_acknowledged: $("engage").checked,
      duration_s: Number($("duration").value),
    });
    $("run-message").textContent =
      "Trial requested. Waiting for guarded runner…";
  } catch (error) {
    $("run-message").textContent = error.message;
    $("run-message").classList.add("error");
  }
};
$("stop-button").onclick = async () => {
  try {
    await command("/api/stop", {});
  } catch (error) {
    $("run-message").textContent = error.message;
  }
};
try {
  const [sceneResponse, replayResponse] = await Promise.all([
    fetch("/api/scene"),
    fetch("/assets/replay.json"),
  ]);
  if (!sceneResponse.ok || !replayResponse.ok)
    throw new Error("Scene or replay unavailable");
  const data = await sceneResponse.json();
  replay = await replayResponse.json();
  $("replay-time").max = replay.duration;
  hero = new TwinView($("hero-view"), data);
  recorded = new TwinView($("replay-view"), data);
  live = new TwinView($("live-view"), data, true);
  showReplay(0);
  requestAnimationFrame(animate);
} catch (error) {
  console.error(error);
  document.querySelectorAll(".viewer").forEach((e) => {
    e.textContent = "3D unavailable. WebGL and the MuJoCo scene are required.";
  });
}
connect();
cameraTick();
