"""Prompt one simulated Haller arm with FLUX 3 Action from a browser.

    MUJOCO_GL=egl ../.venv/bin/python app.py --policy http://GPU_BOX:8100  # FLUX on the GPU box
    MUJOCO_GL=egl ../.venv/bin/python app.py --policy demo                 # scripted wave
    then open http://localhost:8000

One worker thread owns MuJoCo and its renderers (EGL contexts are per thread). While idle it steps
physics in real time; during an episode the clock is sim time: each 30 Hz control tick renders the
D455 (`top`) and the arm's wrist camera, sends them with the joint state, applies the returned
command, and steps 1/30 s, however long the policy took.
"""

from __future__ import annotations

import argparse
import base64
import io
import math
import queue
import threading
import time
from pathlib import Path

import mujoco
import numpy as np
import requests
import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse, Response, StreamingResponse
from PIL import Image
from pydantic import BaseModel

HERE = Path(__file__).parent
SCENE = HERE.parent / "scene.xml"
JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
REST_DEG = (0.0, -100.0, 97.0, 70.0, 0.0, 0.0)  # folded on the plate, gripper closed
CONTROL_HZ = 30.0  # FLUX SO-101 checkpoint rate
MAX_STEP_DEG = 6.0  # per-tick clamp on arm joints = 180 deg/s; the model bounds nothing itself
# Camera renders at the real streams' aspect ratios (D455 recorded 640x480, OV2710 16:9); the policy's
# processor resizes each view to 256x256.
SCENE_WH, WRIST_WH, VIEW_WH = (640, 480), (640, 360), (800, 600)
CUBES = ("cube_red", "cube_green", "cube_amber")


def jpeg(img: np.ndarray, quality: int = 90) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


class Sim:
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path(str(SCENE))
        self.data = mujoco.MjData(self.model)
        self.renderers: dict[tuple[int, int], mujoco.Renderer] = {}
        self.substeps = round(1.0 / CONTROL_HZ / self.model.opt.timestep)
        m = self.model
        self.cube_home = {c: m.body(c).pos.copy() for c in CUBES}
        self.qadr = {(a, j): m.joint(f"{a}_{j}").qposadr[0] for a in ("left", "right") for j in JOINTS}
        self.act = {(a, j): m.actuator(f"{a}_{j}").id for a in ("left", "right") for j in JOINTS}
        self.grip_lo, self.grip_hi = m.jnt_range[m.joint("right_gripper").id]
        self.reset()

    def reset(self, jitter_m: float = 0.0, seed: int | None = None) -> None:
        mujoco.mj_resetData(self.model, self.data)
        rng = np.random.default_rng(seed)
        for arm in ("left", "right"):
            self.set_arm(arm, REST_DEG, teleport=True)
        for c in CUBES:
            adr = self.model.jnt_qposadr[self.model.body(c).jntadr[0]]
            pos = self.cube_home[c].copy()
            pos[:2] += rng.uniform(-jitter_m, jitter_m, 2)
            yaw = rng.uniform(-math.pi, math.pi) if jitter_m else 0.0
            self.data.qpos[adr:adr + 7] = [*pos, math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
        mujoco.mj_forward(self.model, self.data)

    # LeRobot SO-101 units: degrees for the arm joints (zero = mid-range, same as this MJCF),
    # 0-100 for the gripper (0 = closed).
    def get_arm(self, arm: str) -> list[float]:
        q = [float(self.data.qpos[self.qadr[arm, j]]) for j in JOINTS]
        grip = (q[5] - self.grip_lo) / (self.grip_hi - self.grip_lo) * 100.0
        return [math.degrees(v) for v in q[:5]] + [grip]

    def set_arm(self, arm: str, cmd: list[float] | tuple[float, ...], teleport: bool = False) -> None:
        rad = [math.radians(v) for v in cmd[:5]]
        rad.append(self.grip_lo + np.clip(cmd[5], 0.0, 100.0) / 100.0 * (self.grip_hi - self.grip_lo))
        for j, v in zip(JOINTS, rad, strict=True):
            self.data.ctrl[self.act[arm, j]] = v  # ctrlrange clips to the joint limits
            if teleport:
                self.data.qpos[self.qadr[arm, j]] = v

    def step(self, n: int | None = None) -> None:
        mujoco.mj_step(self.model, self.data, nstep=n or self.substeps)

    def render(self, camera: str, wh: tuple[int, int]) -> np.ndarray:
        r = self.renderers.get(wh)
        if r is None:
            r = self.renderers[wh] = mujoco.Renderer(self.model, height=wh[1], width=wh[0])
        r.update_scene(self.data, camera=camera)
        return r.render()

    def cubes(self) -> dict[str, list[float]]:
        return {c: np.round(self.data.body(c).xpos, 3).tolist() for c in CUBES}


class DemoPolicy:
    """Stand-in that waves the shoulder and opens the gripper: proves the loop without a model."""

    def reset(self, task: str) -> None:
        self.t = 0

    def act(self, task: str, state: list[float], scene: bytes, wrist: bytes) -> tuple[list[float], float | None]:
        self.t += 1
        s = self.t / CONTROL_HZ
        return [30 * math.sin(s), -60.0, 60.0, 60.0, 0.0, 50 + 50 * math.sin(2 * s)], None


class RemotePolicy:
    def __init__(self, url: str):
        self.url = url.rstrip("/")
        self.http = requests.Session()

    def reset(self, task: str) -> None:
        # The offloaded server encodes a new prompt here, which can take a while.
        self.http.post(f"{self.url}/reset", json={"task": task}, timeout=600).raise_for_status()

    def act(self, task: str, state: list[float], scene: bytes, wrist: bytes) -> tuple[list[float], float | None]:
        r = self.http.post(f"{self.url}/act", timeout=300, json={
            "task": task, "state": state,
            "scene": base64.b64encode(scene).decode(), "wrist": base64.b64encode(wrist).decode()})
        r.raise_for_status()
        body = r.json()
        if "error" in body:
            raise RuntimeError(body["error"])
        action = body.get("action")
        if (not isinstance(action, list) or len(action) != len(JOINTS)
                or any(isinstance(value, bool) or not isinstance(value, (int, float))
                       or not math.isfinite(value) for value in action)):
            raise RuntimeError("Policy returned an invalid action: expected six finite numbers")
        return action, body.get("infer_ms")


class Worker(threading.Thread):
    def __init__(self, policy):
        super().__init__(daemon=True, name="sim")
        self.policy = policy
        self.cmds: queue.Queue = queue.Queue()
        self.frames: dict[str, bytes] = {}
        self.cond = threading.Condition()
        self.status = {"state": "idle", "task": "", "arm": "right", "scene_cam": "top", "tick": 0, "sim_s": 0.0,
                       "last_infer_ms": None, "error": None, "action": None, "cubes": {}}

    def publish(self, **imgs: np.ndarray) -> None:
        enc = {k: jpeg(v, 80) for k, v in imgs.items()}
        with self.cond:
            self.frames.update(enc)
            self.cond.notify_all()

    def run(self) -> None:
        self.sim = Sim()
        episode = None
        while True:
            try:
                cmd = self.cmds.get_nowait()
            except queue.Empty:
                cmd = None
            if cmd:
                kind, arg = cmd
                if kind == "run":
                    episode = self.begin(**arg)
                elif kind == "stop":
                    episode = None
                    self.status["state"] = "idle"
                elif kind == "reset":
                    episode = None
                    self.sim.reset(**arg)
                    self.status.update(state="idle", tick=0, sim_s=0.0, action=None, error=None)
            if episode:
                episode = self.tick(episode)
            else:
                t0 = time.perf_counter()
                self.sim.step()
                st = self.status
                self.publish(view=self.sim.render("front", VIEW_WH),
                             scene=self.sim.render(st["scene_cam"], SCENE_WH),
                             wrist=self.sim.render(f"{st['arm']}_wrist", WRIST_WH))
                time.sleep(max(0.0, 1 / CONTROL_HZ - (time.perf_counter() - t0)))
            self.status["cubes"] = self.sim.cubes()

    def begin(self, task: str, arm: str, seconds: float, scene_cam: str) -> dict | None:
        self.status.update(state="running", task=task, arm=arm, scene_cam=scene_cam, tick=0, sim_s=0.0,
                           error=None, action=None)
        try:
            self.policy.reset(task)
        except Exception as e:  # noqa: BLE001 - surface any policy failure in the UI
            self.status.update(state="error", error=f"reset failed: {e}")
            return None
        return {"task": task, "arm": arm, "scene_cam": scene_cam, "ticks": int(seconds * CONTROL_HZ)}

    def tick(self, ep: dict) -> dict | None:
        t0 = time.perf_counter()
        sim, arm = self.sim, ep["arm"]
        scene = sim.render(ep["scene_cam"], SCENE_WH)
        wrist = sim.render(f"{arm}_wrist", WRIST_WH)
        state = sim.get_arm(arm)
        try:
            action, ms = self.policy.act(ep["task"], state, jpeg(scene, 90), jpeg(wrist, 90))
        except Exception as e:  # noqa: BLE001
            self.status.update(state="error", error=str(e))
            return None
        safe = [s + float(np.clip(a - s, -MAX_STEP_DEG, MAX_STEP_DEG))
                for s, a in zip(state[:5], action[:5], strict=True)]
        sim.set_arm(arm, safe + [float(action[5])])
        sim.step()
        st = self.status
        st["tick"] += 1
        st["sim_s"] = round(st["tick"] / CONTROL_HZ, 2)
        st["action"] = [round(v, 1) for v in action]
        if ms is not None:
            st["last_infer_ms"] = round(ms)
        self.publish(view=sim.render("front", VIEW_WH), scene=scene, wrist=wrist)
        time.sleep(max(0.0, 1 / CONTROL_HZ - (time.perf_counter() - t0)))
        if st["tick"] >= ep["ticks"]:
            st["state"] = "done"
            return None
        return ep


class RunReq(BaseModel):
    task: str
    arm: str = "right"
    seconds: float = 30.0
    # `top` = Haller's D455 (pose estimated, see README); `front` sees the whole bench.
    scene_cam: str = "top"


class ResetReq(BaseModel):
    jitter_m: float = 0.0
    seed: int | None = None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="demo", help="'demo' or the URL of flux/server.py, e.g. http://1.2.3.4:8100")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    worker = Worker(DemoPolicy() if args.policy == "demo" else RemotePolicy(args.policy))
    worker.start()
    app = FastAPI()

    @app.get("/")
    def index():
        return FileResponse(HERE / "static" / "index.html")

    @app.get("/stream/{name}")
    def stream(name: str):
        def gen():
            last = None
            while True:
                with worker.cond:
                    worker.cond.wait(timeout=1.0)
                    frame = worker.frames.get(name)
                if frame is not None and frame is not last:
                    last = frame
                    yield b"--f\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=f")

    @app.get("/frame/{name}.jpg")
    def frame(name: str):
        f = worker.frames.get(name)
        return Response(f, media_type="image/jpeg") if f else Response(status_code=404)

    @app.get("/status")
    def status():
        return {**worker.status, "policy": args.policy, "demo": args.policy == "demo"}

    @app.post("/preview")
    def preview(req: RunReq):
        # Idle tiles show what the model would see for the selected arm/camera.
        if req.arm in ("left", "right") and req.scene_cam in ("top", "front") and worker.status["state"] != "running":
            worker.status.update(arm=req.arm, scene_cam=req.scene_cam)
        return {"ok": True}

    @app.post("/run")
    def run(req: RunReq):
        if req.arm not in ("left", "right") or req.scene_cam not in ("top", "front") or not req.task.strip():
            return Response("need a task, arm left|right, scene_cam top|front", status_code=400)
        worker.cmds.put(("run", {"task": req.task.strip(), "arm": req.arm, "seconds": req.seconds,
                                 "scene_cam": req.scene_cam}))
        return {"ok": True}

    @app.post("/stop")
    def stop():
        worker.cmds.put(("stop", None))
        return {"ok": True}

    @app.post("/reset")
    def reset(req: ResetReq):
        worker.cmds.put(("reset", {"jitter_m": req.jitter_m, "seed": req.seed}))
        return {"ok": True}

    print(f"open http://localhost:{args.port}  (policy: {args.policy})")
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
