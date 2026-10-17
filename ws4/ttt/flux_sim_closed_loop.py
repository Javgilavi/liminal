"""Closed loop in sim: FLUX 3 Action drives the simulated XLeRobot arm while the P2 progress monitor watches.

Everything else in ws4/ttt used the scripted IK stand-ins to move the arm; here the policy server's chunks are
executed. The pieces are the shipped ones, used read-only: ``flux.dual_arm_runner.DualArmRunner`` with one
per-arm target per backend (``per_arm_target``: head camera as ``scene``, the arm's wrist camera), the default
``SafetyGuard``, ``flux.remote.RemoteBackend`` over the SSH tunnel, and the orchestrator's pack-box scene
(``orchestrator.sim_backend.PackBoxSim``) with its ground-truth predicates. Inference is synchronous: the arm
holds still while a chunk is computed, as in LeRobot's runner, and no sim time passes meanwhile.

Per plan step (skills and exact FLUX prompts from ``orchestrator/skills_sim_pack_box.yaml``):

1. FLUX attempt: the target starts with the skill's prompt; before every re-plan the scene's ground-truth check
   (``PackBoxSim.predicate``) runs; the attempt ends on success or after ``timeout_s`` of executed motion.
2. Repair (default on): if FLUX failed, the scripted stand-in runs the same step, so the next step starts from its
   intended precondition; if that fails too, the objects are placed by hand (logged). Repairs never count as
   FLUX successes.

At every re-plan (FLUX or scripted) the progress monitor (``progress_api.ProgressMonitor``, default the
frame-feature GRU) gets one update with the raw renders and state of that tick, like the orchestrator would.

Output per episode, in the ``sim_episodes.py`` layout (``episode.json``, ``ticks.npz``, ``frames.npz``), so
``features.frame_features``, ``claude_stage_labels.py`` and ``train_progress.py`` apply directly. ``episode.json``
``extra`` adds the per-re-plan backend, latency and monitor estimate, and per-step results and diagnostics;
``ticks.npz`` adds the TCP and object positions; ``overlay.mp4`` shows head | wrist with the monitor overlay.

    # dry runs on the laptop (no GPU): the stand-in as the "policy", or a policy that never moves
    MUJOCO_GL=egl python ws4/ttt/flux_sim_closed_loop.py --backend scripted --episodes 2
    MUJOCO_GL=egl python ws4/ttt/flux_sim_closed_loop.py --backend fake --episodes 1 --max-chunks 3
    # the real thing, through ssh -N -L 8765:127.0.0.1:8765 <pod>; token in $FLUX_TOKEN (never on argv)
    MUJOCO_GL=egl python ws4/ttt/flux_sim_closed_loop.py --backend remote --package so101 --episodes 6
"""

from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ttt.labels import EpisodeLabels, Segment, replan_targets  # noqa: E402

WS4 = Path(__file__).resolve().parents[1]
CAMERAS = ("head", "left_wrist", "right_wrist")
N_OBS_STEPS = 8
EXECUTE = 32  # commands run per chunk (SO-101 contract n_action_steps)
DEFAULT_PLAN = "open_lid:left,close_lid:left"
DEFAULT_MONITOR = WS4 / "ttt" / "models" / "progress_frame_gru_s0.pt"
LID_KNOB_Z = 0.022
SETTLE_TICKS = 10  # hold after a step before the final check, so a released object can come to rest


def _imports():
    if str(WS4) not in sys.path:
        sys.path.insert(0, str(WS4))
    import orchestrator  # noqa: F401  (adds the repo root for ws4.sim imports)
    from flux import dual_arm_runner, safety
    from flux.contract import ARMS, OBS_STATE, SO101_CONTRACT, arm_slice
    from orchestrator import sim_backend
    from orchestrator.skills import SIM_VOCABULARY, load_vocabulary

    def load_sim_vocabulary():  # the pack-box skills (skills.yaml holds the real robot's demo task)
        return load_vocabulary(SIM_VOCABULARY)

    return dual_arm_runner, safety, ARMS, OBS_STATE, SO101_CONTRACT, arm_slice, sim_backend, load_sim_vocabulary


def parse_plan(text: str) -> list[tuple[str, str]]:
    """``"open_lid:left,close_lid:left"`` -> ``[("open_lid", "left"), ("close_lid", "left")]``."""
    out = []
    for item in text.split(","):
        skill, _, arm = item.strip().partition(":")
        out.append((skill, arm or "left"))
    return out


def _jpeg(img: np.ndarray, quality: int = 90) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def _target_xyz(skill, poses: dict) -> np.ndarray | None:
    """Where the arm has to go for this skill: the lid knob, or the object to pick."""
    spec = skill.sim or {}
    if spec.get("action") in ("place_lid", "remove_lid"):
        return poses["lid"][:3] + np.array([0.0, 0.0, LID_KNOB_Z])
    if spec.get("action") == "pick_place":
        return poses[spec["object"]][:3].copy()
    return None


def _target_object(skill) -> str | None:
    spec = skill.sim or {}
    return "lid" if spec.get("action") in ("place_lid", "remove_lid") else spec.get("object")


class _Recorder:
    """``RobotInterface`` wrapper: logs every tick (state, command, stage, TCP and object positions) and hands
    the observation of each tick to ``on_tick`` before the command is applied.

    With a sim built with ``render=False`` it renders the cameras itself, only where they are used: in sync mode a
    target re-plans exactly every 32 ticks after its start, and FLUX encodes only window ticks 0 and 7 (this tick
    and 7 ticks back), so renders at offsets 0 and 25 mod 32 from ``anchor`` give the policy and the monitor the
    same images as rendering every tick. The other ticks repeat the last render, which nothing reads.
    """

    def __init__(self, robot, obs_state_key: str, sim=None, render_all: bool = True):
        self.robot = robot
        self.key = obs_state_key
        self.sim = sim
        self.render_all = render_all
        self.anchor: int | None = None  # tick at which the active target started
        self.rendered: set[int] = set()
        self._cache_key = None
        self._cache_obs: dict = {}
        self._images: dict | None = None
        self.states, self.commands, self.stages, self.tcp, self.objects = [], [], [], [], []
        self.stage = -1
        self.on_tick = None
        self.recent: deque = deque(maxlen=N_OBS_STEPS)  # (tick, obs) of the last ticks, for the oldest snapshot

    @property
    def tick(self) -> int:
        return len(self.states)

    def _render_due(self, tick: int) -> bool:
        if self.render_all or self._images is None:
            return True
        return self.anchor is not None and (tick - self.anchor) % EXECUTE in (0, EXECUTE - (N_OBS_STEPS - 1))

    def get_observation(self, force_render: bool = False) -> dict:
        obs = self.robot.get_observation()
        if self.sim is None or "observation.images.head" in obs:
            return obs
        key = (self.tick, id(obs))
        if self._cache_key != key or (force_render and self.tick not in self.rendered):
            if force_render or self._render_due(self.tick):
                with self.sim.lock:
                    self._images = {f"observation.images.{c}": self.sim.env.render(c) for c in CAMERAS}
                self.rendered.add(self.tick)
            self._cache_obs = {**obs, **self._images}
            self._cache_key = key
        return self._cache_obs

    def send_action(self, action: np.ndarray) -> None:
        obs = self.get_observation()
        priv = obs["privileged"]
        if self.on_tick is not None:
            self.on_tick(self.tick, obs, np.asarray(action, dtype=np.float64))
        self.recent.append((self.tick, obs))
        self.states.append(np.asarray(obs[self.key], dtype=np.float32))
        self.commands.append(np.asarray(action, dtype=np.float32))
        self.stages.append(self.stage)
        self.tcp.append(np.stack([priv["ee_poses"][s][:3] for s in ("left", "right")]).astype(np.float32))
        self.objects.append(np.stack([priv["object_poses"][n][:3] for n in ("lid", "apple", "yellow_block")]))
        self.robot.send_action(action)

    def obs_at(self, tick: int) -> dict | None:
        for t, obs in self.recent:
            if t == tick:
                return obs
        return None


class _Video:
    """head | arm wrist, plus a text band: what runs, the ground truth, the monitor's estimate. ffmpeg pipe."""

    BAND = 110

    def __init__(self, path: Path, hw: tuple[int, int], fps: float = 30.0, scale: int = 2):
        from PIL import ImageFont

        self.h, self.w = hw[0] * scale, hw[1] * scale
        self.size = (2 * self.w, self.h + self.BAND)
        font = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        try:
            self.font = ImageFont.truetype(font, 17)
            self.bold = ImageFont.truetype(font.replace("Sans.ttf", "Sans-Bold.ttf"), 17)
        except OSError:
            self.font = self.bold = ImageFont.load_default()
        self.proc = subprocess.Popen(
            ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s",
             f"{self.size[0]}x{self.size[1]}", "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "ultrafast",
             "-crf", "23", "-pix_fmt", "yuv420p", "-threads", "2", str(path)],
            stdin=subprocess.PIPE,
        )  # fmt: skip

    def frame(self, head: np.ndarray, wrist: np.ndarray, lines: list[tuple[str, tuple[int, int, int], bool]],
              bar: list[tuple[str, int]]) -> None:
        from PIL import Image, ImageDraw

        canvas = Image.new("RGB", self.size, (22, 22, 26))
        canvas.paste(Image.fromarray(head).resize((self.w, self.h), Image.Resampling.BILINEAR), (0, 0))
        canvas.paste(Image.fromarray(wrist).resize((self.w, self.h), Image.Resampling.BILINEAR), (self.w, 0))
        draw = ImageDraw.Draw(canvas)
        y = self.h + 4
        for text, colour, bold in lines:
            draw.text((6, y), text, fill=colour, font=self.bold if bold else self.font)
            y += 22
        # One box per plan step: outline = ground truth, fill = the monitor's stage.
        n = max(1, len(bar))
        bw = (self.size[0] - 12) // n
        for i, (name, flags) in enumerate(bar):
            x0, y0 = 6 + i * bw, self.size[1] - 26
            fill = (60, 120, 200) if flags & 2 else (45, 45, 52)
            outline = (240, 240, 240) if flags & 1 else (90, 90, 96)
            draw.rectangle([x0, y0, x0 + bw - 4, y0 + 22], fill=fill, outline=outline, width=3 if flags & 1 else 1)
            draw.text((x0 + 4, y0 + 1), name, fill=(235, 235, 235), font=self.font)
        self.proc.stdin.write(np.asarray(canvas, dtype=np.uint8).tobytes())

    def close(self) -> None:
        self.proc.stdin.close()
        self.proc.wait()


def _place_object(sim, name: str, xyz, yaw: float = 0.0) -> None:
    """Last-resort repair: put an object where the step should have left it."""
    import mujoco

    with sim.lock:
        m, d = sim.env.model, sim.env.data
        adr = m.jnt_qposadr[m.joint(f"{name}_joint").id]
        dof = m.jnt_dofadr[m.joint(f"{name}_joint").id]
        d.qpos[adr : adr + 3] = xyz
        d.qpos[adr + 3 : adr + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        d.qvel[dof : dof + 6] = 0
        mujoco.mj_forward(m, d)


def _place_post_condition(sim, sb, skill, arm: str) -> None:
    action = (skill.sim or {}).get("action")
    if action == "remove_lid":
        x, y = sb.LID_PARK_XY[arm]
        _place_object(sim, "lid", [x, y, 0.004])
    elif action == "place_lid":
        _place_object(sim, "lid", [sb.BOX_XY[0], sb.BOX_XY[1], sb.BOX_RIM_Z + 0.0045])
    elif action == "pick_place":
        _place_object(sim, skill.sim["object"], [sb.BOX_XY[0], sb.BOX_XY[1] + sb.PLACE_DY[arm], 0.03])


def run_episode(args, sim, robot, policy, vocab, plan, seed: int, out_dir: Path, monitor=None, video: bool = True):
    dual_arm_runner, safety, ARMS, OBS_STATE, SO101_CONTRACT, arm_slice, sb, _ = _imports()
    lid_on_box = plan[0][0] == "open_lid"
    robot.reset(seed=seed, lid_on_box=lid_on_box)
    hw = (sim.env.height, sim.env.width)
    rng = np.random.default_rng(seed)
    rec = _Recorder(robot, OBS_STATE, sim, render_all=video or getattr(args, "render_every_tick", True))
    scripted = {arm: sb.ScriptedSimBackend(sim) for arm in ARMS}
    backends = {"policy": policy, **{f"sim_{arm}": scripted[arm] for arm in ARMS}}
    targets = []
    for arm in ARMS:
        targets.append(_replace_name(dual_arm_runner.per_arm_target(arm, "policy"), f"{arm}:policy"))
        targets.append(_replace_name(dual_arm_runner.per_arm_target(arm, f"sim_{arm}"), f"{arm}:sim"))
    runner = dual_arm_runner.DualArmRunner(rec, targets, backends, safety.SafetyGuard(), async_inference=False)

    steps = [{"id": f"s{i + 1}", "skill": s, "arm": a} for i, (s, a) in enumerate(plan)]
    plan_skills = [s for s, _ in plan]
    if monitor is not None:
        monitor.reset(plan_skills)
    segments: list[Segment] = []
    replans: list[dict] = []
    results: list[dict] = []
    frame_jpeg: dict[int, dict[str, bytes]] = {}
    live = {"step": None, "policy": "", "est": None, "check": "", "gt": -1}
    out_dir.mkdir(parents=True, exist_ok=True)
    vid = _Video(out_dir / "overlay.mp4", hw) if video else None

    def snap(tick: int, ob: dict) -> None:
        if tick not in frame_jpeg:
            frame_jpeg[tick] = {cam: _jpeg(ob[f"observation.images.{cam}"]) for cam in CAMERAS}

    def on_tick(tick: int, ob: dict, command: np.ndarray) -> None:
        if vid is None:
            return
        st = live["step"]
        arm = st["arm"] if st else "left"
        lines = [(f"t {tick / 30:5.1f} s   " + (f"step {st['id']} {st['skill']} ({arm} arm)   policy: {live['policy']}"
                  if st else "between steps (holding)"), (235, 235, 235), True)]  # fmt: skip
        lines.append((f"ground truth {live['check']}", (200, 200, 200), False))
        est = live["est"]
        if est is not None:
            ok = est.stage == live["gt"]
            lines.append((f"monitor (GRU, frame features): stage {est.stage + 1} {est.stage_name}  "
                          f"p={est.confidence:.2f}  done={est.done:.2f}  progress={est.progress:.2f}",
                          (120, 220, 120) if ok else (240, 110, 110), False))  # fmt: skip
        bar = [(f"{i + 1} {s}", (1 if i == live["gt"] else 0) | (2 if est is not None and est.stage == i else 0))
               for i, s in enumerate(plan_skills)]
        vid.frame(ob["observation.images.head"], ob[f"observation.images.{arm}_wrist"], lines, bar)

    rec.on_tick = on_tick

    def hold(n: int) -> None:
        for _ in range(n):
            runner.step()

    def check(skill, arm) -> tuple[bool, str]:
        return sim.predicate(skill.sim_check, robot.get_observation(), arm) if skill.sim_check else (True, "")

    def run_target(target: str, arm: str, skill, kind: str, k: int, max_chunks: int, stop_on_success: bool):
        """Run one attempt; returns (success, chunks, latencies, guard flags per tick, start_tick)."""
        start = rec.tick
        stand_in = kind == "scripted" or args.backend == "scripted"
        if stand_in:  # plans with IK and may raise SimSkillError: before the target claims the arm
            backends[runner._targets[target].config.backend].begin(target, arm, skill)
        runner.start(target, skill.prompt if kind == "flux" else f"scripted {skill.name}")
        rec.anchor = rec.tick
        live["policy"] = {"flux": f"FLUX ({args.package})", "scripted": "scripted stand-in (repair)"}[kind]
        chunks, lat, guards = 0, [], []
        ok, detail = check(skill, arm)
        live["check"] = f"{_check_name(skill)} {'PASSED' if ok else 'not met'}  ({detail})"
        while True:
            if not runner.is_busy(target):  # the next tick re-plans: the chunk boundary
                ok, detail = check(skill, arm)
                live["check"] = f"{_check_name(skill)} {'PASSED' if ok else 'not met'}  ({detail})"
                if chunks >= max_chunks:
                    break
                # FLUX never stops by itself: the step ends on a passed check, as in the orchestrator. The
                # stand-in runs its whole motion (release, clear, home), as in the training episodes.
                early = stop_on_success and (not stand_in or getattr(args, "stand_in_stops_on_check", False))
                if early and ok and chunks > 0:
                    break
                if stand_in and chunks > 0:
                    be = backends[runner._targets[target].config.backend]
                    if be.finished(target, np.asarray(robot.get_observation()[OBS_STATE])[arm_slice(arm)]):
                        break
                t = rec.tick
                ob = rec.get_observation(force_render=True)
                oldest = max(start, t - (N_OBS_STEPS - 1))
                if rec.sim is not None and oldest not in rec.rendered and not rec.render_all:
                    raise RuntimeError(f"tick {oldest} was not rendered: re-plans off the 32-tick grid")
                snap(t, ob)
                old_ob = ob if oldest == t else rec.obs_at(oldest)
                if old_ob is not None:
                    snap(oldest, old_ob)
                est = None
                if monitor is not None:
                    est = monitor.update(
                        frames={cam: ob[f"observation.images.{cam}"] for cam in CAMERAS},
                        state=ob[OBS_STATE],
                        prev_head=None if old_ob is None else old_ob["observation.images.head"],
                    )
                    live["est"] = est
                report = runner.step()
                guards.append(bool(report.guard_intervened))
                ts = runner._targets[target]
                latency = ts.last_response.latency_s if ts.last_response is not None else None
                lat.append(latency)
                replans.append({
                    "tick": t, "arm": arm, "step_id": steps[k]["id"], "prompt": skill.prompt,
                    "snapshot_ticks": [oldest, t], "backend": kind, "latency_s": latency,
                    "estimate": None if est is None else {
                        "stage": est.stage, "stage_name": est.stage_name, "confidence": est.confidence,
                        "done": est.done, "progress": est.progress, "stage_progress": est.stage_progress,
                        "occurrence": est.occurrence,
                    },
                })  # fmt: skip
                chunks += 1
            else:
                guards.append(bool(runner.step().guard_intervened))
        runner.stop(target)
        rec.anchor = None
        if stand_in:
            backends[runner._targets[target].config.backend].end(target)
        return ok, chunks, lat, guards, start

    for k, st in enumerate(steps):
        skill, arm = vocab[st["skill"]], st["arm"]
        rec.stage = -1
        live.update(step=None, gt=-1)
        hold(int(rng.integers(0, 16)))  # hesitation before a step starts, as in sim_episodes.py
        pre_ok, pre_detail = check(skill, arm)
        poses0 = sim.poses()
        obj = _target_object(skill)
        rec.stage = k
        live.update(step=st, gt=k)
        max_chunks = args.max_chunks or int(np.ceil(skill.timeout_s * 30.0 / 32.0))
        try:
            ok, chunks, lat, guards, start = run_target(f"{arm}:policy", arm, skill, "flux", k, max_chunks, True)
        except sb.SimSkillError:  # --backend scripted: the stand-in cannot plan this step from here (IK)
            ok, chunks, lat, guards, start = False, 0, [], [], rec.tick
        flux_end = rec.tick
        hold(SETTLE_TICKS)
        ok_settled, detail = check(skill, arm)
        segments.append(Segment(st["id"], st["skill"], arm, start, rec.tick, bool(ok), 1))
        sl = slice(start, flux_end)
        diag = {}
        if flux_end > start:
            tcp = np.asarray(rec.tcp[sl])[:, 0 if arm == "left" else 1]
            objs = np.asarray(rec.objects[sl])
            cmd = np.asarray(rec.commands[sl])[:, arm_slice(arm)]
            names = ("lid", "apple", "yellow_block")
            # Target point per tick (the lid knob moves with the lid).
            tgt = objs[:, names.index(obj)] + (np.array([0, 0, LID_KNOB_Z]) if obj == "lid" else 0.0)
            dist = np.linalg.norm(tcp - tgt, axis=1)
            obj_start = poses0[obj][:3]
            diag = {
                "tcp_start_to_target_cm": round(float(dist[0] * 100), 1),
                "tcp_min_to_target_cm": round(float(dist.min() * 100), 1),
                "object_moved_max_cm": round(float(np.linalg.norm(objs[:, names.index(obj)] - obj_start,
                                                                  axis=1).max() * 100), 1),
                "joint_travel_deg": [round(float(x), 1) for x in np.abs(np.diff(cmd[:, :5], axis=0)).sum(0)],
                "command_min": [round(float(x), 1) for x in cmd.min(0)],
                "command_max": [round(float(x), 1) for x in cmd.max(0)],
                "gripper_closed_near_target": bool(((cmd[:, 5] < 15) & (dist < 0.03)).any()),
                "tcp_min_z_cm": round(float(tcp[:, 2].min() * 100), 1),
            }  # fmt: skip
        result = {
            "step_id": st["id"], "skill": st["skill"], "arm": arm, "prompt": skill.prompt,
            "precondition_ok": not pre_ok, "pre_check": pre_detail,
            "flux_success": bool(ok), "flux_success_after_settle": bool(ok_settled), "check": detail,
            "chunks": chunks, "ticks": flux_end - start,
            "latency_s_median": float(np.median([x for x in lat if x is not None])) if any(lat) else None,
            "guard_intervened_frac": round(float(np.mean(guards)), 3) if guards else 0.0,
            "diagnostics": diag, "repair": None,
        }  # fmt: skip
        if not ok_settled and args.repair:
            rep = {"scripted": None, "placed": False}
            try:
                r_ok, r_chunks, _, _, r_start = run_target(f"{arm}:sim", arm, skill, "scripted", k, 25, False)
                hold(SETTLE_TICKS)
                r_ok, r_detail = check(skill, arm)
                segments.append(Segment(st["id"], st["skill"], arm, r_start, rec.tick, bool(r_ok), 2))
                rep["scripted"] = {"success": bool(r_ok), "chunks": r_chunks, "check": r_detail}
            except sb.SimSkillError as e:
                rep["scripted"] = {"success": False, "error": str(e)}
                r_ok = False
            if not r_ok:
                rec.stage = -1  # no segment covers the placing ticks
                _place_post_condition(sim, sb, skill, arm)
                robot._obs = sim.env.get_obs()
                hold(SETTLE_TICKS)
                rep["placed"] = True
                rep["after_place"] = check(skill, arm)[1]
            result["repair"] = rep
        results.append(result)
        print(json.dumps({"episode": out_dir.name, **{k2: result[k2] for k2 in
                          ("step_id", "skill", "flux_success", "chunks", "latency_s_median")},
                          "diag": diag, "repaired": result["repair"] is not None}), flush=True)  # fmt: skip
    rec.stage = -1
    live.update(step=None, gt=-1)
    hold(8)  # a short tail, as in sim_episodes.py
    if vid is not None:
        vid.close()
    runner.close()

    n_ticks = rec.tick
    labels = EpisodeLabels(
        episode_id=out_dir.name,
        source="sim_gt",
        fps=30.0,
        n_ticks=n_ticks,
        task="flux_" + "+".join(plan_skills),
        steps=steps,
        segments=segments,
        replan_ticks=[r["tick"] for r in replans],
        extra={
            "seed": seed,
            "layout": sim.layout,
            "replans": replans,
            "results": results,
            "policy": {"backend": args.backend, "package": args.package, "url": args.url if args.backend == "remote"
                       else None},
            "monitor": str(args.monitor) if monitor is not None else None,
            "generator": "ws4/ttt/flux_sim_closed_loop.py",
        },
    )  # fmt: skip
    np.savez_compressed(
        out_dir / "ticks.npz",
        state=np.asarray(rec.states, np.float32),
        command=np.asarray(rec.commands, np.float32),
        stage=np.asarray(rec.stages, np.int64),
        tcp=np.asarray(rec.tcp, np.float32),
        objects=np.asarray(rec.objects, np.float32),
        object_names=np.array(["lid", "apple", "yellow_block"]),
    )
    ticks_sorted = sorted(frame_jpeg)
    frames = {"frame_ticks": np.asarray(ticks_sorted, np.int64)}
    for cam in CAMERAS:
        frames[cam] = np.array([np.frombuffer(frame_jpeg[t][cam], np.uint8) for t in ticks_sorted], dtype=object)
    np.savez(out_dir / "frames.npz", **frames)
    labels.save(out_dir / "episode.json")  # written last: a directory with episode.json is complete
    return labels, results


def _replace_name(cfg, name: str):
    import dataclasses

    return dataclasses.replace(cfg, name=name)


def _check_name(skill) -> str:
    return "check " + ", ".join(k for k in (skill.sim_check or {}))


def monitor_scores(labels: EpisodeLabels) -> dict:
    """Stage accuracy of the live estimates against ground truth, over re-plans inside a step."""
    tg = replan_targets(labels)
    reps = labels.extra["replans"]
    est = np.array([r["estimate"]["stage"] if r.get("estimate") else -2 for r in reps])
    prog = np.array([r["estimate"]["progress"] if r.get("estimate") else np.nan for r in reps])
    kind = np.array([r["backend"] for r in reps])
    mask = tg.stage >= 0
    out = {"replans": int(mask.sum())}
    for name, m in (("all", mask), ("flux", mask & (kind == "flux")), ("scripted", mask & (kind == "scripted"))):
        out[f"stage_acc_{name}"] = float((est[m] == tg.stage[m]).mean()) if m.any() else None
        out[f"n_{name}"] = int(m.sum())
    out["progress_mae"] = float(np.nanmean(np.abs(prog[mask] - tg.progress[mask]))) if mask.any() else None
    return out


def main(argv: list[str] | None = None) -> None:
    from ttt.throttle import throttle

    throttle()  # the laptop also runs 30 Hz robot loops
    os.environ.setdefault("MUJOCO_GL", "egl")
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--backend", choices=("remote", "scripted", "fake"), default="remote")
    ap.add_argument("--url", default="http://127.0.0.1:8765")
    ap.add_argument("--package", default="so101")
    ap.add_argument("--token-file", default=None, help="else $FLUX_TOKEN; never pass the token on argv")
    ap.add_argument("--plan", default=DEFAULT_PLAN, help="skill:arm,... from orchestrator/skills_sim_pack_box.yaml")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-chunks", type=int, default=0, help="per FLUX attempt; 0 = the skill's timeout_s")
    ap.add_argument("--no-repair", dest="repair", action="store_false")
    ap.add_argument("--stand-in-stops-on-check", action="store_true",
                    help="--backend scripted: end steps on the passed check like FLUX (monitor baseline)")
    ap.add_argument("--monitor", type=Path, default=DEFAULT_MONITOR)
    ap.add_argument("--image-size", default="240x320", help="render size (the monitor was trained on 240x320)")
    ap.add_argument("--no-video", dest="video", action="store_false")
    ap.add_argument("--video-episodes", type=int, default=1, help="only the first N episodes get overlay.mp4 (the "
                    "encoder is the slowest part on a loaded laptop)")
    ap.add_argument("--render-every-tick", action="store_true", help="render all cameras on every tick "
                    "(default: only the ticks FLUX and the monitor read, same images, about 2x faster)")
    ap.add_argument("--start", type=int, default=0, help="index of the first episode (seed = seed * 100000 + index)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    dual_arm_runner, safety, ARMS, OBS_STATE, SO101_CONTRACT, arm_slice, sb, load_vocabulary = _imports()
    vocab = load_vocabulary()
    plan = parse_plan(args.plan)
    for skill, arm in plan:
        if skill not in vocab.skills or not vocab[skill].is_flux or arm not in vocab[skill].arms:
            raise SystemExit(f"{skill}:{arm} is not a FLUX skill on that arm in the sim vocabulary")
    h, w = (int(x) for x in args.image_size.lower().split("x"))
    sim = sb.PackBoxSim(image_size=(h, w), randomize=True, render=args.render_every_tick, cameras=CAMERAS)
    robot = sb.SimRobot(sim, seed=args.seed)
    if args.backend == "remote":
        from flux.remote import RemoteBackend

        token = Path(args.token_file).read_text().strip() if args.token_file else os.environ.get("FLUX_TOKEN")
        policy = RemoteBackend(args.url, args.package, token=token)
    elif args.backend == "scripted":
        policy = sb.ScriptedSimBackend(sim)
        args.package = "scripted"
    else:
        from flux.backends import FakeBackend

        policy = FakeBackend(SO101_CONTRACT, velocity=0.0, delay_s=0.0)
        args.package = "fake"
    monitor = None
    if args.monitor and str(args.monitor) != "none":
        from ttt.progress_api import ProgressMonitor

        monitor = ProgressMonitor.load(args.monitor)
    out = args.out or WS4 / ".cache" / "ttt" / "flux_sim" / args.package
    out.mkdir(parents=True, exist_ok=True)
    summaries = []
    for i in range(args.start, args.start + args.episodes):
        seed = args.seed * 100_000 + i
        ep = out / f"{'+'.join(s for s, _ in plan)}-{seed:06d}"
        t0 = time.time()
        video = args.video and i - args.start < args.video_episodes
        labels, results = run_episode(args, sim, robot, policy, vocab, plan, seed, ep, monitor, video)
        s = {"episode": ep.name, "seconds": round(time.time() - t0, 1), "steps": results,
             "monitor": monitor_scores(labels) if monitor is not None else None}  # fmt: skip
        summaries.append(s)
        print(json.dumps({"episode": ep.name, "seconds": s["seconds"], "monitor": s["monitor"]}), flush=True)
        with (out / "index.jsonl").open("a") as f:
            f.write(json.dumps(s) + "\n")
    print(json.dumps(summarize(summaries), indent=1))


def summarize(summaries: list[dict]) -> dict:
    """Per-skill FLUX success over steps whose precondition held, and pooled monitor accuracy."""
    per_skill: dict[str, list[int]] = {}
    for s in summaries:
        for r in s["steps"]:
            if r["precondition_ok"]:
                per_skill.setdefault(r["skill"], []).append(int(r["flux_success"]))
    acc = {}
    for key in ("all", "flux", "scripted"):
        n = sum((s["monitor"] or {}).get(f"n_{key}", 0) for s in summaries)
        hit = sum(((s["monitor"] or {}).get(f"stage_acc_{key}") or 0.0) * (s["monitor"] or {}).get(f"n_{key}", 0)
                  for s in summaries)  # fmt: skip
        acc[key] = {"stage_acc": hit / n if n else None, "replans": n}
    return {
        "episodes": len(summaries),
        "flux_success": {k: {"success": sum(v), "steps": len(v)} for k, v in per_skill.items()},
        "monitor": acc,
    }


if __name__ == "__main__":
    main()
