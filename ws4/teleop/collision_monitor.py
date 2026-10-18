"""Inter-arm clearance monitor for haller VR teleop, on the measured XLeRobot station model.

haller's own collision guard sweeps capsules along the old SO-ARM100 chain, 14 to 36 cm off the real arm at the
same angles, so it is off on this rig (ws4/research/vr_teleop.md section 6). This sidecar checks the arms instead
on ws4/sim/xlerobot_station.xml (so101_new_calib chain, fin-ray fingers, CAD mounts) through ws4/kinematics
`MjCollision`: it polls the backend's GET /teleop/human, maps each session slot to its arm and that arm to its
robot side (haller left = robot RIGHT, calib.py ARMS), converts LeRobot degrees to radians and takes the signed
distance between the two arms' collision meshes (negative = penetration). Below the margin it POSTs /estop once
per session and logs why.

    python3 -I ws4/teleop/collision_monitor.py live            # e-stop below 30 mm, beside up.sh --real
    python3 -I ws4/teleop/collision_monitor.py dry-run         # print clearance only, never e-stops
    python3 -I ws4/teleop/collision_monitor.py replay DATASET  # min clearance per episode of a recording (QA)

It checks the session's committed goals (what the arms are being sent), not measured positions, and only while a
session drives both arms: the backend publishes no pose for an arm outside the session. A second guard, not a
replacement for the person at the E-STOP.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from calib import ARMS, JOINTS  # noqa: E402
from kinematics.so101_kinematics import lerobot_deg_to_rad, lerobot_gripper_to_open  # noqa: E402

DEFAULT_URL = "http://127.0.0.1:8000"
DEFAULT_MARGIN_MM = 30.0
DEFAULT_HZ = 20.0
# Beyond this the exact distance is not computed: the readout saturates and nothing near the margin is lost.
DIST_CAP_M = 0.25
LOG_PATH = Path(os.environ.get("WS4_TELEOP_RUN_DIR", "/tmp/ws4-teleop")) / "collision_monitor.jsonl"
HALLER_TO_ROBOT = {hid: side for side, (hid, *_rest) in ARMS.items()}


def to_model(q_deg: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """robot side -> 6-vector (LeRobot degrees, gripper 0..100) into MjCollision's arm radians + gripper 0..1."""
    q = {s: lerobot_deg_to_rad(v[:5]) for s, v in q_deg.items()}
    g = {s: float(lerobot_gripper_to_open(v[5])) for s, v in q_deg.items()}
    return q, g


class Clearance:
    """Signed distance (m) between the left and right arm collision geoms of the station model.

    Pose and gripper mapping come from ``MjCollision.contacts`` (one source of truth for the qpos layout); the
    distances are MuJoCo's ``mj_geomDistance`` over left x right geom pairs, nearest bounding spheres first.

    MuJoCo 3.8.1 returns exactly 0.0 for about 0.4% of separated mesh pairs within distmax while its ``fromto``
    witness segment still has the true length (1500 random station poses, 155k pairs: every such pair had no
    ``mj_collision`` contact and a witness no shorter than the bounding-sphere bound; every other value equalled
    the witness length). A 0.0 is therefore replaced by the witness length, and ``mj_collision``'s own inter-arm
    penetration is taken as a floor, so the two MuJoCo paths have to agree before a pose counts as clear.
    """

    def __init__(self, scene: Path | None = None, cap: float = DIST_CAP_M):
        import mujoco
        from kinematics.mj_collision import MjCollision

        self.mujoco = mujoco
        self.mc = MjCollision(scene)
        m = self.mc.m
        geoms = {s: [g for g in range(m.ngeom) if self.mc.geom_group[g] == s
                     and (m.geom_contype[g] or m.geom_conaffinity[g])] for s in ("left", "right")}
        self.pairs = np.array([(a, b) for a in geoms["left"] for b in geoms["right"]], dtype=int)
        self.rsum = m.geom_rbound[self.pairs[:, 0]] + m.geom_rbound[self.pairs[:, 1]]
        self.cap = cap
        self._fromto = np.zeros(6)

    def __call__(self, q_deg: dict[str, np.ndarray]) -> tuple[float, tuple[str, str] | None]:
        """Clearance (m, capped) and the closest (left body, right body) for one frame of both robot sides."""
        q, g = to_model(q_deg)
        penetration = self.mc.contacts(q, g, depth=0.0).get("inter_arm", 0.0)  # also sets the pose
        m, d = self.mc.m, self.mc.d
        xpos = d.geom_xpos
        bound = np.linalg.norm(xpos[self.pairs[:, 0]] - xpos[self.pairs[:, 1]], axis=1) - self.rsum
        best, closest = self.cap, None
        for i in np.argsort(bound):
            if bound[i] >= best:
                break
            a, b = self.pairs[i]
            dist = self.mujoco.mj_geomDistance(m, d, int(a), int(b), best, self._fromto)
            if dist == 0.0:
                dist = float(np.linalg.norm(self._fromto[3:] - self._fromto[:3]))
            if dist < best:
                best, closest = float(dist), (self.mc.geom_body[a], self.mc.geom_body[b])
        if penetration > 0 and -penetration < best:
            best = -float(penetration)
        return best, closest


# ---- live: poll the HMI backend ------------------------------------------------------------------------------------

def goals_from_status(status: dict) -> tuple[dict[str, np.ndarray], list[str]]:
    """robot side -> 6-vector of committed goals from a GET /teleop/human body, and why any side is missing.

    ``goal_deg`` is keyed by session SLOT (the headset hand), and ``left_arm`` / ``right_arm`` say which haller arm
    each slot drives: stance "behind" pairs slot left with arm ``right``. Only then does calib.ARMS give the robot
    side. A slot without an arm (one-arm session) or without a full goal is left out.
    """
    out: dict[str, np.ndarray] = {}
    why: list[str] = []
    goals = status.get("goal_deg") or {}
    for slot in ("left", "right"):
        arm = status.get(f"{slot}_arm", slot)  # older backends had no pairing: slot == arm id
        if not arm:
            why.append(f"slot {slot} drives no arm")
            continue
        side = HALLER_TO_ROBOT.get(arm)
        if side is None:
            why.append(f"slot {slot}: unknown haller arm {arm!r} (expected one of {sorted(HALLER_TO_ROBOT)})")
            continue
        goal = goals.get(slot) or {}
        missing = [j for j in JOINTS[:5] if j not in goal]
        if missing:
            why.append(f"haller {arm}: no goal for {', '.join(missing)}")
            continue
        # A goal without the gripper is checked fully open: the widest fins, so the conservative pose.
        v = np.array([float(goal[j]) for j in JOINTS[:5]] + [float(goal.get("gripper", 100.0))])
        if not np.all(np.isfinite(v)):
            why.append(f"haller {arm}: non-finite goal {v.tolist()}")
            continue
        out[side] = v
    return out, why


class Backend:
    """The two haller HMI routes the monitor needs, over plain urllib."""

    def __init__(self, url: str = DEFAULT_URL, timeout: float = 0.5):
        self.url = url.rstrip("/")
        self.timeout = timeout

    def _req(self, method: str, path: str, body: dict | None = None) -> dict:
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(self.url + path, data=data, method=method,
                                     headers={"content-type": "application/json"} if data else {})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read() or b"{}")

    def human_status(self) -> dict:
        return self._req("GET", "/teleop/human")

    def estop(self) -> dict:
        return self._req("POST", "/estop", {})


@dataclass
class Poll:
    state: str  # idle | partial | ok | below | estop | estop_failed | unreachable
    clearance_m: float | None = None
    closest: tuple[str, str] | None = None
    note: str = ""


class Monitor:
    """One poll per ``step()``. E-stops at most once per session (``started_at``); a new session re-arms it."""

    def __init__(self, backend: Backend, clearance: Callable, margin_m: float = DEFAULT_MARGIN_MM / 1000,
                 dry_run: bool = False, log_path: Path | None = LOG_PATH, say: Callable[[str], None] = print):
        self.backend = backend
        self.clearance = clearance
        self.margin_m = margin_m
        self.dry_run = dry_run
        self.log_path = log_path
        self.say = say
        self.tripped_session: object = None
        self.events: list[dict] = []

    def step(self) -> Poll:
        try:
            status = self.backend.human_status()
        except (urllib.error.URLError, OSError, ValueError) as exc:
            return Poll("unreachable", note=str(exc))
        if not status.get("running"):
            reason = status.get("stopped_reason")
            return Poll("idle", note=f"no session running{f' (last stop: {reason})' if reason else ''}")
        session = status.get("started_at")
        q_deg, why = goals_from_status(status)
        if len(q_deg) < 2:
            return Poll("partial", note="; ".join(why) or "one arm in session")
        dist, closest = self.clearance(q_deg)
        if dist >= self.margin_m:
            return Poll("ok", dist, closest)
        if self.dry_run or (session is not None and session == self.tripped_session):
            return Poll("below", dist, closest, "dry run" if self.dry_run else "already e-stopped this session")
        event = {"t_unix": time.time(), "session_started_at": session, "clearance_mm": round(dist * 1000, 1),
                 "margin_mm": round(self.margin_m * 1000, 1), "closest": list(closest or ()),
                 "pairing": {s: status.get(f"{s}_arm") for s in ("left", "right")},
                 "goal_deg_robot_sides": {s: np.round(v, 2).tolist() for s, v in sorted(q_deg.items())}}
        try:
            self.backend.estop()
        except (urllib.error.URLError, OSError, ValueError) as exc:
            event["estop_error"] = str(exc)
            self._log(event)
            return Poll("estop_failed", dist, closest, str(exc))  # not latched: the next poll tries again
        self.tripped_session = session
        self._log(event)
        return Poll("estop", dist, closest)

    def _log(self, event: dict) -> None:
        self.events.append(event)
        what = "E-STOP SENT" if "estop_error" not in event else f"E-STOP FAILED ({event['estop_error']})"
        pair = " / ".join(event["closest"]) or "?"
        self.say(f"[collision_monitor] {what}: arm clearance {event['clearance_mm']:.1f} mm < "
                 f"{event['margin_mm']:.0f} mm ({pair}); robot-side goals {event['goal_deg_robot_sides']}")
        if self.log_path is not None:
            try:
                self.log_path.parent.mkdir(parents=True, exist_ok=True)
                with self.log_path.open("a") as f:
                    f.write(json.dumps(event) + "\n")
            except OSError as exc:
                self.say(f"[collision_monitor] could not append to {self.log_path}: {exc}")

    def run(self, hz: float = DEFAULT_HZ, duration: float | None = None, report_every: float | None = None) -> None:
        """Poll at ``hz``; print state changes, and the clearance every 0.5 s (dry run) or 5 s (live)."""
        period = 1.0 / hz
        report_every = report_every or (0.5 if self.dry_run else 5.0)
        start = last_report = time.monotonic()
        window: list[Poll] = []
        last_state = None
        while duration is None or time.monotonic() - start < duration:
            t0 = time.monotonic()
            p = self.step()
            window.append(p)
            if p.state != last_state and p.state in ("idle", "partial", "unreachable"):
                self.say(f"[collision_monitor] {p.state}: {p.note}")
            last_state = p.state
            if t0 - last_report >= report_every:
                if any(w.clearance_m is not None for w in window):
                    self.say(_report_line(window, self.margin_m))
                window, last_report = [], t0
            time.sleep(max(0.0, period - (time.monotonic() - t0)))


def _report_line(window: list[Poll], margin_m: float) -> str:
    seen = [p for p in window if p.clearance_m is not None]
    low = min(seen, key=lambda p: p.clearance_m)
    flag = f"  BELOW {margin_m * 1000:.0f} mm" if low.clearance_m < margin_m else ""
    return (f"[collision_monitor] {time.strftime('%H:%M:%S')} clearance now {seen[-1].clearance_m * 1000:6.1f} mm, "
            f"min {low.clearance_m * 1000:6.1f} mm ({' / '.join(low.closest or ('-', '-'))}){flag}")


# ---- offline: replay a recording's action column ---------------------------------------------------------------

def _lagged_err(a: np.ndarray, st: np.ndarray, max_lag: int = 15) -> float:
    """Mean |action - state| (deg) at the best lag: state trails the goal, and a fast move would otherwise cost as
    much error as the pose difference between two mirrored arms."""
    T = len(a)
    return min(float(np.abs(a[:T - k] - st[k:]).mean()) for k in range(min(max_lag, T - 1) + 1))


def action_sides(a: np.ndarray, st: np.ndarray, L: list[int], R: list[int]) -> str:
    """Whether one episode's action columns belong to the arm their state columns do.

    haller's tick keys goal_deg by headset slot and the recorder reads it by arm id (human_teleop.py:1826,
    recorder.py:2270), so with a crossed pairing (stance "behind") each arm's goals land under the other arm.
    Dual session: both action halves track the other arm's state ("crossed"). One-arm session: the driven arm's
    goals land in the idle arm's columns and its own columns repeat its state exactly ("solo crossed").
    """
    cols = (L, R)
    err = {(x, y): _lagged_err(a[:, cols[x]], st[:, cols[y]]) for x in (0, 1) for y in (0, 1)}
    same, cross = err[0, 0] + err[1, 1], err[0, 1] + err[1, 0]
    if cross < 0.5 * same and same - cross > 1.0:
        return "crossed"
    for x, y in ((0, 1), (1, 0)):
        repeats_state = np.allclose(a[:, cols[x]], st[:, cols[x]], atol=1e-3)
        if repeats_state and err[y, x] < 0.5 * err[y, y] and err[y, y] - err[y, x] > 1.0:
            return "solo crossed"
    return "as recorded"


def load_actions(dataset: Path, sides_as_recorded: bool = False
                 ) -> tuple[dict[int, dict[str, np.ndarray]], dict[int, str], list[str]]:
    """episode -> robot side -> (T, 6) actions in LeRobot degrees, per-episode `action_sides`, and warnings."""
    import pandas as pd

    info = json.loads((dataset / "meta" / "info.json").read_text())
    warn: list[str] = []
    cal = info.get("haller_joint_calibration") or {}
    if cal.get("state_unit") not in (None, "deg"):
        raise SystemExit(f"{dataset}: joints are in {cal['state_unit']!r}, this replay needs LeRobot degrees")
    sim = sorted(k for k, j in (cal.get("joints") or {}).items() if j.get("source") != "feetech_calibration")
    if sim:
        warn.append(f"{len(sim)} joints not from a Feetech calibration ({sim[0]}, ...): a haller sim rig is not in "
                    "LeRobot units, so these clearances mean nothing")
    names = [n.removesuffix(".pos") for n in info["features"]["action"]["names"]]
    col = {}
    for recorded in ("left", "right"):
        side = recorded if sides_as_recorded else HALLER_TO_ROBOT[recorded]
        try:
            col[side] = [names.index(f"{recorded}_{j}") for j in JOINTS]
        except ValueError as exc:
            raise SystemExit(f"{dataset}: action names lack {recorded}_<joint> ({exc}); names: {names}") from None
    files = sorted((dataset / "data").glob("**/*.parquet"))
    if not files:
        raise SystemExit(f"{dataset}: no data/**/*.parquet")
    has_state = info["features"].get("observation.state", {}).get("names") == info["features"]["action"]["names"]
    cols = ["episode_index", "frame_index", "action"] + (["observation.state"] if has_state else [])
    df = pd.concat([pd.read_parquet(f, columns=cols) for f in files]).sort_values(["episode_index", "frame_index"])
    out: dict[int, dict[str, np.ndarray]] = {}
    sides: dict[int, str] = {}
    for ep, g in df.groupby("episode_index"):
        a = np.stack(g["action"].to_numpy()).astype(float)
        out[int(ep)] = {s: a[:, idx] for s, idx in col.items()}
        if has_state:
            st = np.stack(g["observation.state"].to_numpy()).astype(float)
            sides[int(ep)] = action_sides(a, st, col["left"][:5], col["right"][:5])
    # Checked per episode: haller appends every take of a task to one dataset, front and behind takes alike.
    crossed = [e for e, v in sides.items() if v == "crossed"]
    solo = [e for e, v in sides.items() if v == "solo crossed"]
    if crossed:
        warn.append(f"episodes {crossed}: both action halves track the OTHER arm's observation.state (haller "
                    "recorder with a crossed headset pairing, stance behind); their clearances are for swapped arms. "
                    "Swap action[:, :6] and action[:, 6:] in those episodes before to_flux.py"
                    + ("; --sides-as-recorded replays the whole dataset swapped" if len(crossed) == len(sides) else ""))
    if solo:
        warn.append(f"episodes {solo}: one arm's action repeats its own state while the other arm's action tracks "
                    "it (crossed one-arm take); a swap does not repair these, re-record or fix per arm")
    if not has_state:
        warn.append("no observation.state with the action's names: action/state side pairing not checked")
    return out, sides, warn


def replay(dataset: Path, clearance: Callable, margin_m: float = DEFAULT_MARGIN_MM / 1000,
           sides_as_recorded: bool = False) -> tuple[list[dict], list[str]]:
    """Minimum inter-arm clearance per episode of a recording's commanded actions."""
    episodes, sides, warn = load_actions(dataset, sides_as_recorded)
    rows = []
    for ep, arms in episodes.items():
        T = len(arms["left"])
        dist = np.empty(T)
        closest: list[tuple[str, str] | None] = []
        for t in range(T):
            dist[t], c = clearance({s: v[t] for s, v in arms.items()})
            closest.append(c)
        i = int(np.argmin(dist))
        rows.append({"episode": ep, "frames": T, "min_clearance_mm": round(float(dist[i]) * 1000, 1),
                     "min_frame": i, "closest": list(closest[i] or ()),
                     "frames_below_margin": int(np.sum(dist < margin_m)), "frames_touching": int(np.sum(dist < 0)),
                     "action_sides": sides.get(ep, "unchecked")})
    return rows, warn


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, text in (("live", "poll the backend, POST /estop below the margin"),
                       ("dry-run", "poll the backend and print clearance, never e-stop")):
        p = sub.add_parser(name, help=text)
        p.add_argument("--url", default=os.environ.get("WS4_HMI_URL", DEFAULT_URL))
        p.add_argument("--hz", type=float, default=DEFAULT_HZ)
        p.add_argument("--duration", type=float, default=None, help="seconds, default until Ctrl-C")
        p.add_argument("--log", type=Path, default=LOG_PATH, help="JSON lines of e-stop events")
    r = sub.add_parser("replay", help="min clearance per episode of a recorded dataset's action column")
    r.add_argument("dataset", type=Path)
    r.add_argument("--sides-as-recorded", action="store_true",
                   help="columns already use robot sides (e.g. to_flux.py _xle12); default swaps haller sides")
    r.add_argument("--json", type=Path, default=None, help="also write the report here")
    for p in sub.choices.values():
        p.add_argument("--margin-mm", type=float, default=DEFAULT_MARGIN_MM)
        p.add_argument("--scene", type=Path, default=None, help="MJCF, default ws4/sim/xlerobot_station.xml")
    args = ap.parse_args(argv)
    margin = args.margin_mm / 1000
    clearance = Clearance(args.scene)

    if args.cmd == "replay":
        rows, warn = replay(args.dataset, clearance, margin, args.sides_as_recorded)
        for w in warn:
            print(f"WARNING: {w}")
        print(f"{'episode':>7} {'frames':>6} {'min mm':>7} {'at':>5} {'<margin':>7} {'touch':>5}  {'sides':<12}  "
              "closest")
        for row in rows:
            print(f"{row['episode']:>7} {row['frames']:>6} {row['min_clearance_mm']:>7.1f} {row['min_frame']:>5} "
                  f"{row['frames_below_margin']:>7} {row['frames_touching']:>5}  {row['action_sides']:<12}  "
                  f"{' / '.join(row['closest'])}")
        bad = [row["episode"] for row in rows if row["min_clearance_mm"] < args.margin_mm]
        crossed = [row["episode"] for row in rows if "crossed" in row["action_sides"]]
        print(f"{len(rows)} episodes, {len(bad)} below {args.margin_mm:.0f} mm" + (f": {bad}" if bad else "")
              + (f"; {len(crossed)} with crossed action sides: {crossed}" if crossed else ""))
        if args.json:
            args.json.write_text(json.dumps({"dataset": str(args.dataset), "margin_mm": args.margin_mm,
                                             "warnings": warn, "episodes": rows}, indent=1))
        return 1 if bad or crossed else 0

    mon = Monitor(Backend(args.url), clearance, margin, dry_run=args.cmd == "dry-run", log_path=args.log)
    mode = "dry run, never e-stops" if mon.dry_run else f"POST {args.url}/estop below the margin, log {args.log}"
    print(f"[collision_monitor] {args.url}/teleop/human at {args.hz:g} Hz, margin {args.margin_mm:g} mm ({mode})")
    try:
        mon.run(args.hz, args.duration)
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
