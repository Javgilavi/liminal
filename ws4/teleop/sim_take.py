"""Record scripted bimanual takes against a running haller HMI backend, no headset needed.

It speaks the exact WebXR wire format the Quest page sends (haller_ws scripts/vr_smoke.py `frame()`), so the
take goes through the same session, IK, collision guard and in-process recorder as a real headset take. Use it
to rehearse the recording pipeline and to produce sim episodes for the contract checks. Never point it at real
arms unattended: it moves whatever the backend drives.

    ~/venvs/haller-hmi/bin/python ws4/teleop/sim_take.py --base http://localhost:8000 \
        --repo-id ws4sim/xle_pipeline_check --task "Pick up the red cube" --episodes 3 --seconds 6
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import math
import os
from pathlib import Path

HALLER_WS = Path(os.environ.get("HALLER_WS", Path.home() / "haller_ws"))


def load_vr_smoke(haller_ws: Path = HALLER_WS):
    """haller_ws's own scripted operator (wire format, REST helper). Loaded by path: it is a script, not a package."""
    path = haller_ws / "scripts" / "vr_smoke.py"
    spec = importlib.util.spec_from_file_location("haller_vr_smoke", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def wave(i: int, n: int, amp: float, phase: float = 0.0) -> float:
    """Smooth 0 -> amp -> 0 profile over n samples (both ends at rest, so the take starts and ends still)."""
    return amp * math.sin(math.pi * i / max(1, n - 1)) * math.cos(phase * math.pi * i / max(1, n - 1))


async def one_take(vs, base: str, ws_url: str, repo_id: str, task: str, seconds: float, hz: float = 30.0) -> dict:
    vs.req(base, "/teleop/human/stop", {})
    vs.req(base, "/teleop/human/start", {"left_arm": "left", "right_arm": "right"})
    async with vs.ws_connect(ws_url) as ws:
        st = await vs.wait_for(base, lambda s: vs.authorities(s) == ("driving", "driving"), 8.0, ws=ws)
        if st is None:
            raise SystemExit("both sides never reached DRIVING")
        vs.req(base, "/record/start", {"repo_id": repo_id, "task": task})
        n = max(2, int(seconds * hz))
        for i in range(n):
            msg = vs.json.loads(vs.frame(left_dx=wave(i, n, 0.10), right_dx=-wave(i, n, 0.10, 0.5)))
            # trigger = gripper (0 open, 1 closed): close in the middle of the take
            grip = max(0.0, math.sin(math.pi * i / (n - 1))) ** 2
            msg["left"]["trigger"] = grip
            msg["right"]["trigger"] = 1.0 - grip
            msg["left"]["position"][1] += wave(i, n, 0.05)
            await ws.send(vs.json.dumps(msg))
            await asyncio.sleep(1.0 / hz)
        out = vs.req(base, "/record/stop", {"save": True})
    vs.req(base, "/teleop/human/stop", {})
    return out


async def run(base: str, repo_id: str, task: str, episodes: int, seconds: float, reset_seed: int | None) -> None:
    vs = load_vr_smoke()
    ws_url = base.replace("http", "ws", 1) + "/ws/teleop/vr/in"
    for k in range(episodes):
        if reset_seed is not None:
            vs.req(base, "/sim/scene/reset", {"seed": reset_seed + k, "randomize": True, "home_arms": True})
        out = await one_take(vs, base, ws_url, repo_id, task, seconds)
        print(f"episode {out.get('episode_index')}: frames={out.get('episode_frames')} "
              f"skipped={out.get('skipped_frames')} last_error={out.get('last_error')}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base", default="http://localhost:8000")
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--reset-seed", type=int, default=1001, help="deal the sim scene per take; -1 to skip")
    args = ap.parse_args(argv)
    seed = None if args.reset_seed < 0 else args.reset_seed
    asyncio.run(run(args.base, args.repo_id, args.task, args.episodes, args.seconds, seed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
