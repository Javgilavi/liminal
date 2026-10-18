"""liminal's own FLUX backend with engine: remote, against flux.remote's real HTTP server serving fake packages under
the Haller demo names: the shipped assemble_components task on mock arms, no GPU, no pod."""

import asyncio
import threading
from pathlib import Path

import pytest
import yaml

np = pytest.importorskip("numpy")
pytest.importorskip("PIL")

from liminal.app import build_context  # noqa: E402
from liminal.orchestration import load_task, run_task  # noqa: E402
from liminal.orchestration.component_assembly import DEFINITIONS  # noqa: E402
from liminal.perception.inspector import SceneInspector  # noqa: E402
from liminal.perception.types import InspectionProfile  # noqa: E402
from liminal.ws4_bridge import ws4_on_path  # noqa: E402

REGISTRY = Path("configs/policy/haller_remote_registry.yaml")
PACKAGES = {"lid_open": "skill_open_box", "green_pen_in_slot": "skill_green_pen",
            "yellow_boxcutter_in_slot": "skill_yellow_cutter", "red_pen_in_slot": "skill_red_pen"}


@pytest.fixture
def served(tmp_path):
    ws4_on_path()
    from flux.backends import FakeBackend
    from flux.contract import SO101_CONTRACT
    from flux.remote import make_server

    backends = {pkg: FakeBackend(SO101_CONTRACT, velocity=0.5, delay_s=0.002) for pkg in PACKAGES.values()}
    server = make_server(backends, "127.0.0.1", 0, token="remote-token")
    threading.Thread(target=server.serve_forever, daemon=True).start()
    token = tmp_path / "token"
    token.write_text("remote-token\n")
    yield f"http://127.0.0.1:{server.server_address[1]}", str(token), backends
    server.shutdown()
    server.server_close()


class ServedProgress:
    """A predicate holds once its package has served two chunks (the fake 'policy' did its work)."""

    def __init__(self, backends):
        self.backends = backends

    def truth(self, predicate):
        if predicate == "components_assembled":
            return all(self.truth(p) for p in PACKAGES if p != "lid_open")
        return len(self.backends[PACKAGES[predicate]].requests) >= 2

    async def inspect(self, prompt, images):
        names = [p for p in DEFINITIONS if f"- {p}:" in prompt]
        return {"predicates": {p: {"value": self.truth(p), "evidence": "served chunks"} for p in names}}


def remote_context(served, tmp_path):
    url, token, backends = served
    policy = yaml.safe_load(REGISTRY.read_text())
    policy.update(server=url, token_file=token)
    ctx = build_context("configs/robot/mock_dual.yaml", backend="flux", policy=policy, log_path=tmp_path / "log.jsonl")
    ctx.inspection = InspectionProfile(cameras=("top",), definitions=dict(DEFINITIONS), timeout_s=5.0)
    ctx.inspector = SceneInspector(ServedProgress(backends), ctx.cameras, ctx.world, confirmations=1)
    return ctx


def test_registry_names_the_served_packages():
    cfg = yaml.safe_load(REGISTRY.read_text())
    assert cfg["engine"] == "remote" and cfg["server"].endswith(":8790")
    assert {k: v["package"] for k, v in cfg["checkpoints"].items()} == {
        "open_box": "skill_open_box", "put_green_pen_in_slot": "skill_green_pen",
        "put_yellow_boxcutter_in_slot": "skill_yellow_cutter", "put_red_pen_in_slot": "skill_red_pen"}
    assert cfg["checkpoints"]["open_box"]["targets"] == ["right"]
    assert all(cfg["checkpoints"][k]["targets"] == ["left"] for k in cfg["checkpoints"] if k != "open_box")
    assert "lora_pen84/005000/raw" in cfg["checkpoints"]["put_red_pen_in_slot"]["path"]


def test_assemble_components_on_mock_arms_with_remote_flux(served, tmp_path):
    url, _, backends = served
    ctx = remote_context(served, tmp_path)
    report = asyncio.run(run_task(load_task("configs/tasks/assemble_components.yaml", arms=ctx.arm_names), ctx))
    assert report.goal_verified, report
    task = yaml.safe_load(Path("configs/tasks/assemble_components.yaml").read_text())
    prompts = {}
    for node in [task["tree"]["sequence"][1]["fallback"][1]] + [
            n["fallback"][1] for n in task["tree"]["sequence"][2]["while"]["do"]["sequence"] if "fallback" in n]:
        prompts[node["skill"]["name"]] = node["skill"]["prompt"]
    served_tasks = {pkg: {r.task for r in b.requests} for pkg, b in backends.items()}
    assert served_tasks == {"skill_open_box": {prompts["open_box"]},
                            "skill_green_pen": {prompts["put_green_pen_in_slot"]},
                            "skill_yellow_cutter": {prompts["put_yellow_boxcutter_in_slot"]},
                            "skill_red_pen": {prompts["put_red_pen_in_slot"]}}
    request = backends["skill_red_pen"].requests[0]
    assert request.states.shape == request.commands.shape == (8, 6) and request.snapshot_ticks == (0, 7)
    assert set(request.images) == {"observation.images.scene", "observation.images.wrist"}
    assert request.images["observation.images.wrist"].shape == (2, 256, 256, 3)
    loaded = [e["identity"] for e in ctx.world.events if e["kind"] == "checkpoint_loaded"]
    assert {i["package"] for i in loaded} == set(PACKAGES.values()) and all(i["server"] == url for i in loaded)
    # every executed action went through the supervisor to the mock arm of the skill's own side
    assert ctx.arms["left"].history and ctx.arms["right"].history


def test_remote_engine_refuses_a_contract_that_does_not_match(served, tmp_path):
    from liminal.policy.flux import FluxConfig
    from liminal.policy.remote_engine import RemoteFluxEngine

    url, token, _ = served
    with pytest.raises(ValueError, match="does not match"):
        RemoteFluxEngine(FluxConfig(engine="remote", server=url, token_file=token, chunk_size=50), "skill_red_pen")
    with pytest.raises(RuntimeError, match="401"):
        RemoteFluxEngine(FluxConfig(engine="remote", server=url, token_cmd_env=""), "skill_red_pen")
    with pytest.raises(RuntimeError, match="404"):
        RemoteFluxEngine(FluxConfig(engine="remote", server=url, token_file=token, chunk_size=42), "skill_nope")


def test_pod_worker_speaks_the_official_live_worker_protocol(served):
    """scripts/flux_pod_worker.py: official_live's NDJSON worker protocol, answered by a served package."""
    import base64
    import io
    import json
    import os
    import subprocess
    import sys

    from PIL import Image

    url, token, backends = served
    jpeg = io.BytesIO()
    Image.new("RGB", (640, 480), (90, 40, 20)).save(jpeg, format="JPEG")
    frame = base64.b64encode(jpeg.getvalue()).decode()
    history = [{"state": [0.0, 1.0, 2.0, 3.0, 4.0, 50.0], "command_history": [0.0, 1.0, 2.0, 3.0, 4.0, 50.0],
                "scene": frame, "wrist": frame} for _ in range(8)]
    proc = subprocess.run(
        [sys.executable, "-u", "scripts/flux_pod_worker.py", "--server", url, "--package", "skill_red_pen",
         "--token-file", token],
        input=json.dumps({"prompt": "put the red pen in its slot in the cardboard box", "history": history}) + "\n",
        capture_output=True, text=True, timeout=60, env={**os.environ, "PYTHONPATH": "src"})
    ready, reply = (json.loads(line) for line in proc.stdout.splitlines())
    assert ready["type"] == "ready" and ready["integration"].startswith("lerobot.Flux3Policy")
    assert ready["inference"] == {"chunk_size": 42, "n_action_steps": 16, "fps": 30}
    assert reply["type"] == "actions" and len(reply["actions"]) == 42 and len(reply["actions"][0]) == 6
    request = backends["skill_red_pen"].requests[-1]
    assert request.task == "put the red pen in its slot in the cardboard box"
    assert request.images["observation.images.wrist"].shape == (2, 256, 256, 3)
