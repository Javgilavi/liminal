"""Operator console API on mock/replay only: no GPU, network or serial ports."""

import time

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from liminal.ui.server import create_app  # noqa: E402

WORKBENCH = "configs/replay/workbench.yaml"
GOAL = "put the red block in the box"


@pytest.fixture()
def client(tmp_path):
    app = create_app("configs/robot/mock_dual.yaml", runs_dir=tmp_path)
    with TestClient(app) as c:
        yield c


def ground(client, goal=GOAL):
    r = client.post("/api/ground", json={"scenario": WORKBENCH, "goal": goal, "model": "replay"})
    assert r.status_code == 200, r.text
    return r.json()


def wait_done(client, run_id, timeout=20):
    end = time.time() + timeout
    while time.time() < end:
        s = client.get(f"/api/runs/{run_id}").json()
        if not s["active"]:
            return s
        time.sleep(0.1)
    raise AssertionError("run did not finish")


def test_index_page_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "not hardware" in r.text
    assert client.get("/web/app.js").status_code == 200
    assert client.get("/web/style.css").status_code == 200
    assert client.get("/web/../server.py").status_code == 404


def test_options(client):
    o = client.get("/api/options").json()
    assert WORKBENCH in o["scenarios"]
    assert set(o["modes"]) == {"planner", "authored"} and "replay" in o["models"]
    assert "not hardware" in o["banner"]
    assert "features" in o


def test_ground_free_text_returns_flux_rows(client):
    g = ground(client)
    assert g["rows"] and g["rows"][0]["flux_prompt"]
    assert g["rows"][0]["arm"] in ("left", "right")
    assert g["proposal"]["steps"] and isinstance(g["warnings"], list)


def test_ground_rejects_bad_input(client):
    assert client.post("/api/ground", json={"scenario": WORKBENCH, "goal": " "}).status_code == 422
    assert client.post("/api/ground", json={"scenario": "/etc/passwd", "goal": GOAL}).status_code == 422


def test_forged_procedure_is_422(client):
    proc = ground(client)["proposal"]
    proc["steps"][0]["skill"] = "teleport"
    r = client.post("/api/runs", json={"scenario": WORKBENCH, "procedure": proc})
    assert r.status_code == 422 and r.json()["detail"]
    proc = ground(client)["proposal"]
    proc["steps"][0]["arm"] = "middle"
    assert client.post("/api/runs", json={"scenario": WORKBENCH, "procedure": proc}).status_code == 422


def test_run_completes_with_events_and_memory(client):
    proc = ground(client)["proposal"]
    r = client.post("/api/runs", json={"scenario": WORKBENCH, "mode": "planner", "procedure": proc})
    assert r.status_code == 200, r.text
    run_id = r.json()["run_id"]
    status = wait_done(client, run_id)
    assert status["status"] == "completed", status
    assert status["obligations"] and all(o["status"] in ("verified", "completed") for o in status["obligations"])
    assert "not hardware" in status["banner"]

    ev = client.get(f"/api/runs/{run_id}/events?since=0").json()
    assert ev["events"] and ev["next"] == len(ev["events"])
    assert any(e["kind"].startswith("episode_") for e in ev["events"])
    tail = client.get(f"/api/runs/{run_id}/events?since={ev['next']}").json()
    assert tail["events"] == []

    with client.stream("GET", f"/api/runs/{run_id}/events?stream=1") as s:
        body = "".join(s.iter_text())
    assert "data:" in body and "event: done" in body

    png = client.get(f"/api/runs/{run_id}/scene.png")
    assert png.status_code == 200 and png.content[:4] == b"\x89PNG"

    mem = client.get(f"/api/runs/{run_id}/memory").json()
    assert mem["has_ledger"] and mem["obligations"] and mem["attempts"]
    for f in mem["frames"][:1]:
        img = client.get(f["url"])
        assert img.status_code == 200 and img.content
    assert client.get(f"/api/frames/{run_id}/nope").status_code == 404
    assert client.get("/api/runs/missing").status_code == 404


def test_authored_run(client):
    proc = ground(client)["proposal"]
    r = client.post("/api/runs", json={"scenario": WORKBENCH, "mode": "authored", "procedure": proc})
    assert r.status_code == 200, r.text
    assert wait_done(client, r.json()["run_id"])["status"] == "completed"


def test_interrupt_then_resume(client):
    proc = ground(client, "put the red block in the box and the blue block in the left bin")["proposal"]
    r = client.post("/api/runs", json={"scenario": WORKBENCH, "procedure": proc, "interrupt_after": 1})
    first = wait_done(client, r.json()["run_id"])
    if first["status"] == "completed":
        pytest.skip("interrupt not reached")
    res = client.post(f"/api/runs/{first['run_id']}/resume")
    assert res.status_code == 200, res.text
    assert wait_done(client, res.json()["run_id"])["status"] == "completed"


def test_estop(client):
    proc = ground(client)["proposal"]
    run_id = client.post("/api/runs", json={"scenario": WORKBENCH, "procedure": proc}).json()["run_id"]
    r = client.post("/estop")
    assert r.status_code == 200 and r.json()["estopped"] is True
    assert client.get("/state").json()["estopped"] is True
    wait_done(client, run_id)


def test_ground_video_501_when_unwired(client, monkeypatch):
    from liminal.orchestration import service

    monkeypatch.delattr(service, "ground_demo", raising=False)
    r = client.post("/api/ground_video", json={"scenario": WORKBENCH, "demo": "x"})
    assert r.status_code == 501 and "not wired" in r.json()["detail"]
