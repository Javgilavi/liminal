"""The generated MJCF compiles, has the agreed names, and agrees with station_geometry.yaml."""

import numpy as np
import pytest

from ws4.sim.geometry import ARM_JOINTS, SIDES, head_camera_pose, load_geometry, quat_from_mat, val

from .conftest import SIM

SCENES = [SIM / "xlerobot_station.xml", *sorted((SIM / "scenes").glob("*.xml"))]


@pytest.fixture(scope="module")
def model():
    import mujoco

    return mujoco.MjModel.from_xml_path(str(SIM / "xlerobot_station.xml"))


@pytest.mark.parametrize("path", SCENES, ids=lambda p: p.name)
def test_scene_compiles(path):
    import mujoco

    m = mujoco.MjModel.from_xml_path(str(path))
    d = mujoco.MjData(m)
    mujoco.mj_step(m, d, nstep=50)
    assert np.isfinite(d.qpos).all()


def test_names(model):
    for s in SIDES:
        for j in ARM_JOINTS:
            assert model.joint(f"{s}_{j}").id >= 0
            assert model.actuator(f"{s}_{j}").id >= 0
        assert model.site(f"{s}_ee").id >= 0
        assert model.body(f"{s}_gripper").id >= 0
    cams = {model.camera(i).name for i in range(model.ncam)}
    assert {"head", "left_wrist", "right_wrist"} <= cams
    assert model.nu == 12


def test_ranges_match_yaml(model):
    rng = val(load_geometry()["arms"]["joint_range_rad"])
    for s in SIDES:
        for j in ARM_JOINTS:
            np.testing.assert_allclose(model.jnt_range[model.joint(f"{s}_{j}").id], rng[j], atol=1e-5)
            np.testing.assert_allclose(model.actuator_ctrlrange[model.actuator(f"{s}_{j}").id], rng[j], atol=1e-5)


def test_base_tcp_and_cameras_match_yaml(model):
    import mujoco

    g = load_geometry()
    d = mujoco.MjData(model)
    mujoco.mj_forward(model, d)
    for s in SIDES:
        b = model.body(f"{s}_base").id
        np.testing.assert_allclose(d.xpos[b], val(g["arms"][s]["base_pos"]), atol=1e-6)
        gb = model.body(f"{s}_gripper").id
        tcp_local = d.xmat[gb].reshape(3, 3).T @ (d.site_xpos[model.site(f"{s}_ee").id] - d.xpos[gb])
        np.testing.assert_allclose(tcp_local, val(g["tcp"]["pos_in_gripper"]), atol=1e-6)
    pos, R = head_camera_pose(g)
    c = model.camera("head").id
    np.testing.assert_allclose(d.cam_xpos[c], pos, atol=1e-5)
    np.testing.assert_allclose(d.cam_xmat[c].reshape(3, 3), R, atol=1e-5)
    derived = g["derived"]["head_camera_world"]
    np.testing.assert_allclose(derived["pos"], pos, atol=1e-5)
    np.testing.assert_allclose(derived["quat_mujoco"], quat_from_mat(R), atol=1e-5)


def test_zero_pose_reaches_forward(model):
    """At the LeRobot calibration middle (all zeros) each arm reaches along +x (Oscar confirmed both face forward)."""
    import mujoco

    d = mujoco.MjData(model)
    mujoco.mj_forward(model, d)
    for s in SIDES:
        base = d.xpos[model.body(f"{s}_base").id]
        ee = d.site_xpos[model.site(f"{s}_ee").id]
        assert ee[0] - base[0] > 0.3
        assert abs(ee[1] - base[1]) < 0.02


def test_fins_touch_at_closed_angle(model):
    import mujoco

    g = load_geometry()
    closed = val(g["gripper"]["closed_rad"])
    d = mujoco.MjData(model)
    ja = model.jnt_qposadr[model.joint("left_gripper").id]
    g1, g2 = model.geom("left_fixed_finger").id, model.geom("left_moving_finger").id

    def gap(q):
        d.qpos[:] = 0
        d.qpos[ja] = q
        mujoco.mj_forward(model, d)
        return mujoco.mj_geomDistance(model, d, g1, g2, 0.2, None)

    assert abs(gap(closed)) < 0.001
    assert gap(closed + 0.1) > 0.004
    assert gap(val(g["gripper"]["open_rad"])) > 0.05  # closest hull gap when fully open (wider at the tips)


def test_no_self_contact_at_rest(model):
    import mujoco

    d = mujoco.MjData(model)
    mujoco.mj_forward(model, d)
    for i in range(d.ncon):
        c = d.contact[i]
        b1, b2 = model.body(model.geom_bodyid[c.geom1]).name, model.body(model.geom_bodyid[c.geom2]).name
        assert c.dist > -0.002, f"penetration {c.dist:.4f} between {b1} and {b2}"
