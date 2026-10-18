import re
from pathlib import Path

import numpy as np
import pytest
from kinematics import so101_kinematics as k
from kinematics.se3 import rpy_matrix

CACHE = Path(__file__).resolve().parents[2] / ".cache" / "kin"
URDF = CACHE / "SO-ARM100" / "Simulation" / "SO101" / "so101_new_calib.urdf"
MJCF = CACHE / "SO-ARM100" / "Simulation" / "SO101" / "so101_new_calib.xml"


def test_zero_pose_matches_urdf_geometry():
    # New calibration zero: upper arm up, forearm and gripper horizontal, pointing along +x of base_link.
    T = k.ArmModel().fk(np.zeros(5), frame="flange")
    assert np.allclose(T[:3, 3], [0.3914, 0.0, 0.2265], atol=5e-4)
    assert np.allclose(T[:3, 2], [1.0, 0.0, 0.0], atol=1e-4)  # flange z = approach


def test_batched_and_single_fk_agree():
    arm = k.ArmModel(tcp_extra=0.03)
    q = np.random.default_rng(0).uniform(-1.5, 1.5, (20, 5))
    Tb = arm.fk(q)
    for qi, Ti in zip(q, Tb, strict=True):
        assert np.allclose(arm.fk_single(qi), Ti, atol=1e-12)


def test_tcp_extra_moves_along_approach():
    q = np.array([0.3, -0.4, 0.5, 0.2, 1.0])
    T0 = k.ArmModel().fk(q)
    T1 = k.ArmModel(tcp_extra=0.05).fk(q)
    assert np.allclose(T1[:3, 3] - T0[:3, 3], 0.05 * T0[:3, 2], atol=1e-12)


def test_jacobian_matches_finite_differences():
    arm = k.ArmModel(tcp_extra=0.02)
    q = np.array([0.2, -0.3, 0.7, -0.4, 0.9])
    J, T = arm.jacobian(q)
    h = 1e-6
    for i in range(5):
        dq = np.zeros(5)
        dq[i] = h
        Tp = arm.fk_single(q + dq)
        assert np.allclose((Tp[:3, 3] - T[:3, 3]) / h, J[:3, i], atol=1e-5)
        dR = Tp[:3, :3] @ T[:3, :3].T
        w = np.array([dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0], dR[1, 0] - dR[0, 1]]) / 2 / h
        assert np.allclose(w, J[3:, i], atol=1e-5)


def test_unit_conversions():
    deg = np.array([10.0, -45.0, 90.0, 0.0, 170.0])
    assert np.allclose(k.rad_to_lerobot_deg(k.lerobot_deg_to_rad(deg)), deg)
    assert np.allclose(k.lerobot_deg_to_rad(deg), np.deg2rad(deg))
    assert np.allclose(k.lerobot_m100_to_rad(np.full(5, 100.0)), np.deg2rad(k.M100_DEG_PER_UNIT * 100))
    assert np.allclose(k.lerobot_gripper_to_open([0, 50, 100, 120]), [0, 0.5, 1, 1])
    assert np.allclose(k.open_to_lerobot_gripper(k.lerobot_gripper_to_open([0, 37, 100])), [0, 37, 100])


@pytest.mark.skipif(not URDF.exists(), reason="SO-ARM100 checkout not cached")
def test_constants_match_cached_urdf():
    text = URDF.read_text()
    for name, (xyz, rpy) in zip(k.JOINT_NAMES, k._JOINT_ORIGINS, strict=True):
        block = re.search(rf'<joint name="{name}" type="revolute">(.*?)</joint>', text, re.S).group(1)
        o = re.search(r'<origin xyz="([^"]+)" rpy="([^"]+)"', block)
        assert np.allclose([float(x) for x in o.group(1).split()], xyz)
        assert np.allclose([float(x) for x in o.group(2).split()], rpy)
        lim = re.search(r'lower="([^"]+)" upper="([^"]+)"', block)
        i = k.JOINT_NAMES.index(name)
        assert np.allclose([float(lim.group(1)), float(lim.group(2))], k.JOINT_LIMITS[i])


@pytest.mark.skipif(not MJCF.exists(), reason="SO-ARM100 checkout not cached")
def test_fk_matches_mujoco():
    mujoco = pytest.importorskip("mujoco")
    m = mujoco.MjModel.from_xml_path(str(MJCF))
    d = mujoco.MjData(m)
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
    adr = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in k.JOINT_NAMES]
    arm = k.ArmModel()
    rng = np.random.default_rng(0)
    R_offset = None
    for _ in range(100):
        q = rng.uniform(-1.5, 1.5, 5)
        d.qpos[:] = 0
        d.qpos[adr] = q
        mujoco.mj_kinematics(m, d)
        T = arm.fk(q, frame="flange")
        assert np.linalg.norm(d.site_xpos[sid] - T[:3, 3]) < 1e-5
        # The MJCF site uses other axes than the URDF frame; the offset must be constant.
        off = T[:3, :3].T @ d.site_xmat[sid].reshape(3, 3)
        R_offset = off if R_offset is None else R_offset
        assert np.allclose(off, R_offset, atol=1e-4)


def test_station_defaults_and_frames():
    st = k.Station.load(Path("/nonexistent.yaml"))
    assert "UNVERIFIED" in st.source
    q = np.array([0.1, -0.2, 0.3, 0.4, 0.5])
    T_s = st.fk("left", q)
    assert np.allclose(st.to_base("left", T_s), st.arm.fk(q))
    assert st.T_station_base["left"][1, 3] > 0 > st.T_station_base["right"][1, 3]


def test_station_load_yaml(tmp_path):
    p = tmp_path / "station_geometry.yaml"
    p.write_text("arms:\n  left:\n    base_pose: {xyz: [0.01, 0.2, 0.12], rpy: [0, 0, 0.1]}\ntable_z: -0.01\n")
    st = k.Station.load(p)
    assert np.allclose(st.T_station_base["left"][:3, 3], [0.01, 0.2, 0.12])
    assert np.allclose(st.T_station_base["left"][:3, :3], rpy_matrix(0, 0, 0.1))
    assert st.table_z == -0.01
