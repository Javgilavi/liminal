"""Render the scripted dual-arm shot in the Haller MuJoCo twin for the launch video.

The left arm opens the box lid and keeps holding it; only then does the right arm place the red marker
inside, the order the `while` node of configs/tasks/lid_hold_insert.yaml enforces.
Joint targets come from the twin's own expert planner (IK on its waypoints); motion is keyframed in
joint space, not simulated, so this is an illustration of the task, not a policy rollout.

Run with the twin's venv (mujoco, numpy, pillow), from video/:
  MUJOCO_GL=egl ../examples/dual_so101_sim/.venv/bin/python scripts/render_twin.py public/twin_16x9 1920 1080
Writes frames as JPEGs plus track.json (normalised 2D positions of both grippers per frame).
"""

import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE.parent / "examples/dual_so101_sim"))

import mujoco  # noqa: E402
from PIL import Image  # noqa: E402
from twin import config as C  # noqa: E402
from twin.expert import Arm, open_box_waypoints, pick_place_waypoints, plan_waypoints  # noqa: E402
from twin.scene import JOINTS, REST_DEG, Layout, scene_xml  # noqa: E402

FPS = 30
DUR = 8.0
LID_DEG = 80.0  # the twin expert opens to 75-85 deg; past ~85 the left arm runs out of range
SPEED_LEFT = 2.0
SPEED_RIGHT = 2.2
LEFT_START = 0.2  # the right arm starts once the lid is open, as the WHILE node in lid_hold_insert.yaml requires


class FixedRng:
    """Deterministic stand-in for the expert's rng: no noise, a fixed lid angle."""

    def uniform(self, lo, hi):
        return LID_DEG

    def normal(self, loc, scale, size=None):
        return np.zeros(size) if size else 0.0

    def random(self):
        return 1.0


def studio_xml(layout: Layout, w: int, h: int) -> str:
    xml = scene_xml(layout)
    xml = xml.replace('<global offwidth="1280" offheight="960"/>',
                      f'<global offwidth="{max(w, 1920)}" offheight="{max(h, 1920)}"/>')
    xml = xml.replace('<quality shadowsize="2048"/>', '<quality shadowsize="8192" offsamples="8"/>')
    # Dark studio around the measured tabletop: the table, box and arms keep their twin colours.
    xml = xml.replace('<headlight ambient="0.25 0.25 0.25" diffuse="0.3 0.3 0.3" specular="0 0 0"/>',
                      '<headlight ambient="0.16 0.17 0.17" diffuse="0.12 0.12 0.12" specular="0 0 0"/>'
                      '<rgba haze="0.02 0.03 0.03 1"/>')
    xml = xml.replace('<asset>', '<asset>\n    <texture name="sky" type="skybox" builtin="gradient" '
                      'rgb1="0.03 0.05 0.045" '
                      'rgb2="0.0 0.0 0.0" width="512" height="512"/>\n'
                      '    <material name="floor_m" rgba="0.035 0.04 0.04 1" reflectance="0.3"/>')
    xml = xml.replace('<geom name="floor" type="plane" size="3 3 0.05" pos="0 0 -0.75" rgba="0.82 0.80 0.76 1"/>',
                      '<geom name="floor" type="plane" size="3 3 0.05" pos="0 0 -0.75" material="floor_m"/>')
    xml = xml.replace('rgba="0.94 0.93 0.90 1" contype="0"', 'rgba="0.02 0.025 0.025 1" contype="0"')
    xml = xml.replace('rgba="0.90 0.88 0.84 1" contype="0"', 'rgba="0.02 0.025 0.025 1" contype="0"')
    lights = """
    <light name="key" mode="targetbody" target="box" pos="0.9 -0.9 1.3" diffuse="0.5 0.49 0.47" specular="0.2 0.2 0.2"
           cutoff="50" exponent="4" castshadow="true"/>
    <light name="rim" pos="-0.6 0.9 0.9" dir="0.4 -0.7 -0.5" diffuse="0.22 0.42 0.34" specular="0.2 0.3 0.25"
           castshadow="false"/>
    <camera name="cine" pos="0 -1 0.6" xyaxes="1 0 0 0 0.6 0.8" fovy="38"/>"""
    xml = xml.replace('<geom name="floor"', lights + '\n    <geom name="floor"', 1)
    # Dim the original room lights so the key and rim lights shape the scene.
    xml = xml.replace('name="ceiling" pos="0.2 -0.4 1.6" dir="-0.1 0.2 -1" diffuse="0.55 0.55 0.53"',
                      'name="ceiling" pos="0.2 -0.4 1.6" dir="-0.1 0.2 -1" diffuse="0.25 0.25 0.24"')
    xml = xml.replace('name="window" pos="-1.0 -1.2 1.0" dir="0.6 0.7 -0.4" diffuse="0.25 0.25 0.27"',
                      'name="window" pos="-1.0 -1.2 1.0" dir="0.6 0.7 -0.4" diffuse="0.08 0.08 0.09"')
    return xml


def build(w: int, h: int):
    model = mujoco.MjModel.from_xml_string(studio_xml(Layout(), w, h))
    for i in range(model.nmat):  # CAD materials are yellow; the real arms are printed green
        r, g, b, _ = model.mat_rgba[i]
        if r > 0.8 and g > 0.6 and b < 0.3:
            model.mat_rgba[i] = C.ARM_RGBA
    data = mujoco.MjData(model)
    for arm in ("left", "right"):
        for j, v in zip(JOINTS, REST_DEG, strict=True):
            jid = model.joint(f"{arm}_{j}").id
            lo, hi = model.jnt_range[jid]
            data.qpos[model.jnt_qposadr[jid]] = math.radians(v) if j != "gripper" else lo + v / 100 * (hi - lo)
    mujoco.mj_forward(model, data)
    return model, data


def smooth(s: float) -> float:
    s = min(1.0, max(0.0, s))
    return s * s * (3 - 2 * s)


def keyframes(arm: Arm, wps, sols, start: float, speed: float):
    """[(t_start, t_end, q_from, q_to, grip_from, grip_to, index)] for one arm, joint-space smoothstep."""
    lo, hi = arm.glo, arm.ghi
    q = arm.d.qpos[arm.ik.qadr].copy()
    grip = arm.state()[5]
    t = start
    out = []
    for i, (wp, sol) in enumerate(zip(wps, sols, strict=True)):
        dt = wp.seconds / speed
        out.append((t, t + dt, q.copy(), sol.copy(), grip, wp.grip, i))
        q, grip, t = sol.copy(), wp.grip, t + dt
    return out, (lo, hi)


def pose_at(frames, t):
    for f in frames:
        if t < f[1]:
            s = smooth((t - f[0]) / (f[1] - f[0])) if t >= f[0] else 0.0
            return f[2] + (f[3] - f[2]) * s, f[4] + (f[5] - f[4]) * s, f[6], s
    f = frames[-1]
    return f[3], f[5], f[6], 1.0


def set_arm(model, data, arm: Arm, q, grip, grange):
    lo, hi = grange
    data.qpos[arm.qadr[:5]] = q[:5]
    data.qpos[arm.qadr[5]] = lo + np.clip(grip, 0, 100) / 100 * (hi - lo)


def look_at(pos, target):
    z = np.asarray(pos, float) - target
    z /= np.linalg.norm(z)
    x = np.cross([0, 0, 1.0], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, np.column_stack([x, y, z]).flatten())
    return quat


def project(model, data, cam_id, p, w, h):
    pos = data.cam_xpos[cam_id]
    R = data.cam_xmat[cam_id].reshape(3, 3)
    v = R.T @ (np.asarray(p) - pos)
    f = (h / 2) / math.tan(math.radians(model.cam_fovy[cam_id]) / 2)
    return [round(float((w / 2 + f * v[0] / -v[2]) / w), 4), round(float((h / 2 - f * v[1] / -v[2]) / h), 4)]


def main() -> None:
    out_dir, w, h = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    out_dir.mkdir(parents=True, exist_ok=True)
    model, data = build(w, h)
    rng = FixedRng()
    left, right = Arm(model, data, "left"), Arm(model, data, "right")

    # Left: approach, pinch the flap and swing the lid open; stop there and keep holding it.
    wl = open_box_waypoints(model, data, rng, 1.0)[:10]
    sl = plan_waypoints(model, data, left, wl)
    wr = pick_place_waypoints(model, data, "red_marker", rng, 1.0)
    sr = plan_waypoints(model, data, right, wr)
    assert sl is not None and sr is not None, "twin layout unreachable"
    kl, gl = keyframes(left, wl, sl, LEFT_START, SPEED_LEFT)
    right_start = kl[-1][1]
    kr, gr = keyframes(right, wr, sr, right_start, SPEED_RIGHT)

    lid_adr = model.jnt_qposadr[model.joint("lid_hinge").id]
    marker = model.body("red_marker")
    m_adr = model.jnt_qposadr[model.joint("red_marker_free").id]
    marker_rest = data.qpos[m_adr:m_adr + 7].copy()
    site_r = model.site("right_gripperframe").id
    site_l = model.site("left_gripperframe").id
    cam = model.camera("cine").id
    renderer = mujoco.Renderer(model, height=h, width=w)
    portrait = h > w
    attach, released, release_pose = None, None, None
    floor_z = C.BOX["wall"] + C.MARKER["cap_radius"]
    track = []

    only = {int(x) for x in sys.argv[4].split(",")} if len(sys.argv) > 4 else None  # preview: render these frames
    for k in range(int(DUR * FPS)):
        t = k / FPS
        ql, gpl, il, s_l = pose_at(kl, t)
        qr, gpr, ir, _ = pose_at(kr, t)
        set_arm(model, data, left, ql, gpl, gl)
        set_arm(model, data, right, qr, gpr, gr)
        # The lid follows the left gripper through the six arc waypoints (indices 4-9).
        arc = 0.0 if il < 4 else min(6.0, (il - 4) + s_l) if il <= 9 else 6.0
        if il >= 9 and t >= kl[-1][1]:
            arc = 6.0
        data.qpos[lid_adr] = math.radians(LID_DEG * arc / 6)
        mujoco.mj_forward(model, data)

        # Marker: rests until the jaws close (end of waypoint 3), rides the gripper, drops when they open.
        gpos, gmat = data.site_xpos[site_r].copy(), data.site_xmat[site_r].reshape(3, 3).copy()
        if attach is None and (ir > 3 or (ir == 3 and t >= kr[3][1] - 1e-6)):
            mpos, mmat = marker_rest[:3], np.zeros(9)
            mujoco.mju_quat2Mat(mmat, marker_rest[3:])
            attach = (gmat.T @ (mpos - gpos), gmat.T @ mmat.reshape(3, 3))
        if attach is not None and released is None and ir >= 8 and t >= kr[8][0] + 0.15:
            released = t
        if attach is not None and released is None:
            p = gpos + gmat @ attach[0]
            q = np.zeros(4)
            mujoco.mju_mat2Quat(q, (gmat @ attach[1]).flatten())
            release_pose = (p, q)
            data.qpos[m_adr:m_adr + 3], data.qpos[m_adr + 3:m_adr + 7] = p, q
        elif released is not None:
            p, q = release_pose
            box = data.body("box").xpos
            target = np.array([p[0], p[1], box[2] + floor_z])
            # Settle flat on the box floor, keeping the marker's heading.
            rmat = np.zeros(9)
            mujoco.mju_quat2Mat(rmat, q)
            axis = rmat.reshape(3, 3)[:, 0]
            yaw = math.atan2(axis[1], axis[0])
            flat = np.array([math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)])
            if np.dot(flat, q) < 0:
                flat = -flat
            s = smooth((t - released) / 0.3)
            qs = q + (flat - q) * s
            data.qpos[m_adr:m_adr + 3] = p + (target - p) * s
            data.qpos[m_adr + 3:m_adr + 7] = qs / np.linalg.norm(qs)
        else:
            data.qpos[m_adr:m_adr + 7] = marker_rest

        # Camera: slow orbit and push-in around the box.
        u = smooth(t / DUR) * 0.7 + 0.3 * t / DUR
        target = np.array([0.02, -0.15, 0.09])
        az = math.radians(-38 + 46 * u)
        el = math.radians(30 - 8 * u)
        dist = (1.25 if portrait else 0.98) - 0.18 * u
        cpos = target + dist * np.array([math.cos(el) * math.sin(az), -math.cos(el) * math.cos(az), math.sin(el)])
        model.cam_pos[cam] = cpos
        model.cam_quat[cam] = look_at(cpos, target)
        model.cam_fovy[cam] = 46 if portrait else 38
        mujoco.mj_forward(model, data)

        if only is not None and k not in only:
            track.append({})
            continue
        renderer.update_scene(data, camera="cine")
        Image.fromarray(renderer.render()).save(out_dir / f"f{k:04d}.jpg", quality=93)
        track.append({
            "left": project(model, data, cam, data.site_xpos[site_l], w, h),
            "right": project(model, data, cam, data.site_xpos[site_r], w, h),
            "marker": project(model, data, cam, data.xpos[marker.id], w, h),
            "lid": round(math.degrees(data.qpos[lid_adr]), 1),
            "placed": released is not None and t >= released + 0.25,
            "rightActive": t >= right_start,
        })
        if k % 30 == 0:
            print(f"t={t:.1f}s lid={track[-1]['lid']} marker_attached={attach is not None} released={released}",
                  flush=True)
    (out_dir / "track.json").write_text(json.dumps({"fps": FPS, "frames": track}))


if __name__ == "__main__":
    main()
