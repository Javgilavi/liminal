"""Haller dual-arm digital twin: two SO-101 on the tower, D455, round table, mat, box, markers, cutter.

`build(layout)` returns a compiled MuJoCo model for one episode layout (object poses, box pose,
lid angle). The arms, tower and cameras come from `../scene.xml` (audited against haller_ws).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from . import config as C

HERE = Path(__file__).resolve().parent
SO101 = (HERE.parent / "so101" / "so101.xml").resolve()
JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
REST_DEG = (0.0, -60.0, 60.0, 30.0, 0.0, 30.0)  # arm joints in degrees, gripper in %


@dataclass
class Layout:
    """Poses for one episode. Object poses are (x, y, yaw) on the table/mat surface."""
    box: tuple = (C.BOX["center"][0], C.BOX["center"][1], 0.0)
    lid_deg: float = 0.0  # 0 = closed, C.BOX["lid_open_deg"] = open
    objects: dict = field(default_factory=lambda: {
        "green_marker": (0.14, -0.17, 0.3), "red_marker": (0.22, -0.20, -0.2), "cutter": (0.20, -0.29, 0.1)})
    light_scale: float = 1.0
    camera_jitter: tuple = (0.0, 0.0, 0.0, 0.0)  # dx, dy, dz (m), dpitch (rad) for the D455


def _rgba(c):
    return " ".join(f"{v:.3f}" for v in c)


def _box_xml(layout: Layout) -> str:
    b = C.BOX
    (cx, cy, yaw), (ox, oy, oz), w = layout.box, b["outer"], b["wall"]
    tape = _rgba(C.TAPE_RGBA)
    white = _rgba(C.BOX_RGBA)
    hinge_y = oy / 2  # box frame: hinge on the +y side (toward the robot)
    ly = oy + b['lid_overhang']  # lid length from the hinge
    lid_open = math.radians(b["lid_open_deg"])
    return f"""
    <body name="box" pos="{cx} {cy} 0" euler="0 0 {yaw}">
      <geom name="box_floor" type="box" size="{ox/2} {oy/2} {w/2}" pos="0 0 {w/2}" rgba="{white}" mass="{b['mass']*0.4}"/>
      <geom name="box_wall_front" type="box" size="{ox/2} {w/2} {oz/2}" pos="0 {-oy/2+w/2} {oz/2}" rgba="{white}" mass="{b['mass']*0.15}"/>
      <geom name="box_wall_back" type="box" size="{ox/2} {w/2} {oz/2}" pos="0 {oy/2-w/2} {oz/2}" rgba="{white}" mass="{b['mass']*0.15}"/>
      <geom name="box_wall_left" type="box" size="{w/2} {oy/2-w} {oz/2}" pos="{-ox/2+w/2} 0 {oz/2}" rgba="{white}" mass="{b['mass']*0.15}"/>
      <geom name="box_wall_right" type="box" size="{w/2} {oy/2-w} {oz/2}" pos="{ox/2-w/2} 0 {oz/2}" rgba="{white}" mass="{b['mass']*0.15}"/>
      <geom name="box_tape_front" type="box" size="{ox*0.35} 0.0005 {oz*0.45}" pos="0 {-oy/2-0.0005} {oz/2}" rgba="{tape}" contype="0" conaffinity="0" mass="0"/>
      <geom name="cradle_a" type="box" size="0.05 0.012 0.008" pos="{-ox*0.18} 0.03 {w+0.008}" rgba="{white}" mass="0.002"/>
      <geom name="cradle_b" type="box" size="0.05 0.012 0.008" pos="{-ox*0.18} -0.03 {w+0.008}" rgba="{white}" mass="0.002"/>
      <body name="lid" pos="0 {hinge_y} {oz}">
        <joint name="lid_hinge" type="hinge" axis="-1 0 0" range="0 {lid_open + 0.15}" damping="{b['hinge_damping']}"
               frictionloss="{b['hinge_frictionloss']}" armature="0.0005"/>
        <geom name="lid_plate" type="box" size="{ox/2} {ly/2} {b['lid_thickness']/2}" pos="0 {-ly/2} {b['lid_thickness']/2}"
              rgba="{white}" mass="{b['lid_mass']*0.8}" friction="1.0 0.02 0.001"/>
        <geom name="lid_flap" type="box" size="{ox/2*0.9} {b['lid_thickness']/2} 0.012" pos="0 {-ly+b['lid_thickness']/2} -0.012"
              rgba="{white}" mass="{b['lid_mass']*0.2}" friction="1.2 0.02 0.001"/>
        <geom name="lid_tape" type="box" size="{ox*0.22} {oy*0.3} 0.0005" pos="0.02 {-ly/2} {b['lid_thickness']+0.0005}"
              rgba="{tape}" contype="0" conaffinity="0" mass="0"/>
        <site name="lid_flap_site" pos="0 {-ly+b['lid_thickness']/2} -0.006" size="0.004"/>
      </body>
      <site name="box_inside" pos="0 0 {w+0.01}" size="0.004"/>
    </body>"""


def _object_xml(name: str, pose) -> str:
    x, y, yaw = pose
    # Codex's validated marker contact: condim 6 adds torsional/rolling friction so a held cylinder
    # cannot spin out of the rigid jaws; priority makes the object's contact parameters win.
    contact = 'condim="6" friction="0.8 0.003 0.0001" priority="1"'
    if name == "cutter":
        sx, sy, sz = C.CUTTER["size"]
        z = C.MAT["size"][2] + sz / 2
        return f"""
    <body name="{name}" pos="{x} {y} {z}" euler="0 0 {yaw}">
      <freejoint name="{name}_free"/>
      <geom name="{name}_grip" type="box" size="{sx/2} {sy/2} {sz/2}" rgba="{_rgba(C.CUTTER['rgba'])}" mass="{C.CUTTER['mass']}" {contact}/>
      <geom name="{name}_blade" type="box" size="{sx*0.12} {sy*0.32} {sz*0.08}" pos="{sx/2+sx*0.1} 0 0" rgba="0.75 0.75 0.78 1" mass="0.003" contype="0" conaffinity="0"/>
      <geom name="{name}_slider" type="box" size="0.012 {sy*0.3} 0.002" pos="{sx*0.15} 0 {sz/2+0.001}" rgba="0.1 0.1 0.1 1" mass="0" contype="0" conaffinity="0"/>
    </body>"""
    m = C.MARKER
    spec = C.MARKERS[name]
    z = C.MAT["size"][2] + m["cap_radius"]
    half = m["length"] / 2
    cap0 = half - m["cap_length"]
    return f"""
    <body name="{name}" pos="{x} {y} {z}" euler="0 0 {yaw}">
      <freejoint name="{name}_free"/>
      <geom name="{name}_body" type="cylinder" fromto="{-half} 0 0 {cap0} 0 0" size="{m['radius']}" rgba="{_rgba(spec['body_rgba'])}" mass="{m['mass']*0.7}" {contact}/>
      <geom name="{name}_cap" type="cylinder" fromto="{cap0} 0 0 {half} 0 0" size="{m['cap_radius']}" rgba="{_rgba(spec['cap_rgba'])}" mass="{m['mass']*0.3}" {contact}/>
      <geom name="{name}_label" type="cylinder" fromto="{-half*0.6} 0 0 {-half*0.1} 0 0" size="{m['radius']+0.0003}" rgba="0.92 0.92 0.92 1" mass="0" contype="0" conaffinity="0"/>
    </body>"""


def scene_xml(layout: Layout) -> str:
    t, mat = C.TABLE, C.MAT
    dx, dy, dz, dpitch = layout.camera_jitter
    # D455 from ../scene.xml: pos (0, 0.21, 0.46), looking toward -y and down.
    pitch = math.atan2(0.6654, 0.7465) + dpitch
    cam_y = (0.0, -math.cos(pitch), math.sin(pitch))
    ls = layout.light_scale
    objects = "".join(_object_xml(n, p) for n, p in layout.objects.items())
    tower = _rgba(C.TOWER_RGBA)
    return f"""
<mujoco model="haller_twin">
  <compiler angle="radian"/>
  <option timestep="0.002" gravity="0 0 -9.81" cone="elliptic" impratio="10"/>
  <visual>
    <quality shadowsize="2048"/>
    <global offwidth="1280" offheight="960"/>
    <headlight ambient="{0.25*ls} {0.25*ls} {0.25*ls}" diffuse="{0.3*ls} {0.3*ls} {0.3*ls}" specular="0 0 0"/>
  </visual>
  <asset>
    <model name="so101" file="{SO101}"/>
  </asset>
  <worldbody>
    <light name="ceiling" pos="0.2 -0.4 1.6" dir="-0.1 0.2 -1" diffuse="{0.55*ls} {0.55*ls} {0.53*ls}" castshadow="true"/>
    <light name="window" pos="-1.0 -1.2 1.0" dir="0.6 0.7 -0.4" diffuse="{0.25*ls} {0.25*ls} {0.27*ls}" castshadow="false"/>
    <geom name="floor" type="plane" size="3 3 0.05" pos="0 0 -0.75" rgba="0.82 0.80 0.76 1"/>
    <geom name="wall_far" type="box" size="3 0.02 1.5" pos="0 -1.6 0.5" rgba="0.94 0.93 0.90 1" contype="0" conaffinity="0"/>
    <geom name="wall_side" type="box" size="0.02 3 1.5" pos="-1.8 0 0.5" rgba="0.90 0.88 0.84 1" contype="0" conaffinity="0"/>
    <geom name="table_top" type="cylinder" size="{t['radius']} 0.012" pos="{t['center'][0]} {t['center'][1]} -0.012" rgba="{_rgba(C.TABLE_RGBA)}"/>
    <geom name="table_leg" type="cylinder" size="0.03 0.37" pos="{t['center'][0]} {t['center'][1]} -0.39" rgba="{_rgba(C.TABLE_RGBA)}" contype="0" conaffinity="0"/>
    <geom name="mat" type="box" size="{mat['size'][0]/2} {mat['size'][1]/2} {mat['size'][2]/2}" pos="{mat['center'][0]} {mat['center'][1]} {mat['size'][2]/2}" rgba="{_rgba(C.MAT_RGBA)}" friction="1.3 0.02 0.002"/>
    <body name="tower">
      <geom name="tower_plate" type="box" size="0.26 0.09 0.05" pos="0 0.03 0.05" rgba="{tower}"/>
      <geom name="tower_mast" type="box" size="0.025 0.025 0.25" pos="0 0.25 0.33" rgba="{tower}" contype="0" conaffinity="0"/>
      <geom name="tower_head" type="box" size="0.06 0.03 0.02" pos="0 0.21 0.49" rgba="{tower}" contype="0" conaffinity="0"/>
      <geom name="d455" type="box" size="0.062 0.013 0.013" pos="0 0.205 0.465" rgba="0.92 0.92 0.92 1" contype="0" conaffinity="0"/>
      <camera name="top" pos="{dx} {0.21+dy} {0.46+dz}" xyaxes="-1 0 0 0 {cam_y[1]:.4f} {cam_y[2]:.4f}" fovy="65"/>
    </body>
    <frame name="left_mount" pos="-0.15 0 0.10" euler="0 0 -1.5708">
      <attach model="so101" body="base" prefix="left_"/>
    </frame>
    <frame name="right_mount" pos="0.15 0 0.10" euler="0 0 -1.5708">
      <attach model="so101" body="base" prefix="right_"/>
    </frame>
    {_box_xml(layout)}
    {objects}
    <camera name="front" pos="0 -0.85 0.62" xyaxes="1 0 0 0 0.6529 0.7574" fovy="46"/>
  </worldbody>
</mujoco>"""


def build(layout: Layout | None = None):
    layout = layout or Layout()
    model = mujoco.MjModel.from_xml_string(scene_xml(layout))
    # The CAD materials are yellow; the real arms are printed in green.
    for i in range(model.nmat):
        r, g, b, _ = model.mat_rgba[i]
        if r > 0.8 and g > 0.6 and b < 0.3:
            model.mat_rgba[i] = C.ARM_RGBA
    data = mujoco.MjData(model)
    # Both arms start at, and hold, the collision-free working pose used by the simulator app.
    for arm in ("left", "right"):
        for j, v in zip(JOINTS, REST_DEG):
            jid = model.joint(f"{arm}_{j}").id
            lo, hi = model.jnt_range[jid]
            value = math.radians(v) if j != "gripper" else lo + v / 100 * (hi - lo)
            data.qpos[model.jnt_qposadr[jid]] = value
            data.ctrl[model.actuator(f"{arm}_{j}").id] = value
    lid = model.joint("lid_hinge")
    data.qpos[lid.qposadr[0]] = math.radians(layout.lid_deg)
    mujoco.mj_forward(model, data)
    return model, data
