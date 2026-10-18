# XLeRobot 0.4.0 torso, neck and head meshes

Derived from XLeRobot (Apache-2.0, see LICENSE in this folder):
https://github.com/Vector-Wangel/XLeRobot @ b017b5e6354bd9f61f4247a920c72622ca0aade0
- `hardware/step/XLeRobot_040/XLeRobot040_armbase.step` (arm bases, turntables, centre shell)
- `hardware/step/RGBD_Gimbal/Assembly_RGBD_Gimbal.step` (octagon neck, pan/tilt head, D435-style pitch holder)

Changes: meshed from STEP with gmsh, moved into the station world frame of ws4/sim/station_geometry.yaml, decimated
(ws4/sim/tools/prepare_assets.py). `cad_frames.json` records the frame offsets and part bounds that the MJCF generator uses.
