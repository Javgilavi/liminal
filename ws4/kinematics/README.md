# ws4/kinematics: SO-101 kinematics, EE50 codec and constrained IK for the XLeRobot station

Owner: ws4-ik. Report and verdict: [`ws4/research/ee50_ik.md`](../research/ee50_ik.md).

| Module | What it does |
|---|---|
| `so101_kinematics.py` | FK of one SO-101 from the official `so101_new_calib.urdf` (checked against MuJoCo to 4 µm), LeRobot unit conversion, fin-ray TCP, `Station` (two arms on the torso, read from `ws4/sim/station_geometry.yaml`) |
| `ee50.py` | EE50 codec: absolute TCP poses <-> 50-D vectors relative to an anchor, 6D rotation from the first two rows, gripper slot, per-offset z-score stats |
| `ik.py` | `AnalyticIK` (closed form + 1-D pitch search, exact position, best reachable orientation, joint limits, elbow branch of the data, 30 Hz continuity) and `DLSIK` (task-priority damped least squares reference) |
| `collision.py`, `mj_collision.py` | capsule screening and MuJoCo mesh collisions on the ws4-sim model |
| `hub_data.py` | downloads and loads small Hub samples (SO-101, bimanual SO-101, XLeRobot, UMI) into `ws4/.cache/kin/hub` |
| `validate.py` | the gate: writes `results.json` and `plots/` |

## Use on the robot (EE50 policy output -> joint commands)

```python
from kinematics import ee50
from kinematics.ik import AnalyticIK
from kinematics.so101_kinematics import Station, rad_to_lerobot_deg, open_to_lerobot_gripper

st = Station.load()                                  # fin-ray TCP + arm mounts from station_geometry.yaml
ik = {s: AnalyticIK(st.arm) for s in ("left", "right")}
anchor = {s: st.arm.fk(q_cmd_rad[s]) for s in ("left", "right")}   # per-arm base frame, last command
poses, grip = ee50.decode(stats.denormalize(z_chunk), anchor)        # (T, 4, 4) per arm, grippers 0..1
q = q_cmd_rad
for t in range(len(z_chunk)):
    for s in ("left", "right"):
        r = ik[s].solve(poses[s][t], q[s])             # exact position, projected orientation
        q[s] = r.q                                     # then: safety clamp, rad_to_lerobot_deg, send
```

## Commands

```bash
python -m pytest ws4/kinematics/tests -q            # CPU, no network; MuJoCo/cache tests skip if absent
ruff check ws4/kinematics
cd ws4 && python -I -m kinematics.hub_data          # one-off: ~50 MB of parquet into ws4/.cache/kin/hub
python -I ws4/kinematics/validate.py                # ~3 min on 16 cores; --quick for a smoke run
```
