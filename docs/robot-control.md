# Bimanual XLeRobot control

The controller lives in Liminal: no Haller checkout, ROS installation, Node.js,
GPU, or model weights are required. The existing orchestration API and manual UI
share one `SafetySupervisor` and one connection per arm.

## Install and start

Linux, Python 3.12, [uv](https://docs.astral.sh/uv/getting-started/installation/),
and `fuser` (`psmisc`) are required. Your account needs access to the serial ports
and cameras, normally via the `dialout` and `video` groups.

```bash
cd ~/liminal
./scripts/setup_robot.sh       # isolated .venv-robot; CPU PyTorch; no hardware access
./scripts/control_robot.sh     # http://127.0.0.1:8080
```

**Stop any existing Haller/LeRobot controller first.** The launcher never kills
another process or takes over an occupied serial port. Inspect ownership with
`fuser /dev/ttyACM0 /dev/ttyACM1`. Ctrl+C stops Liminal and releases both arms.

For a preview without hardware:

```bash
.venv-robot/bin/liminal serve --robot configs/robot/mock_dual.yaml --port 8080
```

The page shows joint positions, motor temperature, arm ownership, the scene
camera, task selection, manual jogs, and E-stop. Support an arm, click **Engage at
current pose**, then jog a single joint. **Free-drive** releases that arm.
**E-STOP** releases both arms; they can sag. **Reset E-stop** does not engage them.

## This rig

`configs/robot/xlerobot.yaml` records the board serials observed on 2026-10-17:

| Expected arm, facing the robot | USB board serial |
|---|---|
| left | `5B79018460` |
| right | `5B14030445` |

Use `/dev/serial/by-id` paths, not enumeration-dependent `/dev/ttyACM*` names.
After rewiring, verify the physical sides with torque off. Board identity alone
does not prove which arm is connected. The D455 RGB camera uses a USB-port path;
update it after changing USB sockets. Wrist camera sides are unverified on this
host and are intentionally omitted. Add verified cameras under `cameras` using
`type: opencv` and `index: /dev/v4l/by-path/...`.

On connect, the adapter reads calibration from servo EEPROM and saves it under
`~/.cache/huggingface/lerobot/calibration/robots/so_follower/<id>.json`. An existing
file must match the hardware. It never writes calibration, PID gains, or motor
protection settings. Missing/invalid calibration or inadequate gripper protection
blocks startup; use LeRobot's calibration/setup procedure separately with the
arm supported. Do not copy another arm's calibration to clear an error.

## Safety and limits

- Startup leaves torque **off** and refuses an already-energized or occupied bus.
- Engage parks raw goal registers at a stable, measured pose before enabling torque.
- All policy and UI actions go through `SafetySupervisor.send()`. Hardware also
  bounds speed to 20 degrees/s (20 percentage points/s for the gripper), with a
  0.1-second budget cap. The rig permits at most 2 units per command.
  A failed state read or action write E-stops both arms.
- Software joint limits intersect stored calibration. Wrist roll is restricted to
  ±90° because the existing calibration spans almost a full turn; endpoints have
  not been physically verified. No guessed home pose is provided.
- A per-arm watchdog checks temperature, voltage and alarms every second. At 55°C,
  an alarm, or lost health telemetry it latches a fault and attempts torque release
  on every motor, then asks the supervisor to E-stop the other arm. A fault requires
  inspection and controller restart; no automatic re-engagement occurs.
- Shutdown and E-stop attempt all motors even when one refuses. If torque release
  is unconfirmed, **cut motor power**. Software release cannot clear every servo fault.
- There is **no geometric collision detection**. Keep the workspace clear, watch
  both arms, and keep loads light. These limits do not certify autonomous safety.

Existing task execution remains available with `--backend scripted` or `flux`.
Record and validate waypoints and a home pose before real autonomous tasks. FLUX
loading remains the existing stub. CLI task execution on real arms additionally
requires the explicit `liminal run ... --engage` flag; connect alone never arms them.

## Small API

| Endpoint | Purpose |
|---|---|
| `GET /state`, `GET /tasks` | Live arms, health, ownership and task reports; task list |
| `POST /arms/{left\|right}/torque` | `{"enabled": true}` parks and engages; `false` releases |
| `POST /arms/{left\|right}/goal` | `{"positions": [six values]}` sends one limited step |
| `POST /run` | `{"task": "name"}` starts an existing task; arms must already be engaged |
| `POST /estop`, `POST /reset` | Latched release; explicit reset without torque enable |
| `GET /cameras/{name}.jpg` | Camera snapshot |

Joint order: `shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll,
gripper`. Five joints use **degrees**; gripper uses **0–100%**. Manual controls
refuse commands while a task owns an arm or telemetry is stale. The server binds
to loopback by default and has no login: keep it local.

Run `./scripts/check_robot.sh` for tests and lint in the isolated environment.
Tests use mock arms/fake buses;
they never access serial ports or physical cameras. Safety behavior was adapted
from [Haller](https://github.com/oscardvs/haller_ws) (`b46b827`), with Linux device
mapping and LeRobot 0.5.1 confirmed in the current setup. Transport regressions are
tested against fake buses; motion and physical collision clearance are not tested.
