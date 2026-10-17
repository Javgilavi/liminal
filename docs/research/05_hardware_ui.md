# WS5: Hardware setup, safety, UI and triggering

## Physical layout

- Two SO-101 followers facing a shared ~40×30 cm workspace, bases clamped. Measure base
  positions in a table frame (metres) → `configs/robot/so101_dual.yaml: bases`.
- One fixed **top/scene** camera (rigid mount, never moved after demo collection) + one
  **wrist** camera per arm.
- Leader arm(s) for teleop demos (WS1) and for recording scripted waypoints.
- Demo props: light hinged lid box, block, bowl, cup (obstacle). Matte, distinct colours
  for the colour-zone detector.

## Bring-up checklist

- [ ] `lerobot-find-port` for each arm; set ports in config (consider udev symlinks)
- [ ] `lerobot-calibrate` each arm with the same `id` as the config
- [ ] `SO101Arm.read_state()` matches the physical pose; `send_action` small moves only
- [ ] measure joint limits per arm → `limits`; set `max_step` conservatively
- [ ] measure a safe `home` pose per arm
- [ ] `hold()` keeps the pose for 60 s with the lid's weight; check servo temperature
- [ ] e-stop: Ctrl-C (CLI), `POST /estop`, a big physical key bound to it

## Triggering and UI

`liminal serve` exposes `POST /run` (`task` or `goal`), `POST /estop`, `GET /state`.
Front-ends on top, in priority order:

1. a single web page: task buttons, live facts, arm owners, last report, a big e-stop
2. voice → speech-to-text → `POST /run {"goal": ...}` (planner picks or generates the tree)
3. a judge-friendly "reset scene" flow between trials

## Risks

- Position holding under load can drift or overheat. Keep lids light and holds short.
- USB bandwidth with 3 cameras at 640×480@30. Test early and lower resolution/fps if frames drop.
- Serial port names change on replug. Use udev rules or by-id paths.
