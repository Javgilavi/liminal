# SO101 pick-and-place scene

Versioned scene/model and camera setup used for the first single-arm validation.
See [simulation reproduction](../../docs/simulation-validation.md).

`so101.xml` derives from `google-deepmind/mujoco_menagerie/robotstudio_so101` at
`62ab00f2bc44f6f20c1c9b7a13f3fb1dd41a4e35`. The upstream Apache 2.0
[license](LICENSE) and [changelog](upstream-changelog.md) are retained. Local
changes include the physics/contact setup, wrist view and experiment actuator
settings. Mesh files are external; the compiler uses a relative `assets` directory.

`scene.xml` adds the table, red block and blue container. `official_camera_setup.py`
sets an approximate BFL-style scene view and a Menagerie wrist optical pose with
lens clearance. Its name refers to reference layout roles; the scene extrinsics
are not an official calibrated BFL camera model.

`scripts/sim_flux_lora/prepare_scene.py` copies these definitions into an output
directory and links meshes from the pinned external checkout. It refuses an
unmatched revision. No third-party mesh download or physics simulation happens
at module import.
