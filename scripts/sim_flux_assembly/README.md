# Scripted assembly prototype

Follow-on simulation tooling for a gear/collar task: CAD-derived visuals and
compound collision shapes (`assets.py`), native-coordinate environment and
independent predicates (`scene.py`), and a scripted IK teacher (`teacher.py`).
This is not the learned pick-and-place evaluation or a validated FLUX assembly
policy. No assembly result is claimed in the first single-arm validation.

This prototype requires the prepared single-arm scene, MuJoCo, NumPy and the
external NIST CAD archive expected at `<root>/assets/source/nist-stl.zip`.
Set `LIMINAL_SIM_ROOT` to the prepared scene. Run each script with `--help` for
its required root/output arguments. CAD asset acquisition and a complete learned
assembly pipeline are not yet packaged here.

See [the first validation](../../docs/validation/single-arm-v1.md) and
[simulation guide](../../docs/simulation-validation.md) for the supported experiment.
