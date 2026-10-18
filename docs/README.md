# Documentation

Current architecture: task LoRA on the existing SO-101 embodiment, two independent
per-arm FLUX policies, and the real behavior-tree harness. Start with
[the runtime guide](showcase-runtime.md) and [architecture](architecture.md).
The original unified XLeRobot bimanual fine-tune is superseded; research briefs
and experiment reports preserve historical proposals and measurements.

Start with the [first single-arm validation](validation/single-arm-v1.md) for the
recorded results and their limits.

| Goal | Guide |
|---|---|
| Run the project without hardware | [Repository quick start](../README.md#quick-start-no-hardware-needed) |
| Reproduce FLUX inference on the real SO101 | [Official inference](official-inference.md) |
| Generate demos, train and evaluate in MuJoCo | [Single-arm simulation](simulation-validation.md) |
| Identify devices and use the local operator UI | [Robot control](robot-control.md) |
| Install the external controller dependency | [Haller integration](../integrations/haller/README.md) |
| Understand task trees, conditions and ownership | [Architecture](architecture.md) |
| Inspect the earlier standalone inference path | [Legacy inference](live-inference.md) |

The [research briefs](research/) describe workstream goals. They are plans, not
validation results. Single-arm experiments do not establish bimanual or long-horizon
performance.

- [Box assembly runtime and checkpoint registry](showcase-runtime.md)

- [Pinned Oskrt red-pen checkpoint](pen-checkpoint.md)
