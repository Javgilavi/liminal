"""WS2: generate a clear-then-act task tree from a scene description and print it."""

import argparse

import yaml

from liminal.bimanual import ArmBase, Scene, SceneObject, plan_clear_and_act

ap = argparse.ArgumentParser()
ap.add_argument("--strategy", default="push_aside", choices=["push_aside", "pick_aside", "hold_away"])
args = ap.parse_args()

scene = Scene(
    arms=[ArmBase("left", (-0.20, 0.0)), ArmBase("right", (0.20, 0.0))],
    objects=[
        SceneObject("cup", (-0.05, 0.15), args.strategy, blocks=["red_block"]),
        SceneObject("red_block", (0.05, 0.20)),
    ],
)
spec = plan_clear_and_act(scene, "red_block", "Pick up the red block and place it in the bowl", goal="block_in_bowl")
print(yaml.safe_dump(spec, sort_keys=False))
