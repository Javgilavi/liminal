"""Write a per-episode variation plan for demo collection (WS1)."""

import argparse

from liminal.data.protocol import make_plan, write_plan

ap = argparse.ArgumentParser()
ap.add_argument("--episodes", type=int, default=60)
ap.add_argument("--zones", nargs="+", default=["front_left", "front_right", "center", "back_left", "back_right"])
ap.add_argument("--instruction", action="append",
                default=None, help="repeatable; paraphrases of the task instruction")
ap.add_argument("--out", required=True)
args = ap.parse_args()

instructions = args.instruction or ["Pick up the red block and place it in the bowl"]
plans = make_plan(args.episodes, instructions, args.zones)
write_plan(plans, args.out)
for p in plans[:5]:
    print(p)
print(f"... {len(plans)} episodes -> {args.out}")
