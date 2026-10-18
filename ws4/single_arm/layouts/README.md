# Start layouts for the pen eval (held-out takes)

Task: "put the red pen in its slot in the cardboard box" (verbatim, one task string). Reference = frame 0 of each
held-out take of the final split (`ws4_split.json`, 104 takes, 84 train, 12 held out), head camera, as FLUX sees it.
Every held-out take starts with the task arm at haller home (0, ~2, ~1, 0, 0, 99), like the eval.
[overview.jpg](overview.jpg) shows all twelve; `index.json` has the start joints and lengths.

| layout | box (recording log) | attempt 1 | attempt 2 | attempt 3 |
|---|---|---|---|---|
| A | first position | [take 20](A_ep020_head.jpg) | [take 21](A_ep021_head.jpg) | [take 22](A_ep022_head.jpg) |
| B | box and mat shifted right | [take 36](B_ep036_head.jpg) | [take 37](B_ep037_head.jpg) | [take 38](B_ep038_head.jpg) |
| C | box rotated (take 88: box nudged right) | [take 86](C_ep086_head.jpg) | [take 87](C_ep087_head.jpg) | [take 88](C_ep088_head.jpg) |
| D | box shifted left and rotated | [take 101](D_ep101_head.jpg) | [take 102](D_ep102_head.jpg) | [take 103](D_ep103_head.jpg) |

The baseline (1 attempt per layout) uses the attempt-1 reference. In every take the red pen lies on the black mat to
the left of the box (in the head view), and the slot is the empty one next to the green marker in the box.

## Placing

1. `eval_pen.sh` prints `placement page: http://127.0.0.1:8899/`; open it in a browser beside the terminal.
2. The page shows the reference start (left), the live head camera with the reference's edges in green (right) and a
   50/50 blend (bottom), refreshed about 3 times a second while the runner waits for Enter.
3. Push the box until its rim sits on its green outline, then lay the pen on its outline (same end toward the robot).
   The mat and the box lid follow the reference too. The arm is out of the way: at rest before the first attempt,
   at home after that.
4. Hands out of the view, then Enter. The runner saves `start_check.jpg` (reference | live) in the attempt folder and
   prints the base-plate shift; more than a few px means the mast camera moved since recording (stop and say so).

## Scoring (typed after each attempt, then Enter)

- steps done 0 to 3, in order: **reach** = fingers around the pen on the mat; **grasp** = pen lifted off the mat in
  the gripper; **insert** = pen in its slot and released (the dataset's keep rule: held but never released is not
  done). Count only the steps reached in order without help.
- interventions: each stop key press or hand in the workspace counts 1; a timeout with no help is 0.
