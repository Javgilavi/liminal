import numpy as np
from ttt.cup_stages import PHASE_TO_STAGE, STAGES, stage_from_poses


def test_stage_from_poses_reach_carry_in_cup():
    cup = np.tile([0.25, 0.10, 0.0], (6, 1))
    block = np.array([
        [0.20, 0.05, 0.010],  # on the table
        [0.22, 0.07, 0.080],  # lifted
        [0.25, 0.10, 0.120],  # above the cup but higher than the rim: still carried
        [0.25, 0.11, 0.030],  # inside, below the rim
        [0.30, 0.10, 0.060],  # next to the cup, lifted: not in it
        [0.30, 0.10, 0.020],  # dropped next to the cup: back on the table
    ])  # fmt: skip
    assert stage_from_poses(block, cup).tolist() == [0, 1, 1, 2, 1, 0]


def test_every_scripted_phase_has_a_stage():
    from ttt.sim_block_cup import PHASES

    assert set(PHASE_TO_STAGE) == set(PHASES)
    assert set(PHASE_TO_STAGE.values()) == set(range(len(STAGES)))
