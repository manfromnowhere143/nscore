import sys
import os
from pathlib import Path

sys.path.append('/home/dsnyder5/Documents/Github/nscore/')

import numpy as np
import pytest

from statistical_comparison_core import Decision, Hypothesis
from nscore.nsm import BernoulliNsmTest

paper_data_path = str(
    Path(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "../eval_data/",
        )
    ).resolve()
)

# Part 1: Simulation Data (N=200 per policy per task)
eval_dump_veg = np.load(
    f"{paper_data_path}/LBM/Part1/DumpVegetablesFromSmallToLargeContainer.npy"
)
eval_cont_plate = np.load(
    f"{paper_data_path}/LBM/Part1/PutContainersOnPlate.npy"
)
eval_cover_with_plate = np.load(
    f"{paper_data_path}/LBM/Part1/PutFruitInLargeContainerAndCoverWithPlate.npy"
)
eval_separate_foods = np.load(
    f"{paper_data_path}/LBM/Part1/SeparateFruitsVegetablesIntoContainers.npy"
)
eval_upside_down = np.load(
    f"{paper_data_path}/LBM/Part1/TurnLargeContainerUpsideDown.npy"
)

# Part 2: Hardware Data (N=50 per policy per task)
eval_bike_rotor = np.load(
    f"{paper_data_path}/LBM/Part2/BikeRotorInstall.npy"
)
eval_clean_litter = np.load(
    f"{paper_data_path}/LBM/Part2/CleanLitterBox.npy"
)
eval_clear_counter = np.load(
    f"{paper_data_path}/LBM/Part2/ClearKitchenCounter.npy"
)
eval_slice_apple = np.load(
    f"{paper_data_path}/LBM/Part2/CutAppleIntoSlices.npy"
)
eval_set_table = np.load(
    f"{paper_data_path}/LBM/Part2/SetUpBreakfastTable.npy"
)

##### Binary NSM Test #####
@pytest.fixture(scope="module")
def bin_nsm(request):
    test = BernoulliNsmTest(
        alternative=request.param, 
        alpha=0.01,
        c=np.arange(2), 
        verbose=False, 
    )
    return test

@pytest.mark.parametrize(
    ("bin_nsm"),
    [(Hypothesis.P0LessThanP1), (Hypothesis.P0MoreThanP1)],
    indirect=["bin_nsm"],
)
def test_binary_nsm_input_value_error(bin_nsm):
    # Should raise a ValueError if non-binary data is given.
    with pytest.raises(ValueError):
        bin_nsm.step(1.2, 1)
    with pytest.raises(ValueError):
        bin_nsm.step(1, 1.2)

    # Should raise a ValueError if input sequences have length greater than 1
    with pytest.raises(ValueError):
        bin_nsm.step([0.0, 0.0], [1.0, 1.0, 1.0])
    with pytest.raises(ValueError):
        bin_nsm.step([1.0, 1.0, 1.0], [0.0, 0.0])


@pytest.mark.parametrize(
    ("bin_nsm", "sequence_0", "sequence_1", "expected"),
    [
        # fmt: off
        (Hypothesis.P0LessThanP1, [0, 0, 0], [1, 1, 1], Decision.FailToDecide),
        (Hypothesis.P0MoreThanP1, [0, 0, 0], [1, 1, 1], Decision.FailToDecide),
        (Hypothesis.P0LessThanP1, [1, 1, 1], [0, 0, 0], Decision.FailToDecide),
        (Hypothesis.P0MoreThanP1, [1, 1, 1], [0, 0, 0], Decision.FailToDecide),
        (Hypothesis.P0LessThanP1, np.zeros(15), np.ones(15), Decision.AcceptAlternative),
        (Hypothesis.P0MoreThanP1, np.zeros(15), np.ones(15), Decision.FailToDecide),
        (Hypothesis.P0LessThanP1, np.ones(15), np.zeros(15), Decision.FailToDecide),
        (Hypothesis.P0MoreThanP1, np.ones(15), np.zeros(15), Decision.AcceptAlternative),
        # fmt: on
    ],
    indirect=["bin_nsm"],
)
def test_bin_nsm(bin_nsm, sequence_0, sequence_1, expected):
    result = bin_nsm.run_on_sequence(sequence_0, sequence_1)
    assert result.decision == expected


@pytest.fixture(scope="module")
def bin_nsm_lbm(request):
    test = BernoulliNsmTest(
        alternative=request.param,
        alpha=0.05,
        c=np.arange(2), 
        verbose=False, 
    )
    return test

@pytest.mark.parametrize(
    ("bin_nsm_lbm", "sequence_0", "sequence_1", "expected"),
    [
        # fmt: off
        (Hypothesis.P0LessThanP1, eval_dump_veg[:, 0], eval_dump_veg[:, 1], 199.5),
        (Hypothesis.P0MoreThanP1, eval_dump_veg[:, 0], eval_dump_veg[:, 1], 199.5),
        (Hypothesis.P0LessThanP1, eval_cont_plate[:, 0], eval_cont_plate[:, 1], 199.5),
        (Hypothesis.P0MoreThanP1, eval_cont_plate[:, 0], eval_cont_plate[:, 1], 199.5),
        (Hypothesis.P0LessThanP1, eval_cover_with_plate[:, 0], eval_cover_with_plate[:, 1], 40),
        (Hypothesis.P0MoreThanP1, eval_cover_with_plate[:, 0], eval_cover_with_plate[:, 1], 199.5),
        (Hypothesis.P0LessThanP1, eval_separate_foods[:, 0], eval_separate_foods[:, 1], 85),
        (Hypothesis.P0MoreThanP1, eval_separate_foods[:, 0], eval_separate_foods[:, 1], 199.5),
        (Hypothesis.P0LessThanP1, eval_upside_down[:, 0], eval_upside_down[:, 1], 199.5),
        (Hypothesis.P0MoreThanP1, eval_upside_down[:, 0], eval_upside_down[:, 1], 199.5),

        (Hypothesis.P0LessThanP1, eval_bike_rotor[:, 0], eval_bike_rotor[:, 1], 49.5),
        (Hypothesis.P0MoreThanP1, eval_bike_rotor[:, 0], eval_bike_rotor[:, 1], 49.5),
        (Hypothesis.P0LessThanP1, eval_clean_litter[:, 0], eval_clean_litter[:, 1], 49.5),
        (Hypothesis.P0MoreThanP1, eval_clean_litter[:, 0], eval_clean_litter[:, 1], 49.5),
        (Hypothesis.P0LessThanP1, eval_clear_counter[:, 0], eval_clear_counter[:, 1], 49.5),
        (Hypothesis.P0MoreThanP1, eval_clear_counter[:, 0], eval_clear_counter[:, 1], 25),
        (Hypothesis.P0LessThanP1, eval_slice_apple[:, 0], eval_slice_apple[:, 1], 49.5),
        (Hypothesis.P0MoreThanP1, eval_slice_apple[:, 0], eval_slice_apple[:, 1], 49.5),
        (Hypothesis.P0LessThanP1, eval_set_table[:, 0], eval_set_table[:, 1], 49.5),
        (Hypothesis.P0MoreThanP1, eval_set_table[:, 0], eval_set_table[:, 1], 17),
        # fmt: on
    ],
    indirect=["bin_nsm_lbm"],
)
def test_bin_nsm_lbm_time(bin_nsm_lbm, sequence_0, sequence_1, expected):
    result = bin_nsm_lbm.run_on_sequence(sequence_0, sequence_1)
    assert np.abs(result.info["Time"] - expected) <= 0.6