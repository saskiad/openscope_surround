import unittest

import numpy as np
from numpy import testing as npt

from . import dataset_objects as do

class TestTrialFluorescenceSubsetting(unittest.TestCase):
    def setUp(self):
        self.fluo_matrix = np.array([
            # Trial 0
            [[1, 2],    # Cell 0
             [3, 4],    # Cell 1
             [5, 6]],   # Cell 2
            # Trial 1
            [[7, 8],     # Cell 0
             [9, 10],    # Cell 1
             [11, 12]]   # Cell 2
        ])
        self.trial_fluorescence = do.TrialFluorescence(
            self.fluo_matrix, [0, 1], 1. / 30.
        )

    def test_cell_subset_by_single_int(self):
        # Test whether fluorescence is extracted correctly
        cell_to_extract = 0
        expected_fluo = self.fluo_matrix[:, cell_to_extract, :][:, np.newaxis, :]
        actual_fluo = self.trial_fluorescence.get_cells(cell_to_extract).fluo
        npt.assert_array_equal(expected_fluo, actual_fluo)

        # Test whether cell labels are subsetted correctly
        npt.assert_array_equal(
            [cell_to_extract],
            self.trial_fluorescence.get_cells(cell_to_extract).cell_vec
        )

    def test_cell_subset_by_pair_of_ints(self):
        # Test whether fluorescence is extracted correctly
        expected_fluo = self.fluo_matrix[:, 0:2, :]
        actual_fluo = self.trial_fluorescence.get_cells(0, 2).fluo
        npt.assert_array_equal(expected_fluo, actual_fluo)

        # Test whether cell labels are subsetted correctly
        npt.assert_array_equal(
            [0, 1],
            self.trial_fluorescence.get_cells(0, 2).cell_vec
        )

    def test_cell_subset_by_tuple_of_ints(self):
        # Test whether fluorescence is extracted correctly
        expected_fluo = self.fluo_matrix[:, 0:2, :]
        actual_fluo = self.trial_fluorescence.get_cells((0, 2)).fluo
        npt.assert_array_equal(expected_fluo, actual_fluo)

        # Test whether cell labels are subsetted correctly
        npt.assert_array_equal(
            [0, 1],
            self.trial_fluorescence.get_cells((0, 2)).cell_vec
        )

    def test_cell_subset_by_bool_mask(self):
        mask = [True, False, True]
        expected_fluo = self.fluo_matrix[:, mask, :]
        actual_fluo = self.trial_fluorescence.get_cells(mask).fluo
        npt.assert_array_equal(expected_fluo, actual_fluo)

        # Test whether cell labels are subsetted correctly
        npt.assert_array_equal(
            [0, 2],
            self.trial_fluorescence.get_cells(mask).cell_vec
        )

    def test_trial_subset_by_single_int(self):
        # Test whether fluorescence is extracted correctly
        trial_to_extract = 0
        expected_fluo = self.fluo_matrix[trial_to_extract, :, :][np.newaxis, :, :]
        actual_fluo = self.trial_fluorescence.get_trials(trial_to_extract).fluo
        npt.assert_array_equal(expected_fluo, actual_fluo)

        # Test whether cell labels are subsetted correctly
        npt.assert_array_equal(
            [trial_to_extract],
            self.trial_fluorescence.get_trials(trial_to_extract).trial_vec
        )

    def test_trial_subset_by_bool_mask(self):
        mask = [False, True]
        expected_fluo = self.fluo_matrix[mask, :, :]
        actual_fluo = self.trial_fluorescence.get_trials(mask).fluo
        npt.assert_array_equal(expected_fluo, actual_fluo)

        # Test whether trial labels are subsetted correctly
        npt.assert_array_equal(
            [1],
            self.trial_fluorescence.get_trials(mask).trial_vec
        )



