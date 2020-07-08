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


class TestTrialFluorescenceSummaryStatistics(unittest.TestCase):
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

    def test_trial_mean(self):
        expected = self.fluo_matrix.mean(axis=0)[np.newaxis, :, :]
        actual = self.trial_fluorescence.trial_mean().fluo
        npt.assert_allclose(
            actual, expected, err_msg='Trial mean not correct to within tol.'
        )

    def test_trial_std(self):
        expected = self.fluo_matrix.std(axis=0)[np.newaxis, :, :]
        actual = self.trial_fluorescence.trial_std().fluo
        npt.assert_allclose(
            actual, expected, err_msg='Trial std not correct to within tol.'
        )

    def test_trial_num_isnan_after_mean(self):
        tr_mean = self.trial_fluorescence.trial_mean()
        self.assertEqual(
            len(tr_mean.trial_vec),
            1,
            'Expected only 1 trial after taking mean.'
        )
        self.assertTrue(
            np.isnan(tr_mean.trial_vec[0]),
            'Expected trial_num to be NaN after taking mean across trials'
        )

    def test_trial_num_isnan_after_std(self):
        tr_mean = self.trial_fluorescence.trial_std()
        self.assertEqual(
            len(tr_mean.trial_vec),
            1,
            'Expected only 1 trial after taking std.'
        )
        self.assertTrue(
            np.isnan(tr_mean.trial_vec[0]),
            'Expected trial_num to be NaN after taking std across trials'
        )


class TestTrialFluorescenceIterators(unittest.TestCase):
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

    def test_trial_iterator(self):
        for trial_num, trial_data in self.trial_fluorescence.iter_trials():
            npt.assert_array_equal(
                trial_data.fluo,
                self.fluo_matrix[trial_num, ...][np.newaxis, :, :]
            )

    def test_cell_iterator(self):
        for cell_num, cell_data in self.trial_fluorescence.iter_cells():
            npt.assert_array_equal(
                cell_data.fluo,
                self.fluo_matrix[:, cell_num, :][:, np.newaxis, :]
            )


if __name__ == '__main__':
    unittest.main()
