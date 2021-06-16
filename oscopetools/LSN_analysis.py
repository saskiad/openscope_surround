#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 02:35:35 2020

@author: kailun
"""

import numpy as np
import matplotlib.pyplot as plt
from oscopetools import read_data as rd
from .adjust_stim import *
from .chi_square_lsn import chi_square_RFs
from .greedy_pixelwise_rf import get_receptive_field_greedy
from enum import Enum
import warnings, sys, os


class LSN_analysis:
    _ON_stim_value = 255
    _OFF_stim_value = 0
    _background_stim_value = 127
    _yx_ref = None  # The reference y- and x-positions used for correcting the LSN stimulus array.
    _stim_size_deg = (
        9.3  # The side length of the stimulus in degree (same unit as eye pos).
    )
    _frame_rate_Hz = 30  # The frame rate of fluorescent responses in Hz.
    _CS_center_diameter_deg = (
        30  # The diameter in degrees of the center-surround stimulus' center.
    )

    def __init__(
        self,
        datafile_path,
        LSN_stim_path,
        num_baseline_frames=None,
        use_dff_z_score=False,
        correct_LSN=False,
        use_only_valid_eye_pos=False,
        use_only_positive_responses=False,
        RF_type="Greedy pixelwise RF",
        RF_loc_thresh=0.8,
        verbose=True,
    ):
        """To analyze the locally-sparse-noise-stimulated cell responses.

        Parameters
        ----------
        datafile_path : str
            The path to the data file.
        LSN_stim_path : str
            The path to the LSN stimulus npy file.
        num_baseline_frames : int or None
            The number of baseline frames before the start and after the end of a trial.
        use_dff_z_score : bool
            If True, the cell responses will be converted to z-score before analysis.
        correct_LSN : bool
            If True, the LSN stimulus corrected by eye positions will be used. Otherwise, the original LSN stimulus will be used.
            The stimulus wlll remain unchanged for those frames without valid eye positions.
        use_only_valid_eye_pos : bool
            If True, only stimuli with valid eye positions are used. Otherwise, all stimuli will be used.
        use_only_positive_responses : bool
            If True, the fluorescence responses less than 0 will be set to 0 when computing the avg_responses.
        RF_type : str
            "Greedy pixelwise RF" or "Trial averaged RF". The type of RFs to be computed.
        RF_loc_thresh : float
            The threshold for deciding whether the RF is located within the center or surround or not.
        verbose : bool
            If True, the parameters used will be printed.

        """
        self.datafile_path = datafile_path
        self.LSN_stim_path = LSN_stim_path
        self.num_baseline_frames = num_baseline_frames
        if (self.num_baseline_frames is None) or (self.num_baseline_frames < 0):
            self.num_baseline_frames = 0
        self.is_use_dff_z_score = use_dff_z_score
        self.is_use_corrected_LSN = correct_LSN
        self.is_use_valid_eye_pos = use_only_valid_eye_pos
        self.is_use_positive_fluo = use_only_positive_responses
        self.RF_type = RF_type
        self.RF_loc_thresh = RF_loc_thresh
        self._verbose = verbose
        self.dff_fluo = rd.get_dff_traces(self.datafile_path)
        self.num_cells = self.dff_fluo.num_cells
        self.cell_ids = np.array(rd.get_roi_table(datafile_path).cell_id).tolist()
        self.LSN_stim_table = rd.get_stimulus_table(
            self.datafile_path, "locally_sparse_noise"
        )
        if self.is_use_dff_z_score:
            self.dff_fluo.z_score()
        self.trial_fluo = self.dff_fluo.cut_by_trials(
            self.LSN_stim_table,
            self.num_baseline_frames,
            both_ends_baseline=True,
        )
        self._full_LSN_stim = np.load(self.LSN_stim_path)
        self.eye_tracking = rd.get_eye_tracking(self.datafile_path)
        (
            self._corrected_LSN_stim,
            self.valid_eye_pos,
            self.yx_ref,
        ) = correct_LSN_stim_by_eye_pos(
            self._full_LSN_stim,
            self.LSN_stim_table,
            self.eye_tracking,
            self._yx_ref,
            self._stim_size_deg,
            self._background_stim_value,
        )
        self._all_trial_mask = np.array([True] * self.LSN_stim_table.shape[0])
        self._update_params()
        self._get_CS_center_info()
        self._update_responses()

    def __str__(self):
        return (
            "\nAnalyzing file: {}\n"
            "ON LSN stimulus value: {}\n"
            "OFF LSN stimulus value: {}\n"
            "Background LSN value: {}\n"
            "LSN stimulus size: {} degree\n"
            "Number of cells: {}\n"
            "Current RF type: {}\n"
            "Use DF/F z-score: {}\n"
            "Use corrected LSN: {}\n"
            "Use only valid eye positions: {}\n"
            "Use only positive fluorescence responses: {}"
        ).format(
            self.datafile_path,
            self._ON_stim_value,
            self._OFF_stim_value,
            self._background_stim_value,
            self._stim_size_deg,
            self.num_cells,
            self.RF_type,
            self.is_use_dff_z_score,
            self.is_use_corrected_LSN,
            self.is_use_valid_eye_pos,
            self.is_use_positive_fluo,
        )

    def correct_LSN_by_eye_pos(self, value=True):
        """
        value : bool
            If True, the LSN stimulus corrected by eye positions will be used. Otherwise, the original LSN stimulus will be used.
            The stimulus wlll remain unchanged for those frames without valid eye positions.
        """
        if self.is_use_corrected_LSN == bool(value):
            raise ValueError(
                "LSN stim is already corrected."
                if bool(value)
                else "LSN stim is already original."
            )
        try:
            self.is_use_corrected_LSN = bool(value)
            self._update_responses()
        except:
            print(
                "Failed to change correct_LSN_by_eye_pos to {}! \nRecomputing the responses with correct_LSN_by_eye_pos({})...".format(
                    value, bool(1 - value)
                )
            )
            self.correct_LSN_by_eye_pos(bool(1 - value))

    def use_valid_eye_pos(self, value=True):
        """
        value : bool
            If True, only stimuli with valid eye positions are used. Otherwise, all stimuli will be used.
        """
        if self.is_use_valid_eye_pos == bool(value):
            raise ValueError(
                "The valid eye positions are used."
                if bool(value)
                else "All eye positions are used."
            )
        try:
            self.is_use_valid_eye_pos = bool(value)
            self._update_responses()
        except:
            print(
                "Failed to change use_valid_eye_pos to {}! \nRecomputing the responses with use_valid_eye_pos({})...".format(
                    value, bool(1 - value)
                )
            )
            self.use_valid_eye_pos(bool(1 - value))

    def use_positive_fluo(self, value=True):
        """
        value : bool
            If True, the fluorescence responses less than 0 will be set to 0 when computing the avg_responses.
        """
        if self.is_use_positive_fluo == bool(value):
            raise ValueError(
                "The positive responses are already used."
                if bool(value)
                else "Both positive and negative responses are already used."
            )
        try:
            self.is_use_positive_fluo = bool(value)
            self._update_responses()
        except:
            print(
                "Failed to change use_positive_fluo to {}! \nRecomputing the responses with use_positive_fluo({})...".format(
                    value, bool(1 - value)
                )
            )
            self.use_positive_fluo(bool(1 - value))

    def _update_responses(self):
        self._update_params()
        self.ON_avg_responses = self._compute_avg_pixel_response(
            self.trial_fluo.get_trials(self._trial_mask),
            self.LSN_stim[self._trial_mask],
            self._ON_stim_value,
        )
        self.OFF_avg_responses = self._compute_avg_pixel_response(
            self.trial_fluo.get_trials(self._trial_mask),
            self.LSN_stim[self._trial_mask],
            self._OFF_stim_value,
        )
        if self.RF_type.upper() == "TRIAL AVERAGED RF":
            self.get_trial_avg_RFs()
        elif self.RF_type.upper() == "GREEDY PIXELWISE RF":
            self.get_greedy_RFs()
        else:
            print(
                "Please choose either 'Trial averaged RF' or 'Greedy pixelwise RF' for RF_type."
            )
        if self._is_CS_session:
            self.get_RF_loc_masks(self.RF_loc_thresh)
        if self._verbose:
            print(self)

    def _update_params(self):
        if self.is_use_corrected_LSN:
            self.LSN_stim = self._corrected_LSN_stim
        else:
            self.LSN_stim = self._full_LSN_stim[self.LSN_stim_table.Frame]

        if self.is_use_valid_eye_pos:
            self._trial_mask = self.valid_eye_pos
        else:
            self._trial_mask = self._all_trial_mask

    def _compute_avg_pixel_response(self, trial_response, LSN_stim, target):
        """
        Parameters
        ----------
        trial_response : TrialFluorescence object
            The DF/F trial response.
        LSN_stim : 3d np.array
            LSN stimulus array, shape = (num_frame, ylen, xlen).
        target : int
            The target value (value of interest) in the stimulus array.

        Returns
        -------
        avg_responses : 4d np.array
            The trial-averaged responses within pixel, shape = (num_cells, ylen, xlen, trial_len).
        """
        response = (
            trial_response.positive_part()
            if self.is_use_positive_fluo
            else trial_response
        )
        avg_responses = np.zeros(
            (
                response.num_cells,
                LSN_stim.shape[1],
                LSN_stim.shape[2],
                response.num_timesteps,
            )
        )
        for y in range(LSN_stim.shape[1]):
            for x in range(LSN_stim.shape[2]):
                avg_responses[:, y, x, :] = (
                    response.get_trials(LSN_stim[:, y, x] == target).trial_mean().data
                )
        return avg_responses

    def _get_chi_square_pvals(self, frame_shift, num_shuffles=1000):
        """To do the Chi-square test on the DF/F responses to LSN stimuli.

        Parameters
        ----------
        frame_shift : int
            The frame shift of the window to account for the delay in calcium responses for the Chi-square test.
            Default is 3.

        Creates
        -------
        chi_square_pvals : array-like, 3D
            The p-values from the Chi-square test for each cell. Shape = (num_cells, ylen, xlen).
        """
        assert (
            abs(frame_shift) <= self.num_baseline_frames
        ), "Please use frame_shift with absolute value smaller or equal to num_baseline_frames!"
        if self.is_use_positive_fluo:
            trial_dff = (
                self.trial_fluo.get_trials(self._trial_mask).positive_part().data
            )
        else:
            trial_dff = self.trial_fluo.get_trials(self._trial_mask).data
        stim_trial = trial_dff[
            :,
            :,
            self.num_baseline_frames
            + frame_shift : -self.num_baseline_frames
            + frame_shift,
        ]
        responses = stim_trial.mean(2)
        LSN_template = self.LSN_stim[self._trial_mask]
        with gag():
            self.chi_square_pvals = chi_square_RFs(responses, LSN_template, num_shuffles)

    @staticmethod
    def _remove_non_significant(RF, p_values, significant_lvl=0.05):
        """To remove the non-significant part of the RF.

        Parameters
        ----------
        RF : array-like, 2D
            The receptive field computed by greedy pixelwise approach.
        p_values : array-like, 2D
            The p-values from Chi-square test.
        significant_lvl : float
            The significant level of the Chi-square p-values.

        Returns
        -------
        RF : array-like, 2D
            The receptive field with non-significant parts removed.
        """
        RF = RF.copy()
        non_sig_mask = p_values > significant_lvl
        RF[non_sig_mask] = 0
        return RF

    @staticmethod
    def _normalize_RF(RF):
        """To normalize the receptive field to range from 0 to 1.

        Parameters
        ----------
        RF : array-like
            The receptive field to be normalized.

        Returns
        -------
        RF : array-like
            The normalized RF.
        """
        if np.nanmin(RF) == np.nanmax(RF):
            return np.zeros(RF.shape)
        RF /= np.nanmax(abs(RF))
        return RF

    def get_greedy_RFs(
        self,
        frame_shift=3,
        alpha=0.05,
        sweep_response_type="mean",
        chisq_significant_lvl=0.05,
        norm_RF=False,
    ):
        """To compute the receptive fields using greedy pixelwise approach.

        Parameters
        ----------
        frame_shift : int
            The frame shift of the window to account for the delay in calcium responses for the Chi-square test.
            Default is 3.
        alpha : float
            The significance threshold for a pixel to be included in the RF map.
            This number will be corrected for multiple comparisons (number of pixels).
        sweep_response_type : str
            Choice of 'mean' for mean_sweep_events or 'binary' to make boolean calls of
            whether any events occurred within the sweep window.
        chisq_significant_lvl : float
            The significance threshold of the Chi-square test p-values for the RF pixels to be included.
        norm_RF : bool
            If True, the computed RFs will be normalized to their corresponding max value.

        Creates
        -------
        ON_RFs, OFF_RFs : array-like, 3D
            The ON/OFF receptive subfields. Shape = (num_cells, ylen, xlen).
        """
        self._get_chi_square_pvals(frame_shift)
        self.ON_RFs = []
        self.OFF_RFs = []
        stimulus_table = self.LSN_stim_table.astype(int)
        stimulus_table.columns = ["start", "end", "frame"]
        stimulus_table = stimulus_table[self._trial_mask]
        stimulus_table["start"] = stimulus_table["start"] + frame_shift
        stimulus_table["end"] = stimulus_table["end"] + frame_shift
        LSN_template = self.LSN_stim[self._trial_mask]
        all_L0_events = (
            self.dff_fluo.positive_part().data
            if self.is_use_positive_fluo
            else self.dff_fluo.data
        )

        for idx in range(self.num_cells):
            RF_ON, RF_OFF = get_receptive_field_greedy(
                all_L0_events[idx],
                stimulus_table,
                LSN_template,
                alpha,
                sweep_response_type,
            )
            RF_ON = self._remove_non_significant(
                RF_ON, self.chi_square_pvals[idx], chisq_significant_lvl
            )
            RF_OFF = self._remove_non_significant(
                RF_OFF, self.chi_square_pvals[idx], chisq_significant_lvl
            )
            self.ON_RFs.append(self._normalize_RF(RF_ON) if norm_RF else RF_ON)
            self.OFF_RFs.append(self._normalize_RF(RF_OFF) if norm_RF else RF_OFF)
        self.ON_RFs, self.OFF_RFs = np.array(self.ON_RFs), -np.array(self.OFF_RFs)
        self._integration_window_start = self.num_baseline_frames + frame_shift
        self._integration_window_len = (
            self.ON_avg_responses.shape[-1] - 2 * self.num_baseline_frames
        )
        self.RF_type = "Greedy pixelwise RF"

    def get_trial_avg_RFs(self, threshold=0, window_start=None, window_len=None):
        """To get the ON and OFF RFs and the position of their max response.

        Parameters
        ----------
        threshold : int or float
            Range = [0, 1]. The threshold for the RF, anything below the threshold will be set to 0.
        window_start : int
            The start frame index (within a trial) of the integration window for computing the RFs.
        window_len : int
            The length of the integration window in frames for computing the RFs.

        Creates
        -------
        ON_RFs : 3d np.array
            The ON receptive field array, shape = (num_cells, ylen, xlen).
        OFF_RFs : 3d np.array
            The OFF receptive field array, shape = (num_cells, ylen, xlen).
        ON_RF_peaks_yx : 2d np.array
            The yx-indices of the peak ON responses of each cell, shape = (num_cells, 2).
        OFF_RF_peaks_yx : 2d np.array
            The yx-indices of the peak OFF responses of each cell, shape = (num_cells, 2).
        """
        if window_start is None:
            window_start = self.num_baseline_frames
        if window_len is None:
            window_len = self.ON_avg_responses.shape[-1] - 2 * self.num_baseline_frames
        if window_start + window_len > self.ON_avg_responses.shape[-1]:
            warnings.warn(
                "The integration window [{}:{}] is shifted beyond the trial of length {}!".format(
                    window_start,
                    window_start + window_len,
                    self.ON_avg_responses.shape[-1],
                )
            )
        self._integration_window_start = max(0, int(window_start))
        self._integration_window_len = max(0, int(window_len))
        threshold = max(0, threshold)
        self.ON_RFs = self._compute_RF_subfield(
            "ON",
            threshold,
            self._integration_window_start,
            self._integration_window_len,
        )
        self.OFF_RFs = self._compute_RF_subfield(
            "OFF",
            threshold,
            self._integration_window_start,
            self._integration_window_len,
        )
        self.RF_type = "Trial averaged RF"

    def _get_RF_peaks_yx(self):
        """To get the yx coordinates of max response of the ON and OFF RFs.

        Creates
        -------
        ON_RF_peaks_yx : 2d np.array
            The yx-indices of the peak ON responses of each cell, shape = (num_cells, 2).
        OFF_RF_peaks_yx : 2d np.array
            The yx-indices of the peak OFF responses of each cell, shape = (num_cells, 2).
        """
        ON_cell_peak_idx = self.ON_RFs.reshape(self.ON_RFs.shape[0], -1).argmax(1)
        OFF_cell_peak_idx = self.OFF_RFs.reshape(self.OFF_RFs.shape[0], -1).argmin(1)
        self.ON_RF_peaks_yx = np.column_stack(
            np.unravel_index(ON_cell_peak_idx, self.ON_RFs[0, :, :].shape)
        ).astype(float)
        self.OFF_RF_peaks_yx = np.column_stack(
            np.unravel_index(OFF_cell_peak_idx, self.OFF_RFs[0, :, :].shape)
        ).astype(float)
        for i in range(self.num_cells):
            if self.location_mask_dict["No_ON"][i]:
                self.ON_RF_peaks_yx[i] = [np.nan, np.nan]
            if self.location_mask_dict["No_OFF"][i]:
                self.OFF_RF_peaks_yx[i] = [np.nan, np.nan]

    def _compute_RF_subfield(self, polarity, threshold, window_start, window_len):
        """To compute the ON or OFF subfield given a threshold.

        Parameters
        ----------
        polarity : str
            'ON' or 'OFF'.
        threshold : int or float
            Range = [0, 1]. The threshold for the RF, anything below the threshold will be set to 0.
        window_start : int
            The start index (within a trial) of the integration window for computing the RFs.
        window_len : int
            The length of the integration window in frames for computing the RFs.

        Returns
        -------
        RFs : 3d np.array
            Array containing ON or OFF RFs for all cells, shape = (num_cells, ylen, xlen).
        """
        polarity = ReceptiveFieldPolarity.from_(polarity)
        if polarity == ReceptiveFieldPolarity.ON:
            RFs = self.ON_avg_responses[
                ..., window_start : window_start + window_len
            ].mean(-1)
            pol = 1
        elif polarity == ReceptiveFieldPolarity.OFF:
            RFs = self.OFF_avg_responses[
                ..., window_start : window_start + window_len
            ].mean(-1)
            pol = -1
        else:
            raise ValueError("Please enter 'ON' or 'OFF' for the polarity.")
        RFs -= np.nanmean(RFs, axis=(1, 2))[:, None, None]
        RFs /= np.nanmax(abs(RFs), axis=(1, 2))[:, None, None]
        RFs[RFs < threshold] = 0.0
        RFs *= pol
        return RFs

    def _compute_center_overlap(self, RF_arr, RF_thresh=0, bin_num=1000):
        """Compute the fraction of overlap between ON/OFF receptive subfields and center.

        Parameters
        ----------
        RF_arr : array-like, 2D
            The receptive field. Shape = (ylen, xlen).
        RF_thresh : float
            The threshold of RF, the RF values below the threshold will not be considered. Default is 0.
        bin_num : int
            The number of binning for an LSN pixel when computing the overlapping indices.
            Higher bin_num gives higher precision but will take longer computational time.

        Returns
        -------
        overlapping_index : float
            The overlapping index (fraction) of the RF with the stimulus center.
        """
        RF = abs(np.array(RF_arr).copy())
        RF[RF < RF_thresh] = 0
        RF_ys, RF_xs = np.where(RF > 0)
        total_overlap = 0
        for i, RFy in enumerate(RF_ys):
            RFx = RF_xs[i]
            tmp_ys = np.arange(RFy - 0.5, RFy + 0.5, 1 / bin_num)
            tmp_xs = np.arange(RFx - 0.5, RFx + 0.5, 1 / bin_num)
            xs, ys = np.meshgrid(tmp_xs, tmp_ys)
            tmp_xys = np.vstack((xs.flatten(), ys.flatten())).T
            distances = np.linalg.norm(tmp_xys - self.CS_center_pos_xy_pix, axis=1)
            within_center = distances <= self.CS_center_radius_pix
            overlap_fraction = within_center.sum() / bin_num ** 2
            total_overlap += overlap_fraction * RF[RFy, RFx]
        overlapping_index = total_overlap / RF.sum()
        return overlapping_index

    def _get_center_overlap(self, RF_thresh=0, bin_num=1000):
        """To compute the overlapping index for ON/OFF RFs with inner/outer centers.

        Parameters
        ----------
        RF_thresh : float or int
            The threshold of RFs to be considered. Default is 0.
        bin_num : int
            The number of binning for an LSN pixel when computing the overlapping indices.
            Higher bin_num gives higher precision but will take longer computational time.

        Creates
        -------
        ON_overlap_idx, OFF_overlap_idx : list
            List containing overlapping indices for the ON and OFF RFs with the CS center.
        """
        overlapping_idx_ONOFF = []
        for RFs in [self.ON_RFs, self.OFF_RFs]:
            sublst = []
            for RF in RFs:
                overlap_idx = self._compute_center_overlap(RF, RF_thresh, bin_num)
                sublst.append(overlap_idx)
            overlapping_idx_ONOFF.append(sublst)
        self.ON_overlap_idx, self.OFF_overlap_idx = np.array(overlapping_idx_ONOFF)

    def _get_CS_center_shift(self):
        """
        Creates
        -------
        _center_shift_xy_deg, center_shift_xy_pix : array-like, 1D
            The x- and y-shifts of the CS center relative to the center of the monitor in degrees and LSN pixels.
        """
        with gag():
            stim_table = rd.get_stimulus_table(self.datafile_path, "center_surround")
        center_xs = np.array(stim_table.Center_x)
        center_ys = np.array(stim_table.Center_y)
        is_same_x = center_xs.min() == center_xs.max()
        is_same_y = center_ys.min() == center_ys.max()
        all_same = is_same_x & is_same_y
        if all_same:
            self._center_shift_xy_deg = np.array([center_xs[0], center_ys[0]])
            self._center_shift_xy_pix = self._center_shift_xy_deg / self._stim_size_deg
        else:
            raise ValueError(
                "The center is not fixed at one location for this session: {}".format(
                    self.datafile_path
                )
            )

    def _get_CS_center_info(self):
        """
        Creates
        -------
        CS_center_pos_xy_pix : array-like, 1D
            The x- and y-coordinates of the CS center in LSN pixels (origin at bottom-left).
        CS_center_radius_pix : float
            The radius of the CS center in LSN pixels.
        """
        try:
            self._get_CS_center_shift()
            self.monitor_center_pix_xy = (
                np.array(self.LSN_stim.shape[-2:][::-1]) / 2 - 0.5
            )
            self.CS_center_pos_xy_pix = (
                self.monitor_center_pix_xy + self._center_shift_xy_pix
            )
            CS_center_radius_deg = self._CS_center_diameter_deg / 2
            self.CS_center_radius_pix = CS_center_radius_deg / self._stim_size_deg
            self._is_CS_session = True
        except KeyError:
            print("This is not a center-surround session!")
            self._is_CS_session = False

    def get_RF_loc_masks(self, loc_thresh=0.8, RF_thresh=0, bin_num=1000):
        """To get the boolean masks of RF locations based on their overlapping index with the centers.

        Parameters
        ----------
        loc_thresh : float
            The threshold for deciding whether the RF is located within the center or surround or not.
        RF_thresh : float or int
            The threshold of RFs to be considered. Default is 0.
        bin_num : int
            The number of binning for an LSN pixel when computing the overlapping indices.
            Higher bin_num gives higher precision but will take longer computational time.

        Creates
        -------
        location_mask_dict : dict
            Dictionary containing masks for different conditions.
        """
        self._get_center_overlap(RF_thresh, bin_num)
        self.RF_loc_thresh = loc_thresh
        ON_center = self.ON_overlap_idx >= self.RF_loc_thresh
        ON_surround = self.ON_overlap_idx <= 1 - self.RF_loc_thresh
        ON_border = (self.ON_overlap_idx > 1 - self.RF_loc_thresh) & ~ON_center
        No_ON = ~(ON_center | ON_surround | ON_border)
        OFF_center = self.OFF_overlap_idx >= self.RF_loc_thresh
        OFF_surround = self.OFF_overlap_idx <= 1 - self.RF_loc_thresh
        OFF_border = (self.OFF_overlap_idx > 1 - self.RF_loc_thresh) & ~OFF_center
        No_OFF = ~(OFF_center | OFF_surround | OFF_border)
        both_center = ON_center & OFF_center
        both_surround = ON_surround & OFF_surround
        both_border = ON_border & OFF_border
        No_RF = No_ON & No_OFF
        ON_center_alone = ON_center & No_OFF
        OFF_center_alone = OFF_center & No_ON
        ON_center_OFF_surround = ON_center & OFF_surround
        OFF_center_ON_surround = OFF_center & ON_surround

        self.location_mask_dict = {}
        self.location_mask_dict["ON_center"] = ON_center
        self.location_mask_dict["ON_surround"] = ON_surround
        self.location_mask_dict["ON_border"] = ON_border
        self.location_mask_dict["No_ON"] = No_ON
        self.location_mask_dict["OFF_center"] = OFF_center
        self.location_mask_dict["OFF_surround"] = OFF_surround
        self.location_mask_dict["OFF_border"] = OFF_border
        self.location_mask_dict["No_OFF"] = No_OFF
        self.location_mask_dict["both_center"] = both_center
        self.location_mask_dict["both_surround"] = both_surround
        self.location_mask_dict["both_border"] = both_border
        self.location_mask_dict["No_RF"] = No_RF
        self.location_mask_dict["ON_center_alone"] = ON_center_alone
        self.location_mask_dict["OFF_center_alone"] = OFF_center_alone
        self.location_mask_dict["ON_center_OFF_surround"] = ON_center_OFF_surround
        self.location_mask_dict["OFF_center_ON_surround"] = OFF_center_ON_surround

    def plot_RFs(
        self,
        title,
        cell_idx_lst,
        polarity="both",
        num_cols=5,
        label_peak=True,
        show_CS_center=True,
        contour_levels=[],
    ):
        """To plot the RFs.

        Parameters
        ----------
        title : str
            The title of the figure.
        cell_idx_lst : list or np.array
            The cell numbers to be plotted.
        polarity : str
            'ON', 'OFF', or 'both' (default). The polarity of the RFs to be plotted.
        num_cols : int
            The number of columns of the subplots.
        label_peak : bool
            If True, the pixel with max response will be labeled.
        contour_levels : array-like
            The contour levels to be plotted.
        """
        if label_peak:
            self._get_RF_peaks_yx()
        ON_RFs = [
            self._normalize_RF(self.ON_RFs[i].copy()) for i in range(self.num_cells)
        ]
        OFF_RFs = [
            self._normalize_RF(self.OFF_RFs[i].copy()) for i in range(self.num_cells)
        ]
        polarity = ReceptiveFieldPolarity.from_(polarity)
        figsize_x = num_cols * 2
        num_rows = np.ceil(len(cell_idx_lst) / num_cols).astype(int)
        figsize_factor = (
            (self.LSN_stim.shape[1] * num_rows)
            / (self.LSN_stim.shape[2] * num_cols)
            * 1.5
        )
        figsize_y = figsize_x * figsize_factor
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(figsize_x, figsize_y))
        axes = axes.flatten()
        fig.tight_layout()
        fig.subplots_adjust(
            wspace=0.1,
            hspace=0.2,
            top=0.95,
            bottom=0.01,
            left=0.002,
            right=0.998,
        )
        fig.suptitle(title)
        for i, ax in enumerate(axes):
            if i < len(cell_idx_lst) and cell_idx_lst[i] < self.num_cells:
                idx = cell_idx_lst[i]
                if polarity == ReceptiveFieldPolarity.ON:
                    pcol = ax.pcolormesh(ON_RFs[idx], cmap="coolwarm")
                    if label_peak:
                        ax.plot(
                            self.ON_RF_peaks_yx[idx, 1] + 0.5,
                            self.ON_RF_peaks_yx[idx, 0] + 0.5,
                            ".r",
                        )
                if polarity == ReceptiveFieldPolarity.OFF:
                    pcol = ax.pcolormesh(OFF_RFs[idx], cmap="coolwarm")
                    if label_peak:
                        ax.plot(
                            self.OFF_RF_peaks_yx[idx, 1] + 0.5,
                            self.OFF_RF_peaks_yx[idx, 0] + 0.5,
                            ".b",
                        )
                if polarity == ReceptiveFieldPolarity.BOTH:
                    pcol = ax.pcolormesh(
                        ON_RFs[idx] + OFF_RFs[idx], cmap="coolwarm"
                    )  # plus because OFF_RFs are already negative.
                    if label_peak:
                        ax.plot(
                            self.ON_RF_peaks_yx[idx, 1] + 0.5,
                            self.ON_RF_peaks_yx[idx, 0] + 0.5,
                            ".r",
                        )
                        ax.plot(
                            self.OFF_RF_peaks_yx[idx, 1] + 0.5,
                            self.OFF_RF_peaks_yx[idx, 0] + 0.5,
                            ".b",
                        )
                ax.set_aspect("equal", "box")
                pcol.set_edgecolor("face")
                pcol.set_clim([-1, 1])
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title("Cell {}".format(self.cell_ids[idx]), y=0.99)
                # ax.set_ylim(ax.get_ylim()[::-1])
                if show_CS_center:
                    CS_center = plt.Circle(
                        self.CS_center_pos_xy_pix + 0.5,
                        self.CS_center_radius_pix,
                        color="k",
                        fill=False,
                    )
                    ax.add_patch(CS_center)
                if contour_levels:
                    if polarity != ReceptiveFieldPolarity.ON:
                        ax.contour(
                            -OFF_RFs[idx],
                            contour_levels,
                            colors="deepskyblue",
                            origin="lower",
                        )
                    if polarity != ReceptiveFieldPolarity.OFF:
                        ax.contour(
                            ON_RFs[idx],
                            contour_levels,
                            colors="gold",
                            origin="lower",
                        )
            else:
                ax.set_visible(False)
        return fig

    def plot_pixel_avg_dff_traces(
        self, polarity, cell_idx, num_std=2, ax=None, **pltargs
    ):
        """To plot the trial-averaged responses within pixels (all pixels of the LSN stimulus) for a cell.

        Parameters
        ----------
        polarity : str
            'ON' or 'OFF'. The polarity of the responses to be plotted.
        cell_idx : int
            The cell index to be plotted.
        num_std : int or float
            Number of standard deviation from mean for plotting the horizontal span.
        pltargs
            Other kwargs as for plt.plot().
        """
        polarity = ReceptiveFieldPolarity.from_(polarity)
        if polarity == ReceptiveFieldPolarity.ON:
            avg_responses = self.ON_avg_responses
        elif polarity == ReceptiveFieldPolarity.OFF:
            avg_responses = self.OFF_avg_responses
        else:
            raise ValueError("Please enter 'ON' or 'OFF' for the polarity.")
        flat_response = avg_responses[cell_idx].reshape(
            -1, self.trial_fluo.num_timesteps
        )
        response_mean = np.nanmean(flat_response)
        response_std = np.nanstd(flat_response)
        lower_bound = response_mean - num_std * response_std
        upper_bound = response_mean + num_std * response_std
        if ax is None:
            ax = plt.gca()
        if polarity == ReceptiveFieldPolarity.ON:
            target = self._ON_stim_value
        else:
            target = self._OFF_stim_value
        if self.is_use_positive_fluo:
            single_cell_data = (
                self.trial_fluo.get_trials(self._trial_mask)
                .get_cells(cell_idx)
                .positive_part()
            )
        else:
            single_cell_data = self.trial_fluo.get_trials(self._trial_mask).get_cells(
                cell_idx
            )
        stimulus_highlighted = (
            False  # Add a flag so we can avoid highlighting the stimulus multiple times
        )
        for y in range(self.LSN_stim.shape[1]):
            for x in range(self.LSN_stim.shape[2]):
                trial_mean = single_cell_data.get_trials(
                    self.LSN_stim[self._trial_mask, y, x] == target
                ).trial_mean()
                if not stimulus_highlighted:
                    trial_mean.plot(
                        ax=ax,
                        fill_mean_pm_std=False,
                        highlight_non_baseline=True,
                        **pltargs
                    )
                    stimulus_highlighted = True
                else:
                    trial_mean.plot(
                        ax=ax,
                        fill_mean_pm_std=False,
                        highlight_non_baseline=False,
                        **pltargs
                    )
        integration_start_sec = (
            self._integration_window_start * self.trial_fluo.timestep_width
            - self.trial_fluo._baseline_duration
        )
        integration_end_sec = (
            integration_start_sec
            + (self._integration_window_len - 1) * self.trial_fluo.timestep_width
        )
        ax.axvspan(
            integration_start_sec,
            integration_end_sec,
            color="lightblue",
            alpha=0.5,
            label="RF integration window",
        )
        ax.axhspan(
            lower_bound,
            upper_bound,
            color="lightgreen",
            alpha=0.5,
            label="Mean $\pm$ {} std".format(num_std),
        )
        ax.legend()
        ax.set_title(
            "Cell {} ({} responses)\nTrial-averaged DF/F traces within pixel".format(
                cell_idx, polarity.name
            )
        )
        return ax

    def save_data(self, save_path):
        data_dict = {}
        data_dict['cell IDs'] = self.cell_ids
        data_dict['Chi-square p-values'] = self.chi_square_pvals
        data_dict['CS center pos xy (pix)'] = self.CS_center_pos_xy_pix
        data_dict['CS center radius (pix)'] = self.CS_center_radius_pix
        data_dict['analyzed data file'] = self.datafile_path
        data_dict['is use corrected LSN'] = self.is_use_corrected_LSN
        data_dict['is use DFF z-score'] = self.is_use_dff_z_score
        data_dict['is use positive fluo'] = self.is_use_positive_fluo
        data_dict['is use valid eye pos'] = self.is_use_valid_eye_pos
        data_dict['location masks'] = self.location_mask_dict
        data_dict['LSN stimuli'] = self.LSN_stim
        data_dict['monitor center xy (pix)'] = self.monitor_center_pix_xy
        data_dict['num baseline frames'] = self.num_baseline_frames
        data_dict['number of cells'] = self.num_cells
        data_dict['OFF averaged responses'] = self.OFF_avg_responses
        data_dict['OFF overlapping index'] = self.OFF_overlap_idx
        data_dict['OFF RFs'] = self.OFF_RFs
        data_dict['ON averaged responses'] = self.ON_avg_responses
        data_dict['ON overlapping index'] = self.ON_overlap_idx
        data_dict['ON RFs'] = self.ON_RFs
        data_dict['RF location threshold'] = self.RF_loc_thresh
        data_dict['RF type'] = self.RF_type
        data_dict['valid eye pos masks'] = self.valid_eye_pos
        data_dict['ref pos for LSN stim correction'] = self.yx_ref
        data_dict['CS center xy shifts (deg)'] = self._center_shift_xy_deg
        data_dict['CS center xy shifts (pix)'] = self._center_shift_xy_pix
        data_dict['corrected LSN stim by eye pos'] = self._corrected_LSN_stim
        data_dict['CS center diametere (deg)'] = self._CS_center_diameter_deg
        data_dict['Fluo frame rate (Hz)'] = self._frame_rate_Hz
        data_dict['Original full LSN stim'] = self._full_LSN_stim
        data_dict['RF integration window length'] = self._integration_window_len
        data_dict['RF integration window start'] = self._integration_window_start
        data_dict['is CS session'] = self._is_CS_session
        data_dict['LSN grid size (deg)'] = self._stim_size_deg
        data_dict['LSN trial masks'] = self._trial_mask
        np.save(save_path, data_dict)
        print("Data saved!")


class ReceptiveFieldPolarity(Enum):
    ON = 1
    OFF = 2
    BOTH = 3

    @staticmethod
    def from_(polarity):
        """Coerce `polarity` to a ReceptiveFieldPolarity."""
        if isinstance(polarity, ReceptiveFieldPolarity):
            return polarity
        elif polarity.upper() == "ON":
            return ReceptiveFieldPolarity.ON
        elif polarity.upper() == "OFF":
            return ReceptiveFieldPolarity.OFF
        elif polarity.upper() == "ANY":
            pol_value = ReceptiveFieldPolarity._get_any()
            return (
                ReceptiveFieldPolarity.ON
                if pol_value == 1
                else ReceptiveFieldPolarity.OFF
            )
        elif polarity.upper() == "BOTH":
            return ReceptiveFieldPolarity.BOTH
        else:
            raise ValueError("Polarity must be 'ON', 'OFF', 'ANY' or 'BOTH'.")

    def _get_any():
        return np.random.randint(1, 3)


class gag:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._original_stdout