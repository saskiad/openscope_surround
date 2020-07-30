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
from enum import Enum
import warnings


class LSN_analysis:
    _ON_stim_value = 255
    _OFF_stim_value = 0
    _background_stim_value = 127
    _yx_ref = None  # The reference y- and x-positions used for correcting the LSN stimulus array.
    _stim_size = (
        10  # The side length of the stimulus in degree (same unit as eye pos).
    )
    _frame_rate_Hz = 30  # The frame rate of fluorescent responses in Hz.

    def __init__(
        self,
        datafile_path,
        LSN_stim_path,
        num_baseline_frames=None,
        use_dff_z_score=False,
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
        
        """
        self.datafile_path = datafile_path
        self.LSN_stim_path = LSN_stim_path
        self.num_baseline_frames = num_baseline_frames
        if (self.num_baseline_frames is None) or (
            self.num_baseline_frames < 0
        ):
            self.num_baseline_frames = 0
        self.is_use_dff_z_score = use_dff_z_score
        self.is_use_corrected_LSN = False
        self.is_use_valid_eye_pos = False
        self.is_use_positive_fluo = False
        self.dff_fluo = rd.get_dff_traces(self.datafile_path)
        self.num_cells = self.dff_fluo.num_cells
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
            self._stim_size,
            self._background_stim_value,
        )
        self.LSN_stim = self._full_LSN_stim[self.LSN_stim_table.Frame]
        self._trial_mask = self.valid_eye_pos
        self._update_responses()

    def __str__(self):
        return (
            "Analyzing file: {}\n"
            "ON LSN stimulus value: {}\n"
            "OFF LSN stimulus value: {}\n"
            "Background LSN value: {}\n"
            "LSN stimulus size: {} degree\n"
            "Number of cells: {}\n"
            "Use DF/F z-score: {}\n"
            "Use corrected LSN: {}\n"
            "Use only valid eye positions: {}\n"
            "Use only positive fluorescence responses: {}"
        ).format(
            self.datafile_path,
            self._ON_stim_value,
            self._OFF_stim_value,
            self._background_stim_value,
            self._stim_size,
            self.num_cells,
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
        if value:
            self.LSN_stim = self._corrected_LSN_stim
        else:
            self.LSN_stim = self._full_LSN_stim[self.LSN_stim_table.Frame]
        self._update_responses()
        self.is_use_corrected_LSN = bool(value)

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
        if value:
            self._trial_mask = self.valid_eye_pos
        else:
            self._trial_mask = np.array([True] * self.LSN_stim_table.shape[0])
        self._update_responses()
        self.is_use_valid_eye_pos = bool(value)

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
        self.is_use_positive_fluo = bool(value)
        self._update_responses()

    def _update_responses(self):
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
        self.get_RFs()

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
                    response.get_trials(LSN_stim[:, y, x] == target)
                    .trial_mean()
                    .data
                )
        return avg_responses

    def _compute_p_values(self):
        raise NotImplementedError

    def get_RFs(self, threshold=0, window_start=None, window_len=None):
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
            window_len = (
                self.ON_avg_responses.shape[-1] - 2 * self.num_baseline_frames
            )
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
        ON_cell_peak_idx = self.ON_RFs.reshape(
            self.ON_RFs.shape[0], -1
        ).argmax(1)
        OFF_cell_peak_idx = self.OFF_RFs.reshape(
            self.OFF_RFs.shape[0], -1
        ).argmin(1)
        self.ON_RF_peaks_yx = np.column_stack(
            np.unravel_index(ON_cell_peak_idx, self.ON_RFs[0, :, :].shape)
        )
        self.OFF_RF_peaks_yx = np.column_stack(
            np.unravel_index(OFF_cell_peak_idx, self.OFF_RFs[0, :, :].shape)
        )

    def _compute_RF_subfield(
        self, polarity, threshold, window_start, window_len
    ):
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

    def plot_RFs(
        self,
        title,
        cell_idx_lst,
        polarity="both",
        num_cols=5,
        label_peak=True,
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
        polarity = ReceptiveFieldPolarity.from_(polarity)
        figsize_x = num_cols * 2
        num_rows = np.ceil(len(cell_idx_lst) / num_cols).astype(int)
        figsize_factor = (
            (self.LSN_stim.shape[1] * num_rows)
            / (self.LSN_stim.shape[2] * num_cols)
            * 1.5
        )
        figsize_y = figsize_x * figsize_factor
        fig, axes = plt.subplots(
            num_rows, num_cols, figsize=(figsize_x, figsize_y)
        )
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
                    pcol = ax.pcolormesh(self.ON_RFs[idx])
                    if label_peak:
                        ax.plot(
                            self.ON_RF_peaks_yx[idx, 1] + 0.5,
                            self.ON_RF_peaks_yx[idx, 0] + 0.5,
                            ".r",
                        )
                if polarity == ReceptiveFieldPolarity.OFF:
                    pcol = ax.pcolormesh(self.OFF_RFs[idx])
                    if label_peak:
                        ax.plot(
                            self.OFF_RF_peaks_yx[idx, 1] + 0.5,
                            self.OFF_RF_peaks_yx[idx, 0] + 0.5,
                            ".b",
                        )
                if polarity == ReceptiveFieldPolarity.BOTH:
                    pcol = ax.pcolormesh(
                        self.ON_RFs[idx] + self.OFF_RFs[idx]
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
                ax.set_title("Cell {}".format(idx), y=0.99)
                if contour_levels:
                    if polarity != ReceptiveFieldPolarity.ON:
                        ax.contour(
                            -self.OFF_RFs[idx],
                            contour_levels,
                            colors="deepskyblue",
                            origin="lower",
                        )
                    if polarity != ReceptiveFieldPolarity.OFF:
                        ax.contour(
                            self.ON_RFs[idx],
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
            single_cell_data = self.trial_fluo.get_trials(
                self._trial_mask
            ).get_cells(cell_idx)
        stimulus_highlighted = False  # Add a flag so we can avoid highlighting the stimulus multiple times
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
            + (self._integration_window_len - 1)
            * self.trial_fluo.timestep_width
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
        raise NotImplementedError


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
