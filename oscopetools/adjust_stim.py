#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 17:18:47 2020

@author: kailun
"""

import numpy as np


def correct_LSN_stim_by_eye_pos(
    LSN_stim,
    LSN_stim_table,
    eye_tracking,
    yx_ref=None,
    stim_size=10,
    stim_background_value=127,
):
    """
    To correct the LSN stimulus array by using the eye position averaged within trial. 
    
    Parameters
    ----------
    LSN_stim : 3d np.array
        The LSN stimulus array, shape = (num_trials, ylen, xlen).
    LSN_stim_table : pd.DataFrame
        The stim_table of loccally sparse noise.
    eye_tracking : EyeTracking
        The eye tracking data.
    yx_ref : list, np.array, or None, default None
        The reference y- and x-positions (hypothetical eye position looking at the center of the stimulus monitor), 
        where corrected_stim_pos = original_stim_pos - yx_ref. If None, the mean y- and x-positions of the 
        eye during LSN stimuli will be the yx_ref.
    stim_size : int, default 10
        The side length of the stimulus in degree.
    stim_background_value : int, default 127
        The background value (gray) of the LSN stimulus.
    
    Returns
    -------
    corrected_stim_arr : 3d np.array
        The corrected LSN stimulus array according to the eye positions averaged within trial.
    isvalid_eye_pos : bool vector-like
        Boolean array showing valid eye position (not NaN). Use corrected_stim_arr[isvalid_eye_pos]
        to get the trials with valid eye position.
    yx_ref : 1d np.array
        The reference y- and x-positions used for correcting the LSN stimulus array.
    """
    eye_trials = eye_tracking.cut_by_trials(LSN_stim_table)
    eye_trial_mean = eye_trials.trial_mean(within_trial=True, ignore_nan=True)
    yx_eye_pos = np.squeeze(
        np.dstack(
            [eye_trial_mean.data.y_pos_deg, eye_trial_mean.data.x_pos_deg]
        )
    )
    if yx_ref is None:
        yx_ref = np.nanmean(yx_eye_pos, 0)
    border = (
        np.ceil(np.nanmax(abs(yx_eye_pos - yx_ref)) / stim_size).astype(int)
        + 1
    )  # + 1 for ensuring that the border is wide enough
    corrected_stim_arr = np.zeros(
        (
            LSN_stim_table.shape[0],
            LSN_stim.shape[1] + 2 * border,
            LSN_stim.shape[2] + 2 * border,
        ),
        dtype="int32",
    )
    corrected_stim_arr += stim_background_value
    corrected_stim_arr[:, border:-border, border:-border] = LSN_stim[
        LSN_stim_table["Frame"]
    ]
    isvalid_eye_pos = []
    for i in range(LSN_stim_table.shape[0]):
        if np.isnan(yx_eye_pos[i]).any():
            isvalid_eye_pos.append(False)
            continue
        else:
            isvalid_eye_pos.append(True)
        yx_deviation = np.around((yx_eye_pos[i] - yx_ref) / stim_size).astype(
            int
        )
        corrected_stim_arr[i] = np.roll(
            corrected_stim_arr[i], (yx_deviation[0], yx_deviation[1]), (0, 1)
        )
    return (
        corrected_stim_arr[:, border:-border, border:-border],
        np.array(isvalid_eye_pos),
        yx_ref,
    )
