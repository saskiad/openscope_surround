#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 05:34:22 2020

@author: kailun
"""

import numpy as np
from oscopetools.LSN_analysis import LSN_analysis

# The path to the data file.
datafile_path = '/home/kailun/Desktop/PhD/other_projects/surround_suppression_neural_code/Multiplex/Center_Surround_976474801_data.h5'
# The path to the LSN stimulus npy file.
LSN_stim_path = '/home/kailun/Desktop/PhD/other_projects/surround_suppression_neural_code/openscope_surround-master/stimulus/sparse_noise_8x14.npy'
num_baseline_frames = 3   # int or None. The number of baseline frames before the start and after the end of a trial.
use_dff_z_score = False   # True or False. If True, the cell responses will be converted to z-score before analysis.

# To initialize the analysis.
LSN_data = LSN_analysis(datafile_path, LSN_stim_path, num_baseline_frames, use_dff_z_score)

#%%
# To get an overview of the data.
print(LSN_data)

#%%
# Other variables (RFs, ON/OFF responses, etc.) will be automatically updated.
correct_LSN = False   # If True, the LSN stimulus corrected by eye positions will be used. Otherwise, the original LSN stimulus will be used.
LSN_data.correct_LSN_by_eye_pos(correct_LSN)

#%%
# Other variables (RFs, ON/OFF responses, etc.) will be automatically updated.
use_only_valid_eye_pos = False   # If True, only stimuli with valid eye positions are used. Otherwise, all stimuli will be used.
LSN_data.use_valid_eye_pos(use_only_valid_eye_pos)

#%%
# Other variables (RFs, ON/OFF responses, etc.) will be automatically updated.
use_only_positive_responses = False   # If True, the fluorescence responses less than 0 will be set to 0 when computing the avg_responses.
LSN_data.use_positive_fluo(use_only_positive_responses)

#%%
# The RFs are computed during initialization with default parameters. Here, we can change the threshold and integration window for RFs.
# To compute the RFs by using different thresholds (default = 0) and different integration windows by adjusting the window_start (shifting)
# and window_len (length of the integration window).

threshold = 0.   # int or float, range = [0, 1]. The threshold for the RF, anything below the threshold will be set to 0.
window_start = None   # int or None. The start index (within a trial) of the integration window for computing the RFs.
window_len = None   # int or None. The length of the integration window in frames for computing the RFs.
LSN_data.get_RFs(threshold, window_start, window_len)

#%%
# To plot the RFs.
fig_title = "Receptive fields"   # The title of the figure.
cell_idx_lst = np.arange(100)   # list or np.array. The cell numbers to be plotted.
polarity = 'both'   # 'ON', 'OFF', or 'both'. The polarity of the RFs to be plotted.
num_cols = 10   # int. The number of columns of the subplots.
label_peak = True   # bool. If True, the pixel with max response will be labeled. The ON peaks are labeled with red dots and OFF peaks with blue dots.
contour_levels = [0.6]   # list or array-like. The contour levels to be plotted. Examples: [], [0.5], [0.6, 0.8].
LSN_data.plot_RFs(fig_title, cell_idx_lst, polarity, num_cols, label_peak, contour_levels)

#%%
# To plot the trial-averaged responses within pixels (all pixels of the LSN stimulus) for a cell.
# Other keyword arguments can be added for plt.plot().
polarity = 'ON'   # 'ON' or 'OFF'. The polarity of the responses to be plotted.
cell_idx = 10   # The cell index to be plotted.
num_std = 2   # int or float. Number of standard deviation from mean for plotting the horizontal span.
LSN_data.plot_pixel_avg_dff_traces(polarity, cell_idx, num_std)