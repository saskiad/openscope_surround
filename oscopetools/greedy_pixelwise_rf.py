#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 22:20:46 2020

@author: danielm
"""

import numpy as np
from statsmodels.sandbox.stats.multicomp import multipletests

def get_receptive_field_greedy(L0_events,
                               stimulus_table,
                               LSN_template,
                               alpha=0.05,
                               sweep_response_type='mean'):
    # INPUTS:
    #
    # LO_events: 1D numpy array with shape (num_2p_imaging_frames_in_session,) 
    #               that is the timeseries of detected L0 events for the entire
    #               imaging session for a single ROI (e.g. cell/soma).
    # stimulus_table: pandas DataFrame that contains 'start' and 'end' columns
    #                 that indicate the imaging frames that bound the presentation
    #                 of each stimulus frame (i.e. one frame of LSN pixels, NOT one
    #                 monitor refresh cycle).
    #
    # LSN_template: 3D numpy array with shape (num_stim_frames,num_y_pixels,num_x_pixels)
    #               where each stimulus frame contains the locally sparse noise stimulus
    #               on a single trials (i.e. one row of 'stimulus_table').
    #
    # alpha: the significance threshold for a pixel to be included in the RF map.
    #        This number will be corrected for multiple comparisons (number of pixels).
    #
    # sweep_response_type: Choice of 'mean' for mean_sweep_events or 'binary' to
    #                       make boolean calls of whether any events occurred 
    #                       within the sweep window.
    #
    # OUTPUTS:
    #
    # receptive_field_on, receptive_field_off: 2D numpy arrays of the stimulus triggered
    #           average ('STA') receptive fields, after masking to show only
    #           responses for pixels that are determined to be significant
    #           by bootstrapping.
    
    # determine the type of calculation for sweep responses
    if sweep_response_type == 'mean':
        sweep_events = get_mean_sweep_events(L0_events,stimulus_table)
    else: #to stay consistent with Nic's original analysis
        sweep_events = binarize_sweep_events(L0_events,stimulus_table)
    
    # calculate p-values for each pixel to determine if the response is significant
    pvalues_on, pvalues_off = greedy_pixelwise_pvals(sweep_events,
                                                     stimulus_table,
                                                     LSN_template,
                                                     alpha=alpha)
    mask_on = pvals_to_mask(pvalues_on,alpha=alpha)
    mask_off = pvals_to_mask(pvalues_off,alpha=alpha)
    
    A = get_design_matrix(stimulus_table,LSN_template)

    STA_on, STA_off = calc_STA(A,sweep_events,LSN_template)

    # apply mask to get only the significant pixels of the STA
    receptive_field_on = STA_on * mask_on
    receptive_field_off = STA_off * mask_off
    
    return receptive_field_on, receptive_field_off

def calc_STA(A,sweep_events,LSN_template):
    
    (num_frames,num_y_pixels,num_x_pixels) = np.shape(LSN_template)
    number_of_pixels = A.shape[0] // 2

    STA = A.dot(sweep_events)
    STA_on = STA[:number_of_pixels].reshape(num_y_pixels,num_x_pixels)
    STA_off = STA[number_of_pixels:].reshape(num_y_pixels,num_x_pixels)
    
    return STA_on, STA_off

def pvals_to_mask(pixelwise_pvals,alpha=0.05):
    return pixelwise_pvals < alpha

def greedy_pixelwise_pvals(sweep_events,
                           stimulus_table,
                           LSN_template,
                           alpha=0.05):

    # INPUTS:
    #
    # sweep_events: 1D numpy array with shape (num_sweeps,) that has the response
    #               on each sweep.
    # 
    # stimulus_table: pandas DataFrame that contains 'start' and 'end' columns
    #                 that indicate the imaging frames that bound the presentation
    #                 of each stimulus frame (i.e. one frame of LSN pixels, NOT one
    #                 monitor refresh cycle).
    #
    # LSN_template: 3D numpy array with shape (num_stim_frames,num_y_pixels,num_x_pixels)
    #               where each stimulus frame contains the locally sparse noise stimulus
    #               on a single trials (i.e. one row of 'stimulus_table').
    #
    # OUTPUTS:
    # 
    # fdr_corrected_pvalues_on, fdr_corrected_pvalues_off: 2D numpy arrays
    #               containing the p-values for each pixel location after 
    #               correction for multiple comparisons.
    
    (num_stim_frames,num_y_pixels,num_x_pixels) = np.shape(LSN_template)
    number_of_pixels = num_y_pixels * num_x_pixels
    
    # compute p-values for each pixel by comparing the number of sweeps with
    #   an event against a null distribution obtained by shuffling.
    pvalues = events_to_pvalues_no_fdr_correction(sweep_events,stimulus_table,LSN_template)

    # correct the p-values for the multiple comparisons that were performed across
    #   all pixels, default is Holm-Sidak:
    fdr_corrected_pvalues = multipletests(pvalues, alpha=alpha)[1]

    # convert to 2D pixel arrays, split by On/Off pixels
    fdr_corrected_pvalues_on = fdr_corrected_pvalues[:number_of_pixels].reshape(num_y_pixels,num_x_pixels)
    fdr_corrected_pvalues_off = fdr_corrected_pvalues[number_of_pixels:].reshape(num_y_pixels,num_x_pixels)

    return fdr_corrected_pvalues_on, fdr_corrected_pvalues_off

def events_to_pvalues_no_fdr_correction(sweep_events,
                                        stimulus_table,
                                        LSN_template,
                                        number_of_shuffles=20000, 
                                        seed=1):

    # get stimulus design matrix:
    A = get_design_matrix(stimulus_table,LSN_template)

    # initialize random seed for reproducibility:
    np.random.seed(seed)

    # generate null distribution of pixel responses by shuffling with replacement
    shuffled_STAs = get_shuffled_pixelwise_responses(sweep_events, A, number_of_shuffles=number_of_shuffles)

    # p-values are the fraction of times the shuffled response to each pixel is greater than the
    #   actual observed response to that pixel.
    actual_STA = A.dot(sweep_events)
    p_values = np.mean(actual_STA.reshape(A.shape[0],1) <= shuffled_STAs,axis=1)
    
    return p_values

def get_design_matrix(stimulus_table,LSN_template,GRAY_VALUE=127):

    # construct the design matrix
    #
    # OUTPUTS:
    #
    # A: 2D numpy array with size (num_pixels, num_sweeps) where each element is
    #       True if the pixel was active during that sweep.
    
    (num_stim_frames,num_y_pixels,num_x_pixels) = np.shape(LSN_template)
    num_sweeps = len(stimulus_table)
    num_pixels = num_y_pixels * num_x_pixels
    
    sweep_stim_frames = stimulus_table['frame'].values.astype(np.int)
    
    # check that the inputs are complete
    assert np.max(sweep_stim_frames) <= num_stim_frames
    
    A = np.zeros((2*num_pixels, num_sweeps))
    for i_sweep,sweep_frame in enumerate(sweep_stim_frames):
        A[:num_pixels, i_sweep] = (LSN_template[sweep_frame,:,:].flatten() > GRAY_VALUE).astype(float)
        A[num_pixels:, i_sweep] = (LSN_template[sweep_frame,:,:].flatten() < GRAY_VALUE).astype(float)

    return A

def binarize_sweep_events(L0_events,stimulus_table):

    # INPUTS:
    #
    # LO_events: 1D numpy array with shape (num_2p_imaging_frames_in_session,) 
    #               that is the timeseries of detected L0 events for the entire
    #               imaging session for a single ROI (e.g. cell/soma).
    # 
    # stimulus_table: pandas DataFrame that contains 'start' and 'end' columns
    #                 that indicate the imaging frames that bound the presentation
    #                 of each stimulus frame (i.e. one frame of LSN pixels, NOT one
    #                 monitor refresh cycle).
    #
    # OUTPUTS:
    #
    # sweep_has_event: 1D numpy array of type bool with shape (num_sweeps,)
    #                   that has a binary call for each sweep of whether or not
    #                   the ROI had any events during the sweep.
    
    num_imaging_frames = len(L0_events)
    last_imaging_frame_during_stim = np.max(stimulus_table['end'].values)
    
    assert last_imaging_frame_during_stim <= num_imaging_frames
    
    num_sweeps = len(stimulus_table)
    sweep_has_event = np.zeros(num_sweeps, dtype=np.bool)
    for i_sweep, (start_frame, end_frame) in enumerate(zip(stimulus_table['start'].values, stimulus_table['end'].values)):
        
        if L0_events[start_frame:end_frame].max() > 0:
            sweep_has_event[i_sweep] = True

    return sweep_has_event

def get_mean_sweep_events(L0_events,stimulus_table):

    # INPUTS:
    #
    # LO_events: 1D numpy array with shape (num_2p_imaging_frames_in_session,) 
    #               that is the timeseries of detected L0 events for the entire
    #               imaging session for a single ROI (e.g. cell/soma).
    # 
    # stimulus_table: pandas DataFrame that contains 'start' and 'end' columns
    #                 that indicate the imaging frames that bound the presentation
    #                 of each stimulus frame (i.e. one frame of LSN pixels, NOT one
    #                 monitor refresh cycle).
    #
    # OUTPUTS:
    #
    # mean_sweep_events: 1D numpy array of type float with shape (num_sweeps,)
    #                   that has the mean event size for each sweep for the ROI.
    
    num_imaging_frames = len(L0_events)
    last_imaging_frame_during_stim = np.max(stimulus_table['end'].values)
    
    assert last_imaging_frame_during_stim <= num_imaging_frames
    
    num_sweeps = len(stimulus_table)
    mean_sweep_events = np.zeros(num_sweeps, dtype=np.float)
    for i_sweep, (start_frame, end_frame) in enumerate(zip(stimulus_table['start'].values, stimulus_table['end'].values)):
        mean_sweep_events[i_sweep] = L0_events[start_frame:end_frame].mean()

    return mean_sweep_events

def get_shuffled_pixelwise_responses(sweep_events,A,number_of_shuffles=5000):

    # OUTPUTS:
    #
    # shuffled_STAs: 2D numpy array with shape (num_pixels,num_shuffles)
    #               where each column is an STA generated from bootstrap resampling the sweep
    #               responses, with replacement.
    
    num_sweeps = len(sweep_events)
    
    shuffled_STAs = np.zeros((A.shape[0],number_of_shuffles))
    for i_shuffle in range(number_of_shuffles):

        shuffled_sweeps = np.random.choice(num_sweeps, size=(num_sweeps,), replace=True)
        shuffled_events = sweep_events[shuffled_sweeps]

        shuffled_STAs[:,i_shuffle] = A.dot(shuffled_events)

    return shuffled_STAs
