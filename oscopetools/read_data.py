#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 21:59:56 2020

@author: saskiad
"""

import h5py
import pandas as pd
import numpy as np


def get_dff_traces(file_path):
    f = h5py.File(file_path)
    dff = f['dff_traces'][()]
    f.close()
    return dff


def get_raw_traces(file_path):
    f = h5py.File(file_path)
    raw = f['raw_traces'][()]
    f.close()
    return raw


def get_running_speed(file_path):
    f = h5py.File(file_path)
    dx = f['running_speed'][()]
    f.close()
    return dx


def get_cell_ids(file_path):
    f = h5py.File(file_path)
    cell_ids = f['cell_ids'][()]
    f.close()
    return cell_ids


def get_max_projection(file_path):
    f = h5py.File(file_path)
    max_proj = f['max_projection'][()]
    f.close()
    return max_proj


def get_metadata(file_path):
    import ast

    f = h5py.File(file_path)
    md = f.get('meta_data')[...].tolist()
    f.close()
    meta_data = ast.literal_eval(md)
    return meta_data


def get_roi_table(file_path):
    return pd.read_hdf(file_path, 'roi_table')


def get_stimulus_table(file_path, stimulus):
    return pd.read_hdf(file_path, stimulus)


def get_stimulus_epochs(file_path, session_type):
    if session_type == 'drifting_gratings_grid':
        stim_name_1 = 'drifting_gratings_grid'
    elif session_type == 'center_surround':
        stim_name_1 = 'center_surround'
    elif session_type == 'size_tuning':
        stim_name_1 = np.NaN  # TODO: figure this out

    stim1 = get_stimulus_table(file_path, stim_name_1)
    stim2 = get_stimulus_table(file_path, 'locally_sparse_noise')
    stim_epoch = pd.DataFrame(columns=('Start', 'End', 'Stimulus_name'))
    break1 = np.where(np.ediff1d(stim1.Start) > 1000)[0][0]
    break2 = np.where(np.ediff1d(stim2.Start) > 1000)[0][0]
    stim_epoch.loc[0] = [stim1.Start[0], stim1.End[break1], stim_name_1]
    stim_epoch.loc[1] = [stim1.Start[break1 + 1], stim1.End.max(), stim_name_1]
    stim_epoch.loc[2] = [
        stim2.Start[0],
        stim2.End[break2],
        'locally_sparse_noise',
    ]
    stim_epoch.loc[3] = [
        stim2.Start[break2 + 1],
        stim2.End.max(),
        'locally_sparse_noise',
    ]
    stim_epoch.sort_values(by='Start', inplace=True)
    stim_epoch.loc[4] = [
        0,
        stim_epoch.Start.iloc[0] - 1,
        'spontaneous_activity',
    ]
    for i in range(1, 4):
        stim_epoch.loc[4 + i] = [
            stim_epoch.End.iloc[i - 1] + 1,
            stim_epoch.Start.iloc[i] - 1,
            'spontaneous_activity',
        ]
    stim_epoch.sort_values(by='Start', inplace=True)
    stim_epoch.reset_index(inplace=True)
    stim_epoch['Duration'] = stim_epoch.End - stim_epoch.Start

    return stim_epoch


def get_eye_tracking(file_path):
    return pd.read_hdf(file_path, 'eye_tracking')
