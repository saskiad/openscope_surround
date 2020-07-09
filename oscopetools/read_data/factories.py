#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 21:59:56 2020

@author: saskiad
"""

__all__ = (
    'get_dff_traces',
    'get_raw_traces',
    'get_cell_ids',
    'get_max_projection',
    'get_metadata',
    'get_stimulus_table',
    'get_stimulus_epochs',
    'get_eye_tracking',
    'get_running_speed',
    'get_roi_table',
)

import warnings

import h5py
import pandas as pd
import numpy as np

from .dataset_objects import RawFluorescence, EyeTracking, RunningSpeed
from .conditions import CenterSurroundStimulus, SetMembershipError

FRAME_RATE = 30.0  # Assumed frame rate in Hz. TODO: load from a file


def get_dff_traces(file_path):
    """Get DFF normalized fluorescence traces.

    Parameters
    ----------
    file_path : str
        Path to an HDF5 file containing DFF-normalized fluorescence traces.

    Returns
    -------
    dff_fluorescence : RawFluorescence
        A `TimeseriesDataset` subclass containing DFF-normalized fluorescence
        traces.

    """
    f = h5py.File(file_path, 'r')
    dff = f['dff_traces'][()]
    f.close()

    fluorescence_dataset = RawFluorescence(dff, 1.0 / FRAME_RATE)
    fluorescence_dataset.is_dff = True

    return fluorescence_dataset


def get_raw_traces(file_path):
    """Get raw fluorescence traces.

    Parameters
    ----------
    file_path : str
        Path to an HDF5 file containing raw fluorescence traces.

    Returns
    -------
    raw_fluoresence : RawFluorescence
        A `TimeseriesDataset` subclass containing fluorescence traces.

    """
    f = h5py.File(file_path, 'r')
    raw = f['raw_traces'][()]
    f.close()

    fluorescence_dataset = RawFluorescence(raw, 1.0 / FRAME_RATE)
    fluorescence_dataset.is_dff = False

    return fluorescence_dataset


def get_running_speed(file_path):
    f = h5py.File(file_path, 'r')
    dx = f['running_speed'][()]
    f.close()

    speed = RunningSpeed(dx, 1.0/FRAME_RATE)
    return speed


def get_cell_ids(file_path):
    f = h5py.File(file_path, 'r')
    cell_ids = f['cell_ids'][()]
    f.close()
    return cell_ids


def get_max_projection(file_path):
    f = h5py.File(file_path, 'r')
    max_proj = f['max_projection'][()]
    f.close()
    return max_proj


def get_metadata(file_path):
    import ast

    f = h5py.File(file_path, 'r')
    md = f.get('meta_data')[...].tolist()
    f.close()
    meta_data = ast.literal_eval(md)
    return meta_data


def get_roi_table(file_path):
    return pd.read_hdf(file_path, 'roi_table')


def get_stimulus_table(file_path, stimulus):
    """Read stimulus table into a pd.DataFrame.

    Parameters
    ----------
    file_path : str
    stimulus : str
        Type of stimulus to load.

    Returns
    -------
    stimulus_table : pd.DataFrame

    Notes
    -----
    If `stimulus` is 'center_surround', details of the stimulus are loaded
    into `CenterSurroundStimulus` objects. See
    `oscopetools.read_data.CenterSurroundStimulus` for details.

    """
    df = pd.read_hdf(file_path, stimulus)

    center_surround_objects = []
    invalid_rows = []
    if stimulus == 'center_surround':
        for ind, row in df.iterrows():
            try:
                cs_stimulus = CenterSurroundStimulus(
                    row['TF'],
                    row['SF'],
                    row['Contrast'],
                    row['Center_Ori'],
                    row['Surround_Ori'],
                )
                center_surround_objects.append(cs_stimulus)
            except SetMembershipError:
                invalid_rows.append(ind)

        if len(invalid_rows) > 0:
            warnings.warn(
                'Removed {} trials with invalid stimulus parameters: {}'.format(
                    len(invalid_rows), df.loc[invalid_rows, :]
                )
            )

        print(len(center_surround_objects))
        print(len(invalid_rows))
        print(df.shape[0])
        df.drop(index=invalid_rows, inplace=True)
        df['center_surround'] = center_surround_objects
        df.drop(
            columns=['TF', 'SF', 'Contrast', 'Center_Ori', 'Surround_Ori'],
            inplace=True,
        )

    return df


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
    raw_eyetracking_dataset = pd.read_hdf(file_path, 'eye_tracking')
    return EyeTracking(raw_eyetracking_dataset, 1.0 / FRAME_RATE)
