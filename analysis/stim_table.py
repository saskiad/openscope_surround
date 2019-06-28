# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 17:33:28 2019

@author: danielm
"""
import os
import warnings

import numpy as np
import pandas as pd

from sync import Dataset


def coarse_mapping_create_stim_table(exptpath):
    """Return stim_tables for locally sparse noise and drifting gratings grid.

    Input:
        exptpath (str)

    Returns:
        Dict of locally_sparse_noise and drifting_gratings_grid stim tables.

    """
    data = load_stim(exptpath)
    twop_frames, _, _, _ = load_sync(exptpath)

    stim_table = {}
    stim_table['locally_sparse_noise'] = locally_sparse_noise_table(
        data, twop_frames
    )
    stim_table['drifting_gratings_grid'] = DGgrid_table(data, twop_frames)

    return stim_table


def lsnCS_create_stim_table(exptpath):
    """Return stim_tables for locally sparse noise and center surround stimuli.

    Input:
        exptpath (str)

    Returns:
        Dict of center_surround and locally_sparse_noise stim tables.

    """
    data = load_stim(exptpath)
    twop_frames, _, _, _ = load_sync(exptpath)

    stim_table = {}
    stim_table['center_surround'] = center_surround_table(data, twop_frames)
    stim_table['locally_sparse_noise'] = locally_sparse_noise_table(
        data, twop_frames
    )

    return stim_table


def DGgrid_table(data, twop_frames, verbose = True):

    DG_idx = get_stimulus_index(data, 'grating')

    timing_table, actual_sweeps, expected_sweeps = get_sweep_frames(
        data, DG_idx
    )

    if verbose:
        print 'Found {} of {} expected sweeps.'.format(
            actual_sweeps, expected_sweeps
        )

    stim_table = pd.DataFrame(
        np.column_stack((
            twop_frames[timing_table['start']],
            twop_frames[timing_table['end']]
        )),
        columns=('Start', 'End')
    )

    for attribute in ['TF', 'SF', 'Contrast', 'Ori', 'PosX', 'PosY']:
        stim_table[attribute] = get_attribute_by_sweep(
            data, DG_idx, attribute
        )[:len(stim_table)]

    return stim_table


def locally_sparse_noise_table(data, twop_frames, verbose = True):
    """Return stim table for locally sparse noise stimulus.

    """
    lsn_idx = get_stimulus_index(data, 'locally_sparse_noise')

    timing_table, actual_sweeps, expected_sweeps = get_sweep_frames(
        data, lsn_idx
    )
    if verbose:
        print 'Found {} of {} expected sweeps.'.format(
            actual_sweeps, expected_sweeps
        )

    stim_table = pd.DataFrame(
        np.column_stack((
            twop_frames[timing_table['start']],
            twop_frames[timing_table['end']]
        )),
        columns=('Start', 'End')
    )

    stim_table['Frame'] = np.array(
        data['stimuli'][lsn_idx]['sweep_order'][:len(stim_table)]
    )

    return stim_table


def center_surround_table(data, twop_frames, verbose = True):

    center_idx = get_stimulus_index(data, 'center')
    surround_idx = get_stimulus_index(data, 'surround')

    timing_table, actual_sweeps, expected_sweeps = get_sweep_frames(
        data, center_idx
    )
    if verbose:
        print 'Found {} of {} expected sweeps'.format(
            actual_sweeps, expected_sweeps
        )

    stim_table = pd.DataFrame(
        np.column_stack((
            twop_frames[timing_table['start']],
            twop_frames[timing_table['end']]
        )),
        columns=('Start', 'End')
    )

    # TODO make this take either center or surround SF and TF depending on which is not NaN
    for attribute in ['TF', 'SF', 'Contrast']:
        stim_table[attribute] = get_attribute_by_sweep(
            data, center_idx, attribute
        )[:len(stim_table)]
    stim_table['Center_Ori'] = get_attribute_by_sweep(
        data, center_idx, 'Ori'
    )[:len(stim_table)]
    stim_table['Surround_Ori'] = get_attribute_by_sweep(
        data, surround_idx, 'Ori'
    )[:len(stim_table)]

    return stim_table


def get_stimulus_index(data, stim_name):
    """Return the index of stimulus in data.

    Returns the position of the first occurrence of stim_name in data. Raises a
    KeyError if a stimulus with a name containing stim_name is not found.

    Inputs:
        data (dict-like)
            -- Object in which to search for a named stimulus.
        stim_name (str)

    Returns:
        Index of stimulus stim_name in data.

    """
    for i_stim, stim_data in enumerate(data['stimuli']):
        if stim_name in stim_data['stim_path']:
            return i_stim

    raise KeyError('Stimulus with stim_name={} not found!'.format(stim_name))


def get_display_sequence(data, stimulus_idx):

    display_sequence = np.array(
        data['stimuli'][stimulus_idx]['display_sequence']
    )
    pre_blank_sec = int(data['pre_blank_sec'])
    display_sequence += pre_blank_sec
    display_sequence *= int(data['fps'])  # in stimulus frames

    return display_sequence


def get_sweep_frames(data, stimulus_idx):

    sweep_frames = data['stimuli'][stimulus_idx]['sweep_frames']
    timing_table = pd.DataFrame(
        np.array(sweep_frames).astype(np.int),
        columns=('start', 'end')
    )
    timing_table['dif'] = timing_table['end']-timing_table['start']

    display_sequence = get_display_sequence(data, stimulus_idx)

    timing_table.start += display_sequence[0, 0]
    for seg in range(len(display_sequence)-1):
        for index, row in timing_table.iterrows():
            if row.start >= display_sequence[seg, 1]:
                timing_table.start[index] = (
                    timing_table.start[index]
                    - display_sequence[seg, 1]
                    + display_sequence[seg+1, 0]
                )
    timing_table.end = timing_table.start+timing_table.dif
    expected_sweeps = len(timing_table)
    timing_table = timing_table[timing_table.end <= display_sequence[-1, 1]]
    timing_table = timing_table[timing_table.start <= display_sequence[-1, 1]]
    actual_sweeps = len(timing_table)

    return timing_table, actual_sweeps, expected_sweeps


def get_attribute_by_sweep(data, stimulus_idx, attribute):

    attribute_idx = get_attribute_idx(data, stimulus_idx, attribute)

    sweep_order = data['stimuli'][stimulus_idx]['sweep_order']
    sweep_table = data['stimuli'][stimulus_idx]['sweep_table']

    num_sweeps = len(sweep_order)

    attribute_by_sweep = np.zeros((num_sweeps,))
    attribute_by_sweep[:] = np.NaN

    unique_conditions = np.unique(sweep_order)
    for i_condition, condition in enumerate(unique_conditions):
        sweeps_with_condition = np.argwhere(sweep_order == condition)[:, 0]

        if condition > 0:  # blank sweep is -1
            attribute_by_sweep[sweeps_with_condition] = sweep_table[condition][attribute_idx]

    return attribute_by_sweep


def get_attribute_idx(data, stimulus_idx, attribute):
    """Return the index of attribute in data for the given stimulus.

    Returns the position of the first occurrence of attribute. Raises a
    KeyError if not found.
    """
    attribute_names = data['stimuli'][stimulus_idx]['dimnames']
    for attribute_idx, attribute_str in enumerate(attribute_names):
        if attribute_str == attribute:
            return attribute_idx

    raise KeyError('Attribute {} for stimulus_ids {} not found!'.format(
            attribute, stimulus_idx
        ))


def load_stim(exptpath, verbose = True):
    """Load stim.pkl file into a DataFrame.

    Inputs:
        exptpath (str)
            -- Directory in which to search for files with _stim.pkl suffix.
        verbose (bool)
            -- Print filename (if found).

    Returns:
        DataFrame with contents of stim pkl.

    """
    # Look for a file with the suffix '_stim.pkl'
    pklpath = None
    for f in os.listdir(exptpath):
        if f.endswith('_stim.pkl'):
            pklpath = os.path.join(exptpath, f)
            if verbose:
                print "Pkl file:", f

    if pklpath is None:
        raise IOError(
            'No files with the suffix _stim.pkl were found in {}'.format(
                exptpath
            )
        )

    return pd.read_pickle(pklpath)


def load_sync(exptpath, verbose = True):

    #verify that sync file exists in exptpath
    syncpath = None
    for f in os.listdir(exptpath):
        if f.endswith('_sync.h5'):
            syncpath = os.path.join(exptpath, f)
            if verbose:
                print "Sync file:", f
    if syncpath is None:
        raise IOError(
            'No files with the suffix _sync.h5 were found in {}'.format(
                exptpath
            )
        )

    #load the sync data from .h5 and .pkl files
    d = Dataset(syncpath)
    #print d.line_labels

    #set the appropriate sample frequency
    sample_freq = d.meta_data['ni_daq']['counter_output_freq']

    #get sync timing for each channel
    twop_vsync_fall = d.get_falling_edges('2p_vsync')/sample_freq
    stim_vsync_fall = d.get_falling_edges('stim_vsync')[1:]/sample_freq #eliminating the DAQ pulse
    photodiode_rise = d.get_rising_edges('stim_photodiode')/sample_freq

    #make sure all of the sync data are available
    channels = {
        'twop_vsync_fall': twop_vsync_fall,
        'stim_vsync_fall': stim_vsync_fall,
        'photodiode_rise': photodiode_rise
    }
    channel_test = []
    for chan in channels.keys():
        # Check that signal is high at least once in each channel.
        channel_test.append(any(channels[chan]))
    if not all(channel_test):
        raise RuntimeError('Not all channels present. Sync test failed.')
    elif verbose:
        print "All channels present."

    #test and correct for photodiode transition errors
    ptd_rise_diff = np.ediff1d(photodiode_rise)
    short = np.where(np.logical_and(ptd_rise_diff > 0.1, ptd_rise_diff < 0.3))[0]
    medium = np.where(np.logical_and(ptd_rise_diff > 0.5, ptd_rise_diff < 1.5))[0]
    ptd_start = 3
    for i in medium:
        if set(range(i-2, i)) <= set(short):
            ptd_start = i+1
    ptd_end = np.where(photodiode_rise > stim_vsync_fall.max())[0][0] - 1

    if ptd_start > 3 and verbose:
        print 'ptd_start: ' + str(ptd_start)
        print "Photodiode events before stimulus start.  Deleted."

    ptd_errors = []
    while any(ptd_rise_diff[ptd_start:ptd_end] < 1.8):
        error_frames = np.where(ptd_rise_diff[ptd_start:ptd_end] < 1.8)[0] + ptd_start
        print "Photodiode error detected. Number of frames:", len(error_frames)
        photodiode_rise = np.delete(photodiode_rise, error_frames[-1])
        ptd_errors.append(photodiode_rise[error_frames[-1]])
        ptd_end -= 1
        ptd_rise_diff = np.ediff1d(photodiode_rise)

    first_pulse = ptd_start
    stim_on_photodiode_idx = 60 + 120 * np.arange(0, ptd_end - ptd_start, 1)

    stim_on_photodiode = stim_vsync_fall[stim_on_photodiode_idx]
    photodiode_on = photodiode_rise[first_pulse + np.arange(0, ptd_end - ptd_start, 1)]
    delay_rise = photodiode_on - stim_on_photodiode

    delay = np.mean(delay_rise[:-1])
    if verbose:
        print "monitor delay: ", delay

    #adjust stimulus time to incorporate monitor delay
    stim_time = stim_vsync_fall + delay

    #convert stimulus frames into twop frames
    twop_frames = np.empty((len(stim_time), 1))
    for i in range(len(stim_time)):
        # crossings = np.nonzero(np.ediff1d(np.sign(twop_vsync_fall - stim_time[i]))>0)
        crossings = np.searchsorted(twop_vsync_fall, stim_time[i], side='left') - 1
        if crossings < (len(twop_vsync_fall)-1):
            twop_frames[i] = crossings
        else:
            twop_frames[i:len(stim_time)] = np.NaN
            warnings.warn(
                'Acquisition ends before stimulus.', RuntimeWarning
            )
            break

    return twop_frames, twop_vsync_fall, stim_vsync_fall, photodiode_rise

def get_center_coordinates(data):

    center_idx = get_stimulus_index(data,'center')
    stim_definition = data['stimuli'][center_idx]['stim']

    position_idx = stim_definition.find('pos=array(')
    coor_start = position_idx + stim_definition[position_idx:].find('[') + 1
    coor_end = position_idx + stim_definition[position_idx:].find(']')
    comma_idx = position_idx + stim_definition[position_idx:].find(',')

    x_coor = float(stim_definition[coor_start:comma_idx])
    y_coor = float(stim_definition[(comma_idx+1):coor_end])

    return x_coor, y_coor


def print_summary(stim_table):
    """Print summary of generated stim_table.

    Print column names, number of 'unique' conditions per column (treating
    nans as equal), and average number of samples per condition.
    """
    print(
        '{:<20}{:>15}{:>15}\n'.format('Colname', 'No. conditions', 'Mean N/cond')
    )
    for colname in stim_table.columns:
        conditions, occurrences = np.unique(
            np.nan_to_num(stim_table[colname]), return_counts = True
        )
        print(
            '{:<20}{:>15}{:>15.1f}'.format(
                colname, len(conditions), np.mean(occurrences)
            )
        )


if __name__ == '__main__':
    exptpath = r'\\allen\programs\braintv\production\neuralcoding\prod55\specimen_859061987\ophys_session_882666374\\'
    lsnCS_create_stim_table(exptpath)
