# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 2019

@author: Emerson Harkin

Tools for plotting traces from ROIs.
"""

#%% IMPORT MODULES

import os
import sys
import warnings

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import h5py
import pandas as pd

import analysis.stim_table as st


#%% DEFINE GAGPROCESS CLASS

class gagProcess(object):
    """Class to forcibly gag verbose methods.

    Temporarily redirects stdout to block print commands.

    >>> with gagProcess:
    >>>     print 'Things that will not be printed.'

    """

    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._original_stdout


#%% TRY CREATING STIM TABLE

DATA_PATH = os.path.join('/', 'Volumes', '1848', 'openscope2019_data')

stim_stuff = {}
for specimen in os.listdir(DATA_PATH):
    stim_stuff[specimen] = {}
    for session in os.listdir(os.path.join(DATA_PATH, specimen)):
        stim_stuff[specimen][session] = {}
        tmp_path = os.path.join(DATA_PATH, specimen, session)
        try:
            with gagProcess():
                stim_stuff[specimen][session]['tables'] = st.lsnCS_create_stim_table(tmp_path)
        except:
            warnings.warn(
                'Problem with specimen {} session {}'.format(specimen, session),
                RuntimeWarning
            )

tmp_stim_table = stim_stuff['specimen_862622776']['ophys_session_890096614']['tables']['center_surround']


#%% DEFINE UTILITY FUNCTIONS

def print_summary(stim_table):
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


@np.vectorize
def nanequal(a, b):
    """Return true if a==b or a and b are both nan.

    Works only for numeric types.

    """
    if a == b or np.isclose(a, b) or (np.isnan(a) and np.isnan(b)):
        return True
    else:
        return False


def nanuniquerows(data, columns):
    """Find unique rows of data based on specified columns.

    Differs from np.unique(data, axis = 0) in that NaNs are considered equal.

    Inputs:
        data (pd.DateFrame-like)
        columns (list of strings)

    Returns:
        Dict with inds of sample unique rows and vals of columns in those rows.

    """
    uniquerows = {'inds': [], 'vals': []}
    for i in range(data.shape[0]):
        i_isunique = True
        for j in range(i + 1, data.shape[0]):
            if all(nanequal(data.loc[i, columns], data.loc[j, columns])):
                i_isunique = False
                break
            else:
                continue
        if i_isunique:
            uniquerows['inds'].append(i)
            uniquerows['vals'].append(data.loc[i, columns])

    uniquerows['vals'] = pd.DataFrame(uniquerows['vals'])
    return uniquerows


def content_rowmask(data, **filter_conds):
    """Find rows of data matching all filter conditions.

    Inputs:
        data (pd.DataFrame-like)
            -- DataFrame from which to generate a mask based on contents.
        filter_conds (arbitrary types)
            -- Conditions on which to filter. For example, to find rows where
            column 'colname' of data has the value 'somevalue', pass
            'colname = somevalue' as an argument.

    Returns:
        Boolean vector that is True for rows of data when all filter_conds are
        True.

    """
    row_mask = np.ones(data.shape[0], dtype = np.bool)
    for key, val in filter_conds.iteritems():
        row_mask = np.logical_and(row_mask, nanequal(data[key], val))
    return row_mask


def populate_columns(data, Mean_Gray = True, No_Surround = True, Ortho = True, inplace = False):
    if not inplace:
        data = np.copy(data)

    if Mean_Gray:
        data['Mean_Gray'] = np.logical_and(
            np.isnan(data['Surround_Ori']),
            np.isnan(data['Center_Ori'])
        )
    if No_Surround:
        data['No_Surround'] = np.logical_and(
            np.isnan(data['Surround_Ori']),
            ~np.isnan(data['Center_Ori'])
        )
    if Ortho:
        data['Ortho'] = np.logical_and(
            np.logical_and(~data['Mean_Gray'], ~data['No_Surround']),
            np.isclose(
                np.abs(data['Center_Ori'] - data['Surround_Ori']) % 180.,
                90.
            )
        )

    return data


populate_columns(tmp_stim_table, inplace = True)
tmp_stim_table.head()

#tmp = nanuniquerows(tmp_stim_table, ['SF', 'TF', 'Center_Ori', 'Surround_Ori'])
#tmp['vals'].loc[np.isnan(tmp['vals']['SF']), :]

#%%

center_only_windows = tmp_stim_table.loc[
    content_rowmask(tmp_stim_table, No_Surround = True, Mean_Gray = False),
    ['Start', 'End', 'Center_Ori']
]
center_only_windows.reset_index(drop = True, inplace = True)
center_only_windows.head()

#%%

PLOT_PATH = 'plots'

roi_traces = h5py.File(
    os.path.join(
        DATA_PATH, 'specimen_862622776', 'ophys_session_890096614',
        'ophys_experiment_891052243', '891052243_dff.h5'
    ),
    'r'
)

try:
    tuning_curves = []

    """
    For each cell, plot:
        - raw traces
        - mean for each orientation
        - tuning curve
    """
    for cell_no in range(roi_traces['data'].shape[0]):
        print('Plotting roi no. {}'.format(cell_no))
        plt.figure(figsize = (8, 4))

        plt.subplot(131)
        plt.title('Raw traces')
        for tr_no in range(center_only_windows.shape[0]):
            plt.plot(
                roi_traces['data'][cell_no, slice(
                    int(center_only_windows.loc[tr_no, 'Start'] - 30),
                    int(center_only_windows.loc[tr_no, 'End'])
                )].T,
                color = plt.cm.viridis(center_only_windows.loc[tr_no, 'Center_Ori']/360.),
                alpha = 0.5
            )
        plt.axvline(30., color = 'k')
        plt.xlabel('Time (timesteps)')

        plt.subplot(132)
        rowlen = np.min(center_only_windows['End'] - center_only_windows['Start']) + 30
        mean_traces = np.empty(
            (len(np.unique(center_only_windows['Center_Ori'])), int(rowlen))
        )
        for i, val in enumerate(np.unique(center_only_windows['Center_Ori'])):
            data_ls = []
            for j, row in center_only_windows.loc[center_only_windows['Center_Ori'] == val, ['Start', 'End']].iterrows():
                data_ls.append(
                    roi_traces['data'][cell_no, slice(int(row['Start']) - 30, int(row['End']))][:int(rowlen)]
                )
            mean_traces[i, :] = np.mean(data_ls, axis = 0)
            plt.plot(mean_traces[i, :], label = val)
        plt.axvline(30., color = 'k')
        plt.xlabel('Time (timesteps)')

        plt.legend()

        tmp_tuning_curve = np.mean(mean_traces[:, 30:], axis = 1)
        tuning_curves.append(tmp_tuning_curve)

        plt.subplot(133)
        plt.plot(
            np.unique(center_only_windows['Center_Ori']),
            tmp_tuning_curve,
            'k-', label = 'Mean resp.'
        )
        plt.plot(
            np.unique(center_only_windows['Center_Ori']),
            np.max(mean_traces[:, 30:], axis = 1),
            '-', color = 'gray', label = 'Max resp.'
        )
        plt.ylabel('Response')
        plt.xlabel('Orientation')
        plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_PATH, 'roi{}.png'.format(cell_no)), dpi = 200)
        plt.close()

finally:
    roi_traces.close()

#%% PLOT TUNING CURVES AS MATRICES

PLOT_PATH = 'plots'

roi_traces = h5py.File(
    os.path.join(
        DATA_PATH, 'specimen_862622776', 'ophys_session_890096614',
        'ophys_experiment_891052243', '891052243_dff.h5'
    ),
    'r'
)

try:
    tuning_curves = {
        'center_only': [],
        'iso_surround': [],
        'ortho_surround': []
    }

    """
    For each cell, plot:
        - raw traces
        - mean for each orientation
        - tuning curve
    """
    for cell_no in range(roi_traces['data'].shape[0]):
        print('Plotting roi no. {}'.format(cell_no))
        plt.figure(figsize = (12, 6))

        spec_outer = gs.GridSpec(3, 2, width_ratios = [3, 1])
        spec_traces = {
            "center_only": gs.GridSpecFromSubplotSpec(1, 8, spec_outer[0, 0]),
            "iso_surround": gs.GridSpecFromSubplotSpec(1, 8, spec_outer[1, 0]),
            "ortho_surround": gs.GridSpecFromSubplotSpec(1, 8, spec_outer[2, 0])
        }
        spec_tuning = gs.GridSpecFromSubplotSpec(2, 1, spec_outer[:, 1])

        tuning_ax = plt.subplot(spec_tuning[0, :])
        plt.ylabel('Mean resp.')
        plt.xlabel('Orientation')

        windows = {
            'center_only': tmp_stim_table.loc[
                content_rowmask(
                    tmp_stim_table, No_Surround = True, Mean_Gray = False
                ),
                ['Start', 'End', 'Center_Ori']
            ],
            'iso_surround': tmp_stim_table.loc[
                content_rowmask(
                    tmp_stim_table,
                    Ortho = False, No_Surround = False, Mean_Gray = False
                ),
                ['Start', 'End', 'Center_Ori']
            ],
            'ortho_surround': tmp_stim_table.loc[
                content_rowmask(
                    tmp_stim_table,
                    Ortho = True, No_Surround = False, Mean_Gray = False
                ),
                ['Start', 'End', 'Center_Ori']
            ],
        }
        windows = {
            key: val.reset_index(drop = True) for key, val in windows.iteritems()
        }

        # Plot raw traces.
        raw_traces = {}
        mean_traces = {}
        for cond_no, condition in enumerate(windows.keys()):
            raw_traces[condition] = {}
            mean_traces[condition] = []
            for i, orientation in enumerate(np.unique(windows[condition]['Center_Ori'])):
                if np.isnan(orientation):
                    warnings.warn(
                        'NaNs detected in orientations for condition {}'.format(
                            condition
                        )
                    )
                    continue

                # Get start and stop times for each trial with given
                # orientation and condition.
                tmp_windows = windows[condition].loc[
                    content_rowmask(
                        windows[condition], Center_Ori = orientation
                    ),
                    ['Start', 'End']
                ].reset_index(drop = True)

                # Get traces and make plots simultaneously.
                raw_traces[condition][str(orientation)] = []

                if i == 0:
                    firstax = plt.subplot(spec_traces[condition][:, i])
                    plt.title('{}\n{}'.format(condition, orientation))
                    plt.xlabel('Time (timesteps)')
                else:
                    plt.subplot(spec_traces[condition][:, i], sharey = firstax)
                    plt.title(str(orientation))

                for tr_no in range(tmp_windows.shape[0]):
                    # Get trace.
                    raw_traces[condition][str(orientation)].append(
                        roi_traces['data'][cell_no, slice(
                            int(tmp_windows.loc[tr_no, 'Start'] - 30),
                            int(tmp_windows.loc[tr_no, 'End'])
                        )].T
                    )
                    # Plot most recent trace.
                    plt.plot(
                        raw_traces[condition][str(orientation)][-1],
                        'k-', lw = 0.5, alpha = 0.5
                    )

                # Truncate raw traces.
                min_trlen = min(
                    [len(x) for x in raw_traces[condition][str(orientation)]]
                )
                raw_traces[condition][str(orientation)] = np.array([
                    x[:min_trlen] for x in raw_traces[condition][str(orientation)]
                ])

                # Overplot mean.
                mean_traces[condition].append(
                    raw_traces[condition][str(orientation)].mean(axis = 0)
                )
                plt.plot(
                    mean_traces[condition][-1],  # Mean = last added tr.
                    'r-', lw = 2, alpha = 0.7
                )

                if i != 0:
                    plt.setp(plt.gca().get_yticklabels(), visible=False)
                plt.axvline(30., color = 'gray', ls = '--')

            # Compute tuning curve and store.
            mean_traces[condition] = np.array(mean_traces[condition])
            tmp_tuning_curve = np.mean(mean_traces[condition][:, 30:], axis = 1)
            tuning_curves[condition].append(tmp_tuning_curve)

            tmp_orientations = np.unique(windows[condition]['Center_Ori'])
            tmp_orientations = tmp_orientations[~np.isnan(tmp_orientations)]
            tuning_ax.plot(
                tmp_orientations,
                tmp_tuning_curve,
                label = condition
            )

        tuning_ax.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_PATH, 'newroi{}.png'.format(cell_no)), dpi = 200)
        plt.close()

finally:
    roi_traces.close()


#%% PLOT ALL TUNING CURVES

def normalize(x):
    """Normalize a vector x to the unit inverval."""
    normed = (
        (x - x.min(axis=1)[:, np.newaxis])
        / (x - x.min(axis=1)[:, np.newaxis]).max(axis=1)[:, np.newaxis]
    )
    return normed


def sparsity(x, axis = -1):
    """Compute the ratio of 2 and 1 norms of x."""
    sparsity_ = (
        np.linalg.norm(x, ord = 2, axis = axis)
        / np.linalg.norm(x, ord = 1, axis = axis)
    )
    return sparsity_


tuning_curves = {key: np.array(val) for key, val in tuning_curves.iteritems()}

fig = plt.figure(figsize = (8, 8))

tmp_tuning_df = pd.DataFrame({
    'Pref_Ori': np.argmax(tuning_curves['center_only'], axis = 1),
    'Sparsity': sparsity(tuning_curves['center_only'], axis = 1),
    'All_nan': np.all(np.isnan(tuning_curves['center_only']), axis = 1)
})
inds = tmp_tuning_df.sort_values(['All_nan', 'Pref_Ori', 'Sparsity']).index

for i, cond in enumerate(['center_only', 'iso_surround', 'ortho_surround']):

    ax = plt.subplot(3, 3, i*3 + 1)
    plt.title('{}\nMean response'.format(cond))
    aa = ax.matshow(tuning_curves[cond][inds, :], aspect = 'auto')
    fig.colorbar(aa, ax = ax, label = r'Mean $\frac{\Delta F}{F}$')
    plt.xticks([])
    plt.xlabel('Orientation')
    plt.ylabel('ROI no.')

    ax = plt.subplot(3, 3, i*3 + 2)
    plt.title('Normalized response')
    ax.matshow(
        normalize(tuning_curves[cond][inds, :]),
        aspect = 'auto'
    )
    plt.yticks([])
    plt.xticks([])
    plt.xlabel('Orientation')

    plt.subplot(3, 3, i*3 + 3)
    plt.title('Response sparsity')
    plt.plot(sparsity(tuning_curves[cond][inds, :]), np.arange(0, tmp_tuning_df.shape[0]))
    plt.ylim(tmp_tuning_df.shape[0], 0)
    plt.yticks([])
    plt.xlabel(r'$\Vert \mathbf{x} \Vert_2 / \Vert \mathbf{x} \Vert_1$')

    plt.tight_layout()

plt.savefig(os.path.join(PLOT_PATH, 'newtuning_summary.png'), dpi = 200)
plt.show()
