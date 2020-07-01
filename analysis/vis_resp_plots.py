# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 2019

@author: Emerson Harkin

Tools for plotting traces from ROIs.
"""

#%% IMPORT MODULES

import os
import warnings

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import h5py
import pandas as pd

import oscopetools.stim_table as st
from oscopetools import util


#%% DEFINE USEFUL FUNCTIONS FOR OPERATING ON TUNING CURVE MATRICES


def normalize(x):
    """Normalize a vector x to the unit inverval."""
    normed = (x - x.min(axis=1)[:, np.newaxis]) / (
        x - x.min(axis=1)[:, np.newaxis]
    ).max(axis=1)[:, np.newaxis]
    return normed


def sparsity(x, axis=-1):
    """Compute the ratio of 2 and 1 norms of x."""
    sparsity_ = np.linalg.norm(x, ord=2, axis=axis) / np.linalg.norm(
        x, ord=1, axis=axis
    )
    return sparsity_


#%% TRY CREATING STIM TABLE

DATA_PATH = os.path.join('/', 'Volumes', '1848', 'openscope2019_data')

stim_stuff = {}
for specimen in os.listdir(DATA_PATH):
    stim_stuff[specimen] = {}
    for session in os.listdir(os.path.join(DATA_PATH, specimen)):
        stim_stuff[specimen][session] = {}
        tmp_path = os.path.join(DATA_PATH, specimen, session)
        try:
            with util.gagProcess():
                stim_stuff[specimen][session]['tables'] = st.create_stim_tables(
                    tmp_path, ['center_surround']
                )
                util.populate_columns(
                    stim_stuff[specimen][session]['tables']['center_surround'],
                    inplace=True,
                )
        except:
            warnings.warn(
                'Problem with specimen {} session {}'.format(specimen, session),
                RuntimeWarning,
            )


#%% PLOT TRACES

PLOT_PATH = 'plots'

for specimen in list(stim_stuff.keys()):
    for session in list(stim_stuff[specimen].keys()):
        # Skip is this session wasn't actually loaded above, or doesn't exist.
        try:
            stim_stuff[specimen][session]['tables']  # Check whether loaded.
            os.listdir(
                os.path.join(DATA_PATH, specimen, session)
            )  # Check whether exists.
        except KeyError as OSError:
            continue

        tmp_stim_table = stim_stuff[specimen][session]['tables'][
            'center_surround'
        ]
        tmp_plot_path = os.path.join(PLOT_PATH, specimen, session)
        if not os.path.exists(tmp_plot_path):
            os.makedirs(tmp_plot_path)  # Ensure that appropriate dir exists.

        roi_traces = h5py.File(
            util.find_by_suffix(
                util.find_by_prefix(
                    os.path.join(DATA_PATH, specimen, session),
                    'ophys_experiment',
                    True,
                ),
                '_dff.h5',
                True,
            ),
            'r',
        )

        try:
            tuning_curves = {
                'center_only': [],
                'iso_surround': [],
                'ortho_surround': [],
            }

            """
            For each cell, plot:
                - raw traces
                - mean for each orientation
                - tuning curve
            """
            for cell_no in range(roi_traces['data'].shape[0]):
                print(('Plotting roi no. {}'.format(cell_no)))
                plt.figure(figsize=(12, 6))

                spec_outer = gs.GridSpec(3, 2, width_ratios=[3, 1])
                spec_traces = {
                    "center_only": gs.GridSpecFromSubplotSpec(
                        1, 8, spec_outer[0, 0]
                    ),
                    "iso_surround": gs.GridSpecFromSubplotSpec(
                        1, 8, spec_outer[1, 0]
                    ),
                    "ortho_surround": gs.GridSpecFromSubplotSpec(
                        1, 8, spec_outer[2, 0]
                    ),
                }
                spec_tuning = gs.GridSpecFromSubplotSpec(2, 1, spec_outer[:, 1])

                tuning_ax = plt.subplot(spec_tuning[0, :])
                plt.ylabel('Mean resp.')
                plt.xlabel('Orientation')

                windows = {
                    'center_only': tmp_stim_table.loc[
                        util.content_rowmask(
                            tmp_stim_table, No_Surround=True, Mean_Gray=False
                        ),
                        ['Start', 'End', 'Center_Ori'],
                    ],
                    'iso_surround': tmp_stim_table.loc[
                        util.content_rowmask(
                            tmp_stim_table,
                            Ortho=False,
                            No_Surround=False,
                            Mean_Gray=False,
                        ),
                        ['Start', 'End', 'Center_Ori'],
                    ],
                    'ortho_surround': tmp_stim_table.loc[
                        util.content_rowmask(
                            tmp_stim_table,
                            Ortho=True,
                            No_Surround=False,
                            Mean_Gray=False,
                        ),
                        ['Start', 'End', 'Center_Ori'],
                    ],
                }
                windows = {
                    key: val.reset_index(drop=True)
                    for key, val in windows.items()
                }

                # Plot raw traces.
                raw_traces = {}
                mean_traces = {}
                for cond_no, condition in enumerate(windows.keys()):
                    raw_traces[condition] = {}
                    mean_traces[condition] = []
                    for i, orientation in enumerate(
                        np.unique(windows[condition]['Center_Ori'])
                    ):
                        if np.isnan(orientation):
                            warnings.warn(
                                'NaNs detected in orientations for condition {}'.format(
                                    condition
                                )
                            )
                            continue

                        # Get start and stop times for each trial with given
                        # orientation and condition.
                        tmp_windows = (
                            windows[condition]
                            .loc[
                                util.content_rowmask(
                                    windows[condition], Center_Ori=orientation
                                ),
                                ['Start', 'End'],
                            ]
                            .reset_index(drop=True)
                        )

                        # Get traces and make plots simultaneously.
                        raw_traces[condition][str(orientation)] = []

                        if i == 0:
                            firstax = plt.subplot(spec_traces[condition][:, i])
                            plt.title('{}\n{}'.format(condition, orientation))
                            plt.xlabel('Time (timesteps)')
                        else:
                            plt.subplot(
                                spec_traces[condition][:, i], sharey=firstax
                            )
                            plt.title(str(orientation))

                        for tr_no in range(tmp_windows.shape[0]):
                            # Get trace.
                            raw_traces[condition][str(orientation)].append(
                                roi_traces['data'][
                                    cell_no,
                                    slice(
                                        int(
                                            tmp_windows.loc[tr_no, 'Start'] - 30
                                        ),
                                        int(tmp_windows.loc[tr_no, 'End']),
                                    ),
                                ].T
                            )
                            # Plot most recent trace.
                            plt.plot(
                                raw_traces[condition][str(orientation)][-1],
                                'k-',
                                lw=0.5,
                                alpha=0.5,
                            )

                        # Truncate raw traces.
                        min_trlen = min(
                            [
                                len(x)
                                for x in raw_traces[condition][str(orientation)]
                            ]
                        )
                        raw_traces[condition][str(orientation)] = np.array(
                            [
                                x[:min_trlen]
                                for x in raw_traces[condition][str(orientation)]
                            ]
                        )

                        # Overplot mean.
                        mean_traces[condition].append(
                            raw_traces[condition][str(orientation)].mean(axis=0)
                        )
                        plt.plot(
                            mean_traces[condition][-1],  # Mean = last added tr.
                            'r-',
                            lw=2,
                            alpha=0.7,
                        )

                        if i != 0:
                            plt.setp(plt.gca().get_yticklabels(), visible=False)
                        plt.axvline(30.0, color='gray', ls='--')

                    # Compute tuning curve and store.
                    mean_traces[condition] = np.array(mean_traces[condition])
                    tmp_tuning_curve = np.mean(
                        mean_traces[condition][:, 30:], axis=1
                    )
                    tuning_curves[condition].append(tmp_tuning_curve)

                    tmp_orientations = np.unique(
                        windows[condition]['Center_Ori']
                    )
                    tmp_orientations = tmp_orientations[
                        ~np.isnan(tmp_orientations)
                    ]
                    tuning_ax.plot(
                        tmp_orientations, tmp_tuning_curve, label=condition
                    )

                tuning_ax.legend()

                plt.tight_layout()
                plt.savefig(
                    os.path.join(tmp_plot_path, 'roi{}.png'.format(cell_no)),
                    dpi=200,
                )
                plt.close()

        finally:
            roi_traces.close()

        #%% PLOT ALL TUNING CURVES

        tuning_curves = {
            key: np.array(val) for key, val in tuning_curves.items()
        }

        fig = plt.figure(figsize=(8, 8))

        tmp_tuning_df = pd.DataFrame(
            {
                'Pref_Ori': np.argmax(tuning_curves['center_only'], axis=1),
                'Sparsity': sparsity(tuning_curves['center_only'], axis=1),
                'All_nan': np.all(
                    np.isnan(tuning_curves['center_only']), axis=1
                ),
            }
        )
        inds = tmp_tuning_df.sort_values(
            ['All_nan', 'Pref_Ori', 'Sparsity']
        ).index

        for i, cond in enumerate(
            ['center_only', 'iso_surround', 'ortho_surround']
        ):

            ax = plt.subplot(3, 3, i * 3 + 1)
            plt.title('{}\nMean response'.format(cond))
            aa = ax.matshow(tuning_curves[cond][inds, :], aspect='auto')
            fig.colorbar(aa, ax=ax, label=r'Mean $\frac{\Delta F}{F}$')
            plt.xticks([])
            plt.xlabel('Orientation')
            plt.ylabel('ROI no.')

            ax = plt.subplot(3, 3, i * 3 + 2)
            plt.title('Normalized response')
            ax.matshow(normalize(tuning_curves[cond][inds, :]), aspect='auto')
            plt.yticks([])
            plt.xticks([])
            plt.xlabel('Orientation')

            plt.subplot(3, 3, i * 3 + 3)
            plt.title('Response sparsity')
            plt.plot(
                sparsity(tuning_curves[cond][inds, :]),
                np.arange(0, tmp_tuning_df.shape[0]),
            )
            plt.ylim(tmp_tuning_df.shape[0], 0)
            plt.yticks([])
            plt.xlabel(r'$\Vert \mathbf{x} \Vert_2 / \Vert \mathbf{x} \Vert_1$')

            plt.tight_layout()

        plt.savefig(os.path.join(tmp_plot_path, 'tuning_summary.png'), dpi=200)
        plt.show()
