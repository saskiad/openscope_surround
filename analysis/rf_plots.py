"""Generate receptive field plots for all experiments.

@author: Emerson
"""

#%% IMPORT MODULES

import os
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import h5py

import oscopetools.stim_table as st
from oscopetools import util
import oscopetools.chi_square_lsn as chisq


#%% CREATE STIM TABLES

DATA_PATH = os.path.join(
    '/', 'Volumes', '1848', 'openscope2019_data'
)  # Where data is stored.

sparse = np.load(
    './stimulus/sparse_noise_8x14.npy'
)  # Load stimulus numpy file.

stim_stuff = {}
for specimen in os.listdir(DATA_PATH):
    stim_stuff[specimen] = {}
    for session in os.listdir(os.path.join(DATA_PATH, specimen)):
        stim_stuff[specimen][session] = {}
        tmp_path = os.path.join(DATA_PATH, specimen, session)
        with util.gagProcess():
            try:
                stim_stuff[specimen][session]['tables'] = st.create_stim_tables(
                    tmp_path
                )
                if 'center_surround' in list(
                    stim_stuff[specimen][session]['tables'].keys()
                ):
                    util.populate_columns(
                        stim_stuff[specimen][session]['tables'][
                            'center_surround'
                        ],
                        inplace=True,
                    )
            except IOError:
                warnings.warn(
                    'IOError caught trying to make stim_table from {}'.format(
                        tmp_path
                    )
                )


#%% LOAD EXPERIMENT DESCRIPTIONS

experiments_df = pd.read_csv(os.path.join('analysis', 'manifest.csv'))


#%% GENERATE RF PLOTS

PLOT_PATH = 'plots'  # Where to save output plots.

for specimen in list(stim_stuff.keys()):
    for session in list(stim_stuff[specimen].keys()):
        print(
            (
                'Analyzing RFs for specimen {} session {}'.format(
                    specimen, session
                )
            )
        )

        # Ensure that current session has locally_sparse_noise.
        try:
            tmp_stim_table = stim_stuff[specimen][session]['tables'][
                'locally_sparse_noise'
            ]
        except KeyError:
            continue

        # Open HDF5 file with DF/F traces.
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

        # Collect responses for each cell.
        responses = np.empty(
            (tmp_stim_table.shape[0], roi_traces['data'].shape[0]),
            dtype=np.float32,
        )
        for trial_no in range(tmp_stim_table.shape[0]):
            timeslice = slice(
                int(tmp_stim_table['Start'][trial_no]),
                int(tmp_stim_table['End'][trial_no]),
            )
            responses[trial_no, :] = np.mean(
                roi_traces['data'][:, timeslice], axis=1
            )

        roi_traces.close()  # Close HDF5 file since it's no longer needed.

        # Extract RFs using Dan's code.
        with util.gagProcess():
            p_vals = chisq.chi_square_RFs(
                responses, sparse[tmp_stim_table['Frame'], ...]
            )

        # Make figure to summarize RFs
        fig = plt.figure(figsize=(8, 4))
        plt.suptitle(
            'RF summary: {} {} {}um depth'.format(
                session,
                experiments_df.loc[
                    experiments_df['ophys_sesssion'] == session[-9:],
                    'genotype_name',
                ].values[0],
                experiments_df.loc[
                    experiments_df['ophys_sesssion'] == session[-9:],
                    'imaging_depth',
                ].values[0],
            )
        )

        ax = plt.subplot(221)
        ax.set_title('Pixels with p<0.5')
        mat = ax.matshow(np.sum(p_vals < 0.05, axis=0))
        fig.colorbar(mat, ax=ax, label='No. cells')

        ax = plt.subplot(222)
        ax.set_title('Pixels with lowest p-value')
        mat = ax.matshow(np.sum(chisq.rf_mask(p_vals), axis=0))
        fig.colorbar(mat, ax=ax)

        del ax, mat

        plt.subplot(223)
        plt.title('No. significant pixels per cell')
        plt.hist(p_vals.reshape((p_vals.shape[0])))

        # Ensure there's a place to save a plot.
        if not os.path.exists(os.path.join(PLOT_PATH, specimen, session)):
            os.makedirs(os.path.join(PLOT_PATH, specimen, session))

        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        plt.savefig(
            os.path.join(PLOT_PATH, specimen, session, 'rf_summary.png'),
            dpi=200,
        )
        plt.close('all')
