#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 10:59:54 2018

@author: saskiad
"""

import numpy as np
import pandas as pd
import os, h5py

# import core
from .stim_table import *


def do_sweep_mean(x):
    return x[28:35].mean()


def do_sweep_mean_shifted(x):
    return x[30:40].mean()


class LocallySparseNoise:
    def __init__(self, expt_path):

        self.expt_path = expt_path
        self.session_id = self.expt_path.split('/')[
            -1
        ]  # this might need to be modified for where the data is for you.

        # load dff traces
        for f in os.listdir(self.expt_path):
            if f.endswith('_dff.h5'):
                dff_path = os.path.join(self.expt_path, f)
        f = h5py.File(dff_path, 'r')
        self.dff = f['data'][()]
        f.close()
        self.numbercells = self.dff.shape[0]

        # create stimulus table for locally sparse noise
        stim_dict = lsnCS_create_stim_table(self.expt_path)
        self.stim_table = stim_dict['locally_sparse_noise']

        # load stimulus template
        self.LSN = np.load(lsn_path)

        # run analysis
        (
            self.sweep_response,
            self.mean_sweep_response,
            self.sweep_p_values,
            self.response_on,
            self.response_off,
        ) = self.get_stimulus_response(self.LSN)
        self.peak = self.get_peak()

        # save outputs
        self.save_data()

    def get_stimulus_response(self, LSN):
        '''calculates the response to each stimulus trial. Calculates the mean response to each stimulus condition.

Returns
-------
sweep response: full trial for each trial
mean sweep response: mean response for each trial
sweep p values: p value of each trial compared measured relative to distribution of spontaneous activity
response_on: mean response, s.e.m., and number of responsive trials for each white square
response_off: mean response, s.e.m., and number of responsive trials for each black square


        '''
        sweep_response = pd.DataFrame(
            index=self.stim_table.index.values,
            columns=np.array(list(range(self.numbercells))).astype(str),
        )

        for index, row in self.stim_table.iterrows():
            for nc in range(self.numbercells):
                sweep_response[str(nc)][index] = self.dff[
                    nc, int(row.Start) - 28 : int(row.Start) + 35
                ]

        mean_sweep_response = sweep_response.applymap(do_sweep_mean_shifted)

        # make spontaneous p_values
        # TODO: pilot stimulus does not have spontaneous activity. But real data will and we shoudl re-implement this
        #        shuffled_responses = np.empty((self.numbercells, 10000,10))
        #        idx = np.random.choice(range(self.stim_table_sp.start[0], self.stim_table_sp.end[0]), 10000)
        #        for i in range(10):
        #            shuffled_responses[:,:,i] = self.l0_events[:,idx+i]
        #        shuffled_mean = shuffled_responses.mean(axis=2)
        sweep_p_values = pd.DataFrame(
            index=self.stim_table.index.values,
            columns=np.array(list(range(self.numbercells))).astype(str),
        )
        #        for nc in range(self.numbercells):
        #            subset = mean_sweep_events[str(nc)].values
        #            null_dist_mat = np.tile(shuffled_mean[nc,:], reps=(len(subset),1))
        #            actual_is_less = subset.reshape(len(subset),1) <= null_dist_mat
        #            p_values = np.mean(actual_is_less, axis=1)
        #            sweep_p_values[str(nc)] = p_values

        x_shape = LSN.shape[1]
        y_shape = LSN.shape[2]
        response_on = np.empty((x_shape, y_shape, self.numbercells, 3))
        response_off = np.empty((x_shape, y_shape, self.numbercells, 3))
        for xp in range(x_shape):
            for yp in range(y_shape):
                on_frame = np.where(LSN[:, xp, yp] == 255)[0]
                off_frame = np.where(LSN[:, xp, yp] == 0)[0]
                subset_on = mean_sweep_response[
                    self.stim_table.Frame.isin(on_frame)
                ]
                #                subset_on_p = sweep_p_values[self.stim_table.frame.isin(on_frame)]
                subset_off = mean_sweep_response[
                    self.stim_table.Frame.isin(off_frame)
                ]
                #                subset_off_p = sweep_p_values[self.stim_table.frame.isin(off_frame)]
                response_on[xp, yp, :, 0] = subset_on.mean(axis=0)
                response_on[xp, yp, :, 1] = subset_on.std(axis=0) / np.sqrt(
                    len(subset_on)
                )
                #                response_on[xp,yp,:,2] = subset_on_p[subset_on_p<0.05].count().values/float(len(subset_on_p))
                response_off[xp, yp, :, 0] = subset_off.mean(axis=0)
                response_off[xp, yp, :, 1] = subset_off.std(axis=0) / np.sqrt(
                    len(subset_off)
                )
        #                response_off[xp,yp,:,2] = subset_off_p[subset_off_p<0.05].count().values/float(len(subset_off_p))
        return (
            sweep_response,
            mean_sweep_response,
            sweep_p_values,
            response_on,
            response_off,
        )

    def get_peak(self):
        '''creates a table of metrics for each cell. We can make this more useful in the future

Returns
-------
peak dataframe
        '''
        peak = pd.DataFrame(
            columns=('rf_on', 'rf_off'), index=list(range(self.numbercells))
        )
        peak['rf_on'] = False
        peak['rf_off'] = False
        on_rfs = np.where(self.response_events_on[:, :, :, 2] > 0.25)[2]
        off_rfs = np.where(self.response_events_off[:, :, :, 2] > 0.25)[2]
        peak.rf_on.loc[on_rfs] = True
        peak.rf_off.loc[off_rfs] = True
        return peak

    def save_data(self):
        save_file = os.path.join(
            self.expt_path, str(self.session_id) + "_lsn_analysis.h5"
        )
        print("Saving data to: ", save_file)
        store = pd.HDFStore(save_file)
        store['sweep_response'] = self.sweep_events
        store['mean_sweep_response'] = self.mean_sweep_events
        store['sweep_p_values'] = self.sweep_p_values
        store['peak'] = self.peak
        store.close()
        f = h5py.File(save_file, 'r+')
        dset = f.create_dataset('response_on', data=self.response_events_on)
        dset1 = f.create_dataset('response_off', data=self.response_events_off)
        f.close()


if __name__ == '__main__':
    lsn_path = r'/Users/saskiad/Code/openscope_surround/stimulus/sparse_noise_8x14.npy'  # update this to local path the the stimulus array
    expt_path = r'/Volumes/My Passport/Openscope Multiplex/891653201'
    lsn = LocallySparseNoise(expt_path=expt_path)
