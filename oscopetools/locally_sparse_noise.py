#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 10:59:54 2018

@author: saskiad
"""

import numpy as np
import pandas as pd
import os, h5py
<<<<<<< Updated upstream:oscopetools/locally_sparse_noise.py

# import core
from .stim_table import *

=======
import matplotlib.pyplot as plt
>>>>>>> Stashed changes:analysis/locally_sparse_noise.py

def do_sweep_mean(x):
    return x[28:35].mean()


def do_sweep_mean_shifted(x):
    return x[30:40].mean()

<<<<<<< Updated upstream:oscopetools/locally_sparse_noise.py
=======
def do_eye(x):
    return x[28:32].mean()
>>>>>>> Stashed changes:analysis/locally_sparse_noise.py

class LocallySparseNoise:
    def __init__(self, expt_path):

        self.expt_path = expt_path
<<<<<<< Updated upstream:oscopetools/locally_sparse_noise.py
        self.session_id = self.expt_path.split('/')[
            -1
        ]  # this might need to be modified for where the data is for you.

        # load dff traces
        for f in os.listdir(self.expt_path):
            if f.endswith('_dff.h5'):
                dff_path = os.path.join(self.expt_path, f)
        f = h5py.File(dff_path, 'r')
        self.dff = f['data'][()]
=======
        self.session_id = self.expt_path.split('/')[-1].split('_')[-2]

        #load dff traces
        f = h5py.File(self.expt_path, 'r')
        self.dff = f['dff_traces'][()]
>>>>>>> Stashed changes:analysis/locally_sparse_noise.py
        f.close()

        self.numbercells = self.dff.shape[0]
<<<<<<< Updated upstream:oscopetools/locally_sparse_noise.py

        # create stimulus table for locally sparse noise
        stim_dict = lsnCS_create_stim_table(self.expt_path)
        self.stim_table = stim_dict['locally_sparse_noise']
=======

        #create stimulus table for locally sparse noise
        self.stim_table = pd.read_hdf(self.expt_path, 'locally_sparse_noise')
>>>>>>> Stashed changes:analysis/locally_sparse_noise.py

        # load stimulus template
        self.LSN = np.load(lsn_path)
<<<<<<< Updated upstream:oscopetools/locally_sparse_noise.py

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
=======

        #load eyetracking
        self.pupil_pos = pd.read_hdf(self.expt_path, 'eye_tracking')

        #run analysis
        self.sweep_response, self.mean_sweep_response, self.response_on, self.response_off, self.sweep_eye, self.mean_sweep_eye = self.get_stimulus_response(self.LSN)
        self.peak = self.get_peak()

        #save outputs
#        self.save_data()

        #plot traces
        self.plot_LSN_Traces()
>>>>>>> Stashed changes:analysis/locally_sparse_noise.py

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
<<<<<<< Updated upstream:oscopetools/locally_sparse_noise.py
        sweep_response = pd.DataFrame(
            index=self.stim_table.index.values,
            columns=np.array(list(range(self.numbercells))).astype(str),
        )

        for index, row in self.stim_table.iterrows():
            for nc in range(self.numbercells):
                sweep_response[str(nc)][index] = self.dff[
                    nc, int(row.Start) - 28 : int(row.Start) + 35
                ]
=======
        sweep_response = pd.DataFrame(index=self.stim_table.index.values, columns=np.array(range(self.numbercells)).astype(str))

        sweep_eye = pd.DataFrame(index=self.stim_table.index.values, columns=('x_pos_deg','y_pos_deg'))

        for index,row in self.stim_table.iterrows():
            for nc in range(self.numbercells):
                sweep_response[str(nc)][index] = self.dff[nc, int(row.Start)-28:int(row.Start)+35]
            sweep_eye.x_pos_deg[index] = self.pupil_pos.x_pos_deg[int(row.Start)-28:int(row.Start+35)].values
            sweep_eye.y_pos_deg[index] = self.pupil_pos.y_pos_deg[int(row.Start)-28:int(row.Start+35)].values
>>>>>>> Stashed changes:analysis/locally_sparse_noise.py

        mean_sweep_response = sweep_response.applymap(do_sweep_mean_shifted)
        mean_sweep_eye = sweep_eye.applymap(do_eye)


<<<<<<< Updated upstream:oscopetools/locally_sparse_noise.py
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

=======

>>>>>>> Stashed changes:analysis/locally_sparse_noise.py
        x_shape = LSN.shape[1]
        y_shape = LSN.shape[2]
        response_on = np.empty((x_shape, y_shape, self.numbercells, 2))
        response_off = np.empty((x_shape, y_shape, self.numbercells, 2))
        for xp in range(x_shape):
            for yp in range(y_shape):
<<<<<<< Updated upstream:oscopetools/locally_sparse_noise.py
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

=======
                on_frame = np.where(LSN[:,xp,yp]==255)[0]
                off_frame = np.where(LSN[:,xp,yp]==0)[0]
                subset_on = mean_sweep_response[self.stim_table.Frame.isin(on_frame)]
                subset_off = mean_sweep_response[self.stim_table.Frame.isin(off_frame)]
                response_on[xp,yp,:,0] = subset_on.mean(axis=0)
                response_on[xp,yp,:,1] = subset_on.std(axis=0)/np.sqrt(len(subset_on))
                response_off[xp,yp,:,0] = subset_off.mean(axis=0)
                response_off[xp,yp,:,1] = subset_off.std(axis=0)/np.sqrt(len(subset_off))
        return sweep_response, mean_sweep_response, response_on, response_off, sweep_eye, mean_sweep_eye

>>>>>>> Stashed changes:analysis/locally_sparse_noise.py
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
<<<<<<< Updated upstream:oscopetools/locally_sparse_noise.py
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
=======
        on_rfs = np.where(self.response_on[:,:,:,2]>0.25)[2]
        off_rfs = np.where(self.response_off[:,:,:,2]>0.25)[2]
        peak.rf_on.loc[on_rfs] = True
        peak.rf_off.loc[off_rfs] = True
        return peak

    def save_data(self):
        '''saves intermediate analysis files in an h5 file'''
        save_file = os.path.join(r'/Users/saskiad/Documents/Data/Openscope_Multiplex/analysis', str(self.session_id)+"_lsn_analysis.h5")
        print "Saving data to: ", save_file
>>>>>>> Stashed changes:analysis/locally_sparse_noise.py
        store = pd.HDFStore(save_file)
        store['sweep_response'] = self.sweep_response
        store['mean_sweep_response'] = self.mean_sweep_response
        store['sweep_p_values'] = self.sweep_p_values
        store['peak'] = self.peak
        store.close()
        f = h5py.File(save_file, 'r+')
        dset = f.create_dataset('response_on', data=self.response_on)
        dset1 = f.create_dataset('response_off', data=self.response_off)
        f.close()
<<<<<<< Updated upstream:oscopetools/locally_sparse_noise.py


if __name__ == '__main__':
    lsn_path = r'/Users/saskiad/Code/openscope_surround/stimulus/sparse_noise_8x14.npy'  # update this to local path the the stimulus array
    expt_path = r'/Volumes/My Passport/Openscope Multiplex/891653201'
    lsn = LocallySparseNoise(expt_path=expt_path)
=======

    def plot_LSN_Traces(self):
        '''plots ON and OFF traces for each position for each cell'''
        print "Plotting LSN traces for all cells"

        for nc in range(self.numbercells):
            if np.mod(nc,100)==0:
                print "Cell #", str(nc)
            plt.figure(nc, figsize=(24,20))
            vmax=0
            vmin=0
            one_cell = self.sweep_response[str(nc)]
            for yp in range(8):
                for xp in range(14):
                    sp_pt = (yp*14)+xp+1
                    on_frame = np.where(self.LSN[:,yp,xp]==255)[0]
                    off_frame = np.where(self.LSN[:,yp,xp]==0)[0]
                    subset_on = one_cell[self.stim_table.Frame.isin(on_frame)]
                    subset_off = one_cell[self.stim_table.Frame.isin(off_frame)]
                    ax = plt.subplot(8,14,sp_pt)
                    ax.plot(subset_on.mean(), color='r', lw=2)
                    ax.plot(subset_off.mean(), color='b', lw=2)
                    ax.axvspan(28,35 ,ymin=0, ymax=1, facecolor='gray', alpha=0.3)
                    vmax = np.where(np.amax(subset_on.mean())>vmax, np.amax(subset_on.mean()), vmax)
                    vmax = np.where(np.amax(subset_off.mean())>vmax, np.amax(subset_off.mean()), vmax)
                    vmin = np.where(np.amin(subset_on.mean())<vmin, np.amin(subset_on.mean()), vmin)
                    vmin = np.where(np.amin(subset_off.mean())<vmin, np.amin(subset_off.mean()), vmin)
                    ax.set_xticks([])
                    ax.set_yticks([])
            for i in range(1,sp_pt+1):
                ax = plt.subplot(8,14,i)
                ax.set_ylim(vmin, vmax)

            plt.tight_layout()
            plt.suptitle("Cell " + str(nc+1), fontsize=20)
            plt.subplots_adjust(top=0.9)
            filename = 'Traces LSN Cell_'+str(nc+1)+'.png'
            fullfilename = os.path.join(r'/Users/saskiad/Documents/Data/Openscope_Multiplex/analysis', filename)
            plt.savefig(fullfilename)
            plt.close()


if __name__=='__main__':
    lsn_path = r'/Users/saskiad/Code/openscope_surround/stimulus/sparse_noise_8x14.npy' #update this to local path the the stimulus array
    expt_path = r'/Users/saskiad/Dropbox/Openscope Multiplex/Center Surround/Center_Surround_1010436210_data.h5'
    lsn = LocallySparseNoise(expt_path=expt_path)
>>>>>>> Stashed changes:analysis/locally_sparse_noise.py
