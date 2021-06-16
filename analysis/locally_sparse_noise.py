#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 10:59:54 2018

@author: saskiad
"""

import numpy as np
import pandas as pd
import os, h5py
import matplotlib.pyplot as plt

def do_sweep_mean(x):
    return x[28:35].mean()

def do_sweep_mean_shifted(x):
    return x[30:40].mean()

def do_eye(x):
    return x[28:32].mean()

class LocallySparseNoise:
    def __init__(self, expt_path):

        self.expt_path = expt_path
        self.session_id = self.expt_path.split('/')[-1].split('_')[-2]
        
        #load dff traces 
        f = h5py.File(self.expt_path, 'r')
        self.dff = f['dff_traces'][()]
        f.close()

        self.numbercells = self.dff.shape[0]
        
        #create stimulus table for locally sparse noise
        self.stim_table = pd.read_hdf(self.expt_path, 'locally_sparse_noise')

        #load stimulus template
        self.LSN = np.load(lsn_path)
        
        #load eyetracking
        self.pupil_pos = pd.read_hdf(self.expt_path, 'eye_tracking')
        
        #run analysis
        self.sweep_response, self.mean_sweep_response, self.response_on, self.response_off, self.sweep_eye, self.mean_sweep_eye = self.get_stimulus_response(self.LSN)
        self.peak = self.get_peak()
        
        #save outputs
#        self.save_data()
        
        #plot traces
        self.plot_LSN_Traces()

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
        sweep_response = pd.DataFrame(index=self.stim_table.index.values, columns=np.array(range(self.numbercells)).astype(str))
        
        sweep_eye = pd.DataFrame(index=self.stim_table.index.values, columns=('x_pos_deg','y_pos_deg'))
        
        for index,row in self.stim_table.iterrows():
            for nc in range(self.numbercells):
                sweep_response[str(nc)][index] = self.dff[nc, int(row.Start)-28:int(row.Start)+35]
            sweep_eye.x_pos_deg[index] = self.pupil_pos.x_pos_deg[int(row.Start)-28:int(row.Start+35)].values
            sweep_eye.y_pos_deg[index] = self.pupil_pos.y_pos_deg[int(row.Start)-28:int(row.Start+35)].values

        mean_sweep_response = sweep_response.applymap(do_sweep_mean_shifted)
        mean_sweep_eye = sweep_eye.applymap(do_eye)
        

        
        x_shape = LSN.shape[1]
        y_shape = LSN.shape[2]
        response_on = np.empty((x_shape, y_shape, self.numbercells, 2))
        response_off = np.empty((x_shape, y_shape, self.numbercells, 2))
        for xp in range(x_shape):
            for yp in range(y_shape):
                on_frame = np.where(LSN[:,xp,yp]==255)[0]
                off_frame = np.where(LSN[:,xp,yp]==0)[0]
                subset_on = mean_sweep_response[self.stim_table.Frame.isin(on_frame)]
                subset_off = mean_sweep_response[self.stim_table.Frame.isin(off_frame)]
                response_on[xp,yp,:,0] = subset_on.mean(axis=0)
                response_on[xp,yp,:,1] = subset_on.std(axis=0)/np.sqrt(len(subset_on))
                response_off[xp,yp,:,0] = subset_off.mean(axis=0)
                response_off[xp,yp,:,1] = subset_off.std(axis=0)/np.sqrt(len(subset_off))
        return sweep_response, mean_sweep_response, response_on, response_off, sweep_eye, mean_sweep_eye
    
    def get_peak(self):
        '''creates a table of metrics for each cell. We can make this more useful in the future

Returns
-------
peak dataframe
        '''
        peak = pd.DataFrame(columns=('rf_on','rf_off'), index=range(self.numbercells))
        peak['rf_on'] = False
        peak['rf_off'] = False
        on_rfs = np.where(self.response_on[:,:,:,2]>0.25)[2]
        off_rfs = np.where(self.response_off[:,:,:,2]>0.25)[2]
        peak.rf_on.loc[on_rfs] = True 
        peak.rf_off.loc[off_rfs] = True 
        return peak

    def save_data(self):
        '''saves intermediate analysis files in an h5 file'''
        save_file = os.path.join(r'/Users/saskiad/Documents/Data/Openscope_Multiplex/analysis', str(self.session_id)+"_lsn_analysis.h5")
        print "Saving data to: ", save_file
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