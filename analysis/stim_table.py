# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 17:33:28 2019

@author: danielm
"""
import os, sys
import numpy as np
import pandas as pd

from sync import Dataset

def coarse_mapping_create_stim_table(exptpath):
    
    data = load_stim(exptpath)
    twop_frames, twop_vsync_fall, stim_vsync_fall, photodiode_rise = load_sync(exptpath)
    
    stim_table = {}
    stim_table['locally_sparse_noise'] = locally_sparse_noise_table(data,twop_frames)
    stim_table['drifting_gratings_grid'] = DGgrid_table(data,twop_frames)

    return stim_table
    
def lsnCS_create_stim_table(exptpath):
    
    data = load_stim(exptpath)
    twop_frames, twop_vsync_fall, stim_vsync_fall, photodiode_rise = load_sync(exptpath)
    
    stim_table = {}
    stim_table['center_surround'] = center_surround_table(data,twop_frames)
    stim_table['locally_sparse_noise'] = locally_sparse_noise_table(data,twop_frames)
    
    return stim_table
    
def DGgrid_table(data,twop_frames):
    
    DG_idx = get_stimulus_index(data,'grating')
    
    timing_table, actual_sweeps, expected_sweeps = get_sweep_frames(data,DG_idx)
    print 'Found ' + str(actual_sweeps) + ' of ' + str(expected_sweeps) + ' expected sweeps.'
    
    stim_table = pd.DataFrame(np.column_stack((twop_frames[timing_table['start']],twop_frames[timing_table['end']])), columns=('Start', 'End'))
    
    stim_table['TF'] = get_attribute_by_sweep(data,DG_idx,'TF')[:len(stim_table)]
    stim_table['SF'] = get_attribute_by_sweep(data,DG_idx,'SF')[:len(stim_table)]
    stim_table['Contrast'] = get_attribute_by_sweep(data,DG_idx,'Contrast')[:len(stim_table)]
    stim_table['Ori'] = get_attribute_by_sweep(data,DG_idx,'Ori')[:len(stim_table)]
    stim_table['PosX'] = get_attribute_by_sweep(data,DG_idx,'PosX')[:len(stim_table)]
    stim_table['PosY'] = get_attribute_by_sweep(data,DG_idx,'PosY')[:len(stim_table)]
    
    return stim_table

def locally_sparse_noise_table(data,twop_frames):
    
    lsn_idx = get_stimulus_index(data,'locally_sparse_noise')
    
    timing_table, actual_sweeps, expected_sweeps = get_sweep_frames(data,lsn_idx)
    print 'Found ' + str(actual_sweeps) + ' of ' + str(expected_sweeps) + ' expected sweeps.'
    
    stim_table = pd.DataFrame(np.column_stack((twop_frames[timing_table['start']],twop_frames[timing_table['end']])), columns=('Start', 'End'))

    stim_table['Frame'] = np.array(data['stimuli'][lsn_idx]['sweep_order'][:len(stim_table)])

    return stim_table

def center_surround_table(data,twop_frames):
    
    center_idx = get_stimulus_index(data,'center')
    surround_idx = get_stimulus_index(data,'surround')
    
    timing_table, actual_sweeps, expected_sweeps = get_sweep_frames(data,center_idx)
    print 'Found ' + str(actual_sweeps) + ' of ' + str(expected_sweeps) + ' expected sweeps.'
    
    stim_table = pd.DataFrame(np.column_stack((twop_frames[timing_table['start']],twop_frames[timing_table['end']])), columns=('Start', 'End'))

    stim_table['TF'] = get_attribute_by_sweep(data,center_idx,'TF')[:len(stim_table)]
    stim_table['SF'] = get_attribute_by_sweep(data,center_idx,'SF')[:len(stim_table)]
    stim_table['Contrast'] = get_attribute_by_sweep(data,center_idx,'Contrast')[:len(stim_table)]
    stim_table['Center_Ori'] = get_attribute_by_sweep(data,center_idx,'Ori')[:len(stim_table)]
    stim_table['Surround_Ori'] = get_attribute_by_sweep(data,surround_idx,'Ori')[:len(stim_table)]

    return stim_table
    
def get_stimulus_index(data,stim_name):
    
    for i_stim,stim_data in enumerate(data['stimuli']):
        if stim_data['stim_path'].find(stim_name)!=-1:
            return i_stim
            
    print 'Stimulus ' + stim_name + ' not found!'
    sys.exit()
    
def get_display_sequence(data,stimulus_idx):
    
    display_sequence = np.array(data['stimuli'][stimulus_idx]['display_sequence'])
    pre_blank_sec = int(data['pre_blank_sec'])
    display_sequence += pre_blank_sec
    display_sequence *= int(data['fps']) #in stimulus frames
    
    return display_sequence
    
def get_sweep_frames(data,stimulus_idx):
    
    sweep_frames = data['stimuli'][stimulus_idx]['sweep_frames']
    timing_table = pd.DataFrame(np.array(sweep_frames).astype(np.int),columns=('start','end'))            
    timing_table['dif'] = timing_table['end']-timing_table['start']
    
    display_sequence = get_display_sequence(data,stimulus_idx) 

    timing_table.start += display_sequence[0,0]
    for seg in range(len(display_sequence)-1):
        for index, row in timing_table.iterrows():
            if row.start >= display_sequence[seg,1]:
                timing_table.start[index] = timing_table.start[index] - display_sequence[seg,1] + display_sequence[seg+1,0]
    timing_table.end = timing_table.start+timing_table.dif
    expected_sweeps = len(timing_table)
    timing_table = timing_table[timing_table.end <= display_sequence[-1,1]]
    timing_table = timing_table[timing_table.start <= display_sequence[-1,1]]            
    actual_sweeps = len(timing_table)
    
    return timing_table, actual_sweeps, expected_sweeps

def get_attribute_by_sweep(data,stimulus_idx,attribute):
    
    attribute_idx = get_attribute_idx(data,stimulus_idx,attribute)
    
    sweep_order = data['stimuli'][stimulus_idx]['sweep_order']
    sweep_table = data['stimuli'][stimulus_idx]['sweep_table']
    
    num_sweeps = len(sweep_order)
    
    attribute_by_sweep = np.zeros((num_sweeps,))
    attribute_by_sweep[:] = np.NaN
    
    unique_conditions = np.unique(sweep_order)
    for i_condition,condition in enumerate(unique_conditions):
        sweeps_with_condition = np.argwhere(sweep_order==condition)[:,0]
        
        if condition > 0 : #blank sweep is -1
            attribute_by_sweep[sweeps_with_condition] = sweep_table[condition][attribute_idx]

    return attribute_by_sweep
    
def get_attribute_idx(data,stimulus_idx,attribute):
    
    attribute_names = data['stimuli'][stimulus_idx]['dimnames']
    for attribute_idx,attribute_str in enumerate(attribute_names):
        if attribute_str==attribute:
            return attribute_idx

    print 'Attribute ' + attribute + ' for stimulus_idx ' + str(stimulus_idx) + ' not found!'
    sys.exit()
    
def load_stim(exptpath):
    
    #verify that pkl file exists in exptpath
    pklMissing = True
    for f in os.listdir(exptpath):
        if f.endswith('_stim.pkl'):
            pklpath = os.path.join(exptpath, f)
            pklMissing = False
            print "Pkl file:", f
    if pklMissing:
        print "No pkl file"
        sys.exit()
    
    return pd.read_pickle(pklpath) 
        
def load_sync(exptpath):
    
    #verify that sync file exists in exptpath
    syncMissing = True
    for f in os.listdir(exptpath):
        if f.endswith('_sync.h5'):
            syncpath = os.path.join(exptpath, f)
            syncMissing = False
            print "Sync file:", f
    if syncMissing:
        print "No sync file"
        sys.exit()

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
    channels = {'twop_vsync_fall': twop_vsync_fall, 'stim_vsync_fall':stim_vsync_fall, 'photodiode_rise': photodiode_rise}
    channel_test = []    
    for i in channels:
        channel_test.append(any(channels[i]))
    if all(channel_test):
        print "All channels present."
    else:
        print "Not all channels present. Sync test failed."
        sys.exit()        
        
    #test and correct for photodiode transition errors
    ptd_rise_diff = np.ediff1d(photodiode_rise)
    short = np.where(np.logical_and(ptd_rise_diff>0.1, ptd_rise_diff<0.3))[0]
    medium = np.where(np.logical_and(ptd_rise_diff>0.5, ptd_rise_diff<1.5))[0]
    ptd_start = 3
    for i in medium:
        if set(range(i-2,i)) <= set(short):
            ptd_start = i+1
    ptd_end = np.where(photodiode_rise>stim_vsync_fall.max())[0][0] - 1
    
    if ptd_start > 3:
        print 'ptd_start: ' + str(ptd_start)
        print "Photodiode events before stimulus start.  Deleted."
        
    ptd_errors = []
    while any(ptd_rise_diff[ptd_start:ptd_end] < 1.8):
        error_frames = np.where(ptd_rise_diff[ptd_start:ptd_end]<1.8)[0] + ptd_start
        print "Photodiode error detected. Number of frames:", len(error_frames)
        photodiode_rise = np.delete(photodiode_rise, error_frames[-1])
        ptd_errors.append(photodiode_rise[error_frames[-1]])
        ptd_end-=1
        ptd_rise_diff = np.ediff1d(photodiode_rise)
    
    first_pulse = ptd_start
    stim_on_photodiode_idx = 60+120*np.arange(0,ptd_end+1-ptd_start-1,1)
    
    stim_on_photodiode = stim_vsync_fall[stim_on_photodiode_idx]
    photodiode_on = photodiode_rise[first_pulse + np.arange(0,ptd_end+1-ptd_start-1,1)]
    delay_rise = photodiode_on - stim_on_photodiode
    
    delay = np.mean(delay_rise[:-1])   
    print "monitor delay: " , delay
    
    #adjust stimulus time to incorporate monitor delay
    stim_time = stim_vsync_fall + delay
    
    #convert stimulus frames into twop frames
    twop_frames = np.empty((len(stim_time),1))
    acquisition_ends_early = 0
    for i in range(len(stim_time)):
        # crossings = np.nonzero(np.ediff1d(np.sign(twop_vsync_fall - stim_time[i]))>0)
        crossings = np.searchsorted(twop_vsync_fall,stim_time[i],side='left') -1
        if crossings < (len(twop_vsync_fall)-1):
            twop_frames[i] = crossings
        else:
            twop_frames[i:len(stim_time)]=np.NaN
            acquisition_ends_early = 1
            break
            
    if acquisition_ends_early>0:
        print "Acquisition ends before stimulus"   
        
    return twop_frames, twop_vsync_fall, stim_vsync_fall, photodiode_rise
        
if __name__=='__main__':  
    exptpath = r'\\allen\programs\braintv\production\neuralcoding\prod55\specimen_859061987\ophys_session_882666374\\'
    lsnCS_create_stim_table(exptpath)