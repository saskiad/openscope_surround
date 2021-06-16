# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 14:06:37 2016

@author: danielm
"""
import os, sys

import numpy as np
import pandas as pd
import h5py

import cPickle as pickle
from sync import Dataset
import tifffile as tiff
import matplotlib.pyplot as plt

import nd2reader

def run_analysis():
    
    exp_date = '20190605'
    mouse_ID = '462046'
    im_filetype = 'nd2'#'h5'


    #DON'T MODIFY CODE BELOW THIS POINT!!!!!!!!

    exp_superpath = r'C:\\CAM\\data\\'
    im_superpath = r'E:\\'
    exptpath = find_exptpath(exp_superpath,exp_date,mouse_ID)
    im_directory = find_impath(im_superpath,exp_date,mouse_ID)
    savepath = r'\\allen\\programs\\braintv\\workgroups\\ophysdev\\OPhysCore\\OpenScope\\Multiplex\\coordinates\\'
    
    stim_table = create_stim_table(exptpath)
    
    fluorescence = get_wholefield_fluorescence(stim_table,im_filetype,im_directory,exp_date,mouse_ID,savepath)
    
    mean_sweep_response, sweep_response = get_mean_sweep_response(fluorescence,stim_table)
    
    best_location = plot_sweep_response(sweep_response,stim_table,exp_date,mouse_ID,savepath)

    write_text_file(best_location,exp_date+'_'+mouse_ID,savepath)
    
def find_exptpath(exp_superpath,exp_date,mouse_ID):

    exptpath = None
    for f in os.listdir(exp_superpath):
        if f.lower().find(mouse_ID+'_'+exp_date)!=-1:
            exptpath = exp_superpath+f+'\\'
    return exptpath

def find_impath(im_superpath,exp_date,mouse_ID):

    im_path = None
    for f in os.listdir(im_superpath):
        if f.lower().find(exp_date+'_'+mouse_ID)!=-1:
            im_path = im_superpath+f+'\\'
    return im_path

def write_text_file(best_location,save_name,savepath):
    
    f = open(savepath+save_name+'_coordinates.txt','w')
    f.write(str(best_location[0]))
    f.write(',')
    f.write(str(best_location[1]))
    f.close()
    
def plot_sweep_response(sweep_response,stim_table,exp_date,mouse_ID,exptpath):
    
    x_pos = np.unique(stim_table['PosX'].values)
    x_pos = x_pos[np.argwhere(np.isfinite(x_pos))]
    y_pos = np.unique(stim_table['PosY'].values)
    y_pos = y_pos[np.argwhere(np.isfinite(y_pos))]
    ori = np.unique(stim_table['Ori'].values)
    ori = ori[np.argwhere(np.isfinite(ori))]
    
    num_x = len(x_pos)
    num_y = len(y_pos)
    num_sweeps = len(sweep_response)
    
    plt.figure(figsize=(20,20))
    ax = []
    for x in range(num_x):
        for y in range(num_y):
            ax.append(plt.subplot2grid((num_x,num_y), (x,y), colspan=1) )    

    ori_colors=['k','b','m','r','y','g']  
    
    #convert fluorescence to dff
    baseline_frames = 28
    weighted_average = np.zeros((2,))
    summed_response = 0
    for i in range(num_sweeps):
        baseline = np.mean(sweep_response[i,:baseline_frames])
        sweep_response[i,:] = sweep_response[i,:] - baseline
    
    y_max = np.max(sweep_response.flatten())
    y_min = np.min(sweep_response.flatten())     
    
    for x in range(len(x_pos)):
        is_x = stim_table['PosX'] == x_pos[x][0]
        for y in range(len(y_pos)):
            is_y = stim_table['PosY'] == y_pos[y][0]
            this_ax = ax[num_x*(num_y-1-y)+x]
            position_average = np.zeros((np.shape(sweep_response)[1],))
            num_at_position = 0
            for o in range(len(ori)):
                is_ori = stim_table['Ori'] == ori[o][0]
                is_repeat = (is_x & is_y & is_ori).values
                repetition_idx = np.argwhere(is_repeat)
                if any(repetition_idx==0):
                    repetition_idx = repetition_idx[1:]
                for rep in range(len(repetition_idx)):
                    this_response = sweep_response[repetition_idx[rep]]
                    this_response = this_response[0,:]
                    this_ax.plot(this_response,ori_colors[o])
                    this_ax.set_ylim([y_min, y_max])
                    num_at_position += 1
                    position_average = np.add(position_average,this_response)
            position_average = np.divide(position_average,num_at_position)
            position_response = np.mean(position_average[(baseline_frames+5):(baseline_frames+27)])
            summed_response += np.max([0.0,position_response])
            weighted_average[0] += x_pos[x][0] * np.max([0.0,position_response])
            weighted_average[1] += y_pos[y][0] * np.max([0.0,position_response])
            this_ax.plot(position_average,linewidth=3.0,color='k')
            this_ax.plot([baseline_frames, baseline_frames],[y_min,y_max],'k--')
            this_ax.set_title('X: ' + str(x_pos[x][0]) + ', Y: ' + str(y_pos[y][0]))
    plt.savefig(exptpath+exp_date+'_'+mouse_ID+'_DGgrid_traces.png',dpi=300)
    plt.close() 
    
    weighted_average = weighted_average / summed_response
    
    best_location = (round(weighted_average[0],1),round(weighted_average[1],1))
    
    return best_location


def plot_grid_response(mean_sweep_response,stim_table,exptpath):

    x_pos = np.unique(stim_table['PosX'].values)
    x_pos = x_pos[np.argwhere(np.isfinite(x_pos))]
    y_pos = np.unique(stim_table['PosY'].values)
    y_pos = y_pos[np.argwhere(np.isfinite(y_pos))]
    ori = np.unique(stim_table['Ori'].values)
    ori = ori[np.argwhere(np.isfinite(ori))]
    
    response_grid = np.zeros((len(y_pos),len(x_pos)))
    for o in range(len(ori)):
        is_ori = stim_table['Ori'] == ori[o][0]
        ori_responses = np.zeros((len(y_pos),len(x_pos)))
        for x in range(len(x_pos)):
            is_x = stim_table['PosX'] == x_pos[x][0]
            for y in range(len(y_pos)):
                is_y = stim_table['PosY'] == y_pos[y][0]    
                is_repeat = (is_x & is_y & is_ori).values
                repetition_idx = np.argwhere(is_repeat)
                if any(repetition_idx==0):
                    repetition_idx = repetition_idx[1:]
                repetition_responses = np.zeros((len(repetition_idx),))
                for rep in range(len(repetition_idx)):
                    repetition_responses[rep] = mean_sweep_response[repetition_idx[rep]]
                ori_responses[y,x] = np.mean(repetition_responses)
        ori_responses = np.subtract(ori_responses,np.mean(ori_responses.flatten()))
        response_grid = np.add(response_grid,ori_responses) 
    
    plt.figure()
    plt.imshow(response_grid,vmax=np.max(response_grid),vmin=-np.max(response_grid),cmap=u'bwr',interpolation='none',origin='lower')
    plt.colorbar()
    plt.xlabel('X Pos')
    plt.ylabel('Y Pos')    
    
    x_tick_labels = range(len(x_pos))
    for i in range(len(x_pos)):
        x_tick_labels[i] = str(x_pos[i][0])
    y_tick_labels = range(len(y_pos))
    for i in range(len(y_pos)):
        y_tick_labels[i] = str(y_pos[i][0])
    plt.xticks(np.arange(len(x_pos)),x_tick_labels)
    plt.yticks(np.arange(len(y_pos)),y_tick_labels)
    
    plt.savefig(exptpath+'/DGgrid_response')
    
def get_mean_sweep_response(fluorescence,stim_table):

    sweeplength = int(stim_table.End[1] - stim_table.Start[1])
    interlength = 28
    extralength = 7
    
    num_stim_presentations = len(stim_table['Start'])
    mean_sweep_response = np.zeros((num_stim_presentations,))    
    sweep_response = np.zeros((num_stim_presentations,sweeplength+interlength))
    for i in range(num_stim_presentations):
        start = stim_table['Start'][i]-interlength
        end = stim_table['Start'][i] + sweeplength
        sweep_f = fluorescence[int(start):int(end)]
        sweep_dff = 100*((sweep_f/np.mean(sweep_f[:interlength]))-1)
        sweep_response[i,:] = sweep_f
        mean_sweep_response[i] = np.mean(sweep_dff[interlength:(interlength+sweeplength)])
   
    return mean_sweep_response, sweep_response

def load_single_tif(file_path):  
    return tiff.imread(file_path)
    
def get_wholefield_fluorescence(stim_table,im_filetype,im_directory,exp_date,mouse_ID,savepath):  
    
    if os.path.isfile(savepath+exp_date+'_'+mouse_ID+'_wholefield.npy'):
        avg_fluorescence = np.load(savepath+exp_date+'_'+mouse_ID+'_wholefield.npy')
    else:
    
        im_path = None
        if im_filetype=='nd2':
            for f in os.listdir(im_directory):
                if f.endswith(im_filetype) and f.lower().find('local') == -1:
                    im_path = im_directory + f
                    print im_path
        elif im_filetype=='h5':
            #find experiment directory:
            for f in os.listdir(im_directory):
                if f.lower().find('ophys_experiment_')!=-1:
                    exp_path = im_directory+f+'\\'
                    session_ID = f[17:]
                    print session_ID
        else:
            print 'im_filetype not recognized!'
            sys.exit(1)
                
        if im_filetype=='nd2':
            print 'Reading nd2...'
            read_obj = nd2reader.Nd2(im_path)
            num_frames = len(read_obj.frames)
            avg_fluorescence = np.zeros((num_frames,))
            
            sweep_starts = stim_table['Start'].values
            block_bounds = []
            block_bounds.append((np.min(sweep_starts)-30,np.max(sweep_starts[sweep_starts<50000])+100))
            block_bounds.append((np.min(sweep_starts[sweep_starts>50000])-30,np.max(sweep_starts)+100))
            
            for block in block_bounds:
                frame_start = int(block[0])
                frame_end = int(block[1])
                for f in np.arange(frame_start,frame_end):
                    this_frame = read_obj.get_image(f,0,read_obj.channels[0],0)
                    print 'Loaded frame ' + str(f) + ' of ' + str(num_frames)
                    avg_fluorescence[f] = np.mean(this_frame)
        elif im_filetype=='h5':
            f = h5py.File(exp_path+session_ID+'.h5')
            data = np.array(f['data'])
            avg_fluorescence = np.mean(data,axis=(1,2))
            f.close()
        np.save(savepath+exp_date+'_'+mouse_ID+'_wholefield.npy',avg_fluorescence)
                    
    return avg_fluorescence
    
def create_stim_table(exptpath):
    
    #load stimulus and sync data
    data = load_pkl(exptpath)
    twop_frames, twop_vsync_fall, stim_vsync_fall, photodiode_rise = load_sync(exptpath)
    
    display_sequence = data['stimuli'][0]['display_sequence']
    display_sequence += data['pre_blank_sec']
    display_sequence *= int(data['fps']) #in stimulus frames
    
    sweep_frames = data['stimuli'][0]['sweep_frames']
    stimulus_table = pd.DataFrame(sweep_frames,columns=('start','end'))            
    stimulus_table['dif'] = stimulus_table['end']-stimulus_table['start']
    stimulus_table.start += display_sequence[0,0]
    for seg in range(len(display_sequence)-1):
        for index, row in stimulus_table.iterrows():
            if row.start >= display_sequence[seg,1]:
                stimulus_table.start[index] = stimulus_table.start[index] - display_sequence[seg,1] + display_sequence[seg+1,0]
    stimulus_table.end = stimulus_table.start+stimulus_table.dif
    print len(stimulus_table)
    stimulus_table = stimulus_table[stimulus_table.end <= display_sequence[-1,1]]
    stimulus_table = stimulus_table[stimulus_table.start <= display_sequence[-1,1]]            
    print len(stimulus_table)
    sync_table = pd.DataFrame(np.column_stack((twop_frames[stimulus_table['start']],twop_frames[stimulus_table['end']])), columns=('Start', 'End'))
           
    #populate stimulus parameters
    print data['stimuli'][0]['stim_path']
    
    #get center parameters
    sweep_order = data['stimuli'][0]['sweep_order']
    sweep_order =  sweep_order[:len(stimulus_table)]    
    sweep_table = data['stimuli'][0]['sweep_table']
    dimnames = data['stimuli'][0]['dimnames']  
    sweep_table = pd.DataFrame(sweep_table, columns=dimnames)            
    
    #populate sync_table 
    sync_table['SF'] = np.NaN
    sync_table['TF'] = np.NaN
    sync_table['Contrast'] = np.NaN
    sync_table['Ori'] = np.NaN
    sync_table['PosX'] = np.NaN
    sync_table['PosY'] = np.NaN
    for index in np.arange(len(stimulus_table)):
        if (not np.isnan(stimulus_table['end'][index])) & (sweep_order[index] >= 0):
            sync_table['SF'][index] = sweep_table['SF'][int(sweep_order[index])]
            sync_table['TF'][index] = sweep_table['TF'][int(sweep_order[index])]
            sync_table['Contrast'][index] = sweep_table['Contrast'][int(sweep_order[index])]
            sync_table['Ori'][index] = sweep_table['Ori'][int(sweep_order[index])]
            sync_table['PosX'][index] = sweep_table['PosX'][int(sweep_order[index])]
            sync_table['PosY'][index] = sweep_table['PosY'][int(sweep_order[index])]
            
    return sync_table
    
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
    print d.line_labels
    #set the appropriate sample frequency
    sample_freq = d.meta_data['ni_daq']['counter_output_freq']
    
    #get sync timing for each channel
    twop_vsync_fall = d.get_falling_edges('2p_vsync')/sample_freq
    #stim_vsync_fall = d.get_falling_edges('vsync_stim')[1:]/sample_freq #eliminating the DAQ pulse   
    stim_vsync_fall = d.get_falling_edges('stim_vsync')[1:]/sample_freq #eliminating the DAQ pulse   
    photodiode_rise = d.get_rising_edges('stim_photodiode')/sample_freq
    
    print 'num stim vsyncs: ' + str(len(stim_vsync_fall))
    print 'num 2p frames: ' + str(len(twop_vsync_fall))
    print 'num photodiode flashes: ' + str(len(photodiode_rise))

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


    #find three consecutive pulses at the start of session:
    two_back_lag = photodiode_rise[2:20] - photodiode_rise[:18]
    ptd_start = np.argmin(two_back_lag) + 3
    print 'ptd_start: ' + str(ptd_start)

    #ptd_start = 3
    #for i in medium:
    #    if set(range(i-2,i)) <= set(short):
    #        ptd_start = i+1
    ptd_end = np.where(photodiode_rise>stim_vsync_fall.max())[0][0] - 1

    # plt.figure()
    # plt.hist(ptd_rise_diff)
    # plt.show()
    
    # plt.figure()
    # plt.plot(stim_vsync_fall[:300])
    # plt.title('stim vsync start')
    # plt.show()
    
    # plt.figure()
    # plt.plot(photodiode_rise[:10])
    # plt.title('photodiode start')
    # plt.show()

    # plt.figure()
    # plt.plot(stim_vsync_fall[-300:])
    # plt.title('stim vsync end')
    # plt.show()
    
    # plt.figure()
    # plt.plot(photodiode_rise[-10:])
    # plt.title('photodiode end')
    # plt.show()
    
    print 'ptd_start: ' + str(ptd_start)
    if ptd_start > 3:
        print "Photodiode events before stimulus start.  Deleted."
        
#    ptd_errors = []
#    while any(ptd_rise_diff[ptd_start:ptd_end] < 1.8):
#        error_frames = np.where(ptd_rise_diff[ptd_start:ptd_end]<1.8)[0] + ptd_start
#        #print "Photodiode error detected. Number of frames:", len(error_frames)
#        photodiode_rise = np.delete(photodiode_rise, error_frames[-1])
#        ptd_errors.append(photodiode_rise[error_frames[-1]])
#        ptd_end-=1
#        ptd_rise_diff = np.ediff1d(photodiode_rise)
    
    first_pulse = ptd_start
    stim_on_photodiode_idx = 60+120*np.arange(0,ptd_end+1-ptd_start-1,1)
    
    #stim_vsync_fall = stim_vsync_fall[0] + np.arange(stim_on_photodiode_idx.max()+481) * 0.0166666
    
#    stim_on_photodiode = stim_vsync_fall[stim_on_photodiode_idx]
#    photodiode_on = photodiode_rise[first_pulse + np.arange(0,ptd_end+1-ptd_start-1,1)]
#                                    
#    plt.figure()
#    plt.plot(stim_on_photodiode[:4])
#    plt.title('stim start')
#    plt.show()
#        
#    plt.figure()
#    plt.plot(photodiode_on[:4])
#    plt.title('photodiode start')
#    plt.show()
#                      
#    delay_rise = photodiode_on - stim_on_photodiode
#    init_delay_period = delay_rise < 0.025
#    init_delay = np.mean(delay_rise[init_delay_period])
#    
#    plt.figure()
#    plt.plot(delay_rise[:10])
#    plt.title('delay rise')
#    plt.show()
    
    delay = 0.0#init_delay 
    print "monitor delay: " , delay
    
    #adjust stimulus time with monitor delay
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

def load_pkl(exptpath):
    
    #verify that pkl file exists in exptpath
    logMissing = True
    for f in os.listdir(exptpath):
        if f.endswith('.pkl'):
            logpath = os.path.join(exptpath, f)
            logMissing = False
            print "Stimulus log:", f
    if logMissing:
        print "No pkl file"
        sys.exit()
        
    #load data from pkl file
    f = open(logpath, 'rb')
    data = pickle.load(f)
    f.close()
    
    return data
    
if __name__=='__main__':  
    run_analysis()