# -*- coding: utf-8 -*-
"""
Created on Mon Mar 09 20:46:47 2020

@author: saskiad
"""

import os
import numpy as np
import pandas as pd
import json
import h5py
from PIL import Image
from stim_table import create_stim_tables
from RunningData import get_running_data

def get_all_data(path_name, save_path, expt_name, row):
    
    #get access to sub folders
    for f in os.listdir(path_name):
        if f.startswith('ophys_experiment'):
            expt_path = os.path.join(path_name, f)
    for f in os.listdir(expt_path):
        if f.startswith('processed'):
            proc_path = os.path.join(expt_path, f)
    for f in os.listdir(path_name):
        if f.startswith('eye_tracking'):
            eye_path = os.path.join(path_name, f)
    
    #ROI table
    for fname in os.listdir(expt_path):
        if fname.endswith('output_cell_roi_creation.json'):
            jsonpath= os.path.join(expt_path, fname)
            with open(jsonpath, 'r') as f:
                jin = json.load(f)
                f.close()
            break
    roi_locations = pd.DataFrame.from_dict(data = jin['rois'], orient='index')
    roi_locations.drop(columns=['exclude_code','exclusion_labels','mask_page'], inplace=True) #remove columns I don't think we need
    roi_locations.reset_index(inplace=True) 
    
    session_id = int(
        path_name.split('/')[-1]
    )
    roi_locations['session_id'] = session_id
    
    #dff traces
    for f in os.listdir(expt_path):
        if f.endswith('_dff.h5'):
            dff_path = os.path.join(expt_path, f)
            f = h5py.File(dff_path, 'r')
            dff = f['data'].value
            f.close()

    #raw fluorescence & cell ids
    for f in os.listdir(proc_path):
         if f.endswith('roi_traces.h5'):
             traces_path = os.path.join(proc_path, f)
             f = h5py.File(traces_path, 'r')
             raw_traces = f['data'][()]
             cell_ids = f['roi_names'][()].astype(str)
             f.close()
    roi_locations['cell_id'] = cell_ids #TODO: is the order of cells the same in the dataframe as in the traces array?
    
    #eyetracking
    for fn in os.listdir(eye_path):
        if fn.endswith('eyetracking_dlc_to_screen_mapping.h5'):
            eye_file = os.path.join(eye_path, fn)
    f = h5py.File(eye_file, 'r')
    pupil_area = f['new_pupil_areas']['values'][()]
    f.close()
    #TODO: get eye position
    #TODO: temporal alignment
    
    #max projection
    mp_path = os.path.join(proc_path, 'max_downsample_4Hz_0.png')
    mp = Image.open(mp_path)
    mp_array = np.array(mp)

    #ROI masks
    
    #stimulus table
    stim_table = create_stim_tables(path_name) #returns dictionary. Not sure how to save dictionary so pulling out each dataframe

    #running speed
    dxds, startdate = get_running_data(path_name)
    #pad end with NaNs to match length of dff
    nframes = dff.shape[1] - dxds.shape[0]
    dx = np.append(dxds, np.repeat(np.NaN, nframes))
    
    #meta data
    meta_data = {}
    meta_data['mouse_id'] = row.Mouse_ID
    meta_data['area'] = row.Area
    meta_data['imaging_depth'] = row.Depth
    meta_data['cre'] = row.Cre
    meta_data['container_ID'] = row.Container_ID
    meta_data['session_ID'] = session_id
    meta_data['startdate'] = startdate
    
    #Save Data
    save_file = os.path.join(save_path, expt_name+'_'+str(session_id)+'_data.h5')
    print "Saving data to: ", save_file
    store = pd.HDFStore(save_file)
    store['roi_table'] = roi_locations
    for key in stim_table.keys():
        store[key] = stim_table[key]
            
    store.close()
    f = h5py.File(save_file, 'r+')
    dset = f.create_dataset('dff_traces', data=dff)
    dset1 = f.create_dataset('raw_traces', data=raw_traces)
    dset2 = f.create_dataset('cell_ids', data=cell_ids)
    dset3 = f.create_dataset('pupil_area', data=pupil_area)
    dset4 = f.create_dataset('max_projection', data=mp_array)
    dset5 = f.create_dataset('running_speed', data=dx)
    dset6 = f.create_dataset('meta_data', data=str(meta_data))
    f.close()
    
    
    return


if __name__=='__main__':
    manifest = pd.read_csv(r'/Users/saskiad/Documents/Openscope/2019/Surround suppression/Final dataset/data manifest.csv')
    row = manifest.loc[27]
    expt_id = row.Drifting_Gratings_Grid_Expt_ID
    path_name = os.path.join(r'/Volumes/New Volume', str(int(expt_id)))#975348996'
    expt_name = 'Multiplex'
    save_path = r'/Users/saskiad/Documents/Openscope/2019/Surround suppression/Final dataset'
    get_all_data(path_name, save_path, expt_name, row)
    
    
    

    
    
