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


def get_all_data(path):

    # get access to sub folders
    for f in os.listdir(path_name):
        if f.startswith('ophys_experiment'):
            expt_path = os.path.join(path_name, f)
    for f in os.listdir(expt_path):
        if f.startswith('processed'):
            proc_path = os.path.join(expt_path, f)
    for f in os.listdir(path_name):
        if f.startswith('eye_tracking'):
            eye_path = os.path.join(path_name, f)

    # ROI table
    for fname in os.listdir(proc_path):
        if fname.endswith('input_extract_traces.json'):
            jsonpath = os.path.join(proc_path, fname)
            with open(jsonpath, 'r') as f:
                jin = json.load(f)
                f.close()
            break
    roi_locations = pd.DataFrame.from_records(
        data=jin['rois'],
        columns=['id', 'x', 'y', 'width', 'height', 'valid', 'mask'],
    )
    session_id = int(path_name.split('/')[-2].split('_')[-1])
    roi_locations['session_id'] = session_id

    # dff traces
    for f in os.listdir(expt_path):
        if f.endswith('_dff.h5'):
            dff_path = os.path.join(expt_path, f)
            f = h5py.File(dff_path, 'r')
            dff = f['data'].value
            f.close()

    # raw fluorescence & cell ids
    for f in os.listdir(proc_path):
        if f.endswith('roi_traces.h5'):
            traces_path = os.path.join(proc_path, f)
            f = h5py.File(traces_path, 'r')
            raw_traces = f['data'].value
            cell_ids = f['roi_names'].value
            f.close()

    # eyetracking
    for fn in os.listdir(eye_path):
        if fn.endswith('eyetracking_dlc_to_screen_mapping.h5'):
            eye_file = os.path.join(eye_path, fn)
    f = h5py.File(eye_file, 'r')
    pupil_area = f['new_pupil_areas']['values'].value
    f.close()
    # TODO: get eye position
    # TODO: temporal alignment

    # max projection
    mp_path = os.path.join(proc_path, 'max_downsample_4Hz_0.png')
    mp = Image.open(mp_path)
    mp_array = np.array(mp)

    # events
    # ROI masks
    # stimulus table
    # running speed

    # meta data

    # Save Data
    save_file = os.path.join(
        save_path, expt_name + '_' + srt(session_id) + '_data.h5'
    )
    print("Saving data to: ", save_file)
    store = pd.HDFStore(save_file)
    store['roi_table'] = roi_locations

    store.close()
    f = h5py.File(save_file, 'r+')
    dset = f.create_dataset('dff_traces', data=dff)
    dset1 = f.create_dataset('raw_traces', data=raw_traces)
    dset2 = f.create_dataset('cell_ids', data=cell_ids)
    dset3 = f.create_dataset('pupil_area', data=pupil_area)
    dset4 = f.create_dataset('max_projection', data=mp_array)
    f.close()

    return


if __name__ == '__main__':
    path_name = r'\\allen\programs\braintv\production\neuralcoding\prod60\specimen_772649832\ophys_session_804714847'
    expt_name = 'Contrast'
    save_path = r'C:\Users\saskiad\Documents\in_progress'
    get_all_data(pathname, save_path, exptname, all_flag=False)
