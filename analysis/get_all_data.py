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
from stim_table import create_stim_tables, get_center_coordinates
from RunningData import get_running_data
from oscopetools.get_eye_tracking import align_eye_tracking


def get_all_data(path_name, save_path, expt_name, row):

    # get access to sub folders
    for f in os.listdir(path_name):
        if f.startswith('ophys_experiment'):
            expt_path = os.path.join(path_name, f)
        elif f.startswith('eye_tracking'):
            eye_path = os.path.join(path_name, f)
    for f in os.listdir(expt_path):
        if f.startswith('processed'):
            proc_path = os.path.join(expt_path, f)
    for f in os.listdir(proc_path):
        if f.startswith('ophys_cell_segmentation_run'):
            roi_path = os.path.join(proc_path, f)

    # ROI table
    for fname in os.listdir(expt_path):
        if fname.endswith('output_cell_roi_creation.json'):
            jsonpath = os.path.join(expt_path, fname)
            with open(jsonpath, 'r') as f:
                jin = json.load(f)
                f.close()
            break
    roi_locations = pd.DataFrame.from_dict(data=jin['rois'], orient='index')
    roi_locations.drop(
        columns=['exclude_code', 'mask_page'], inplace=True
    )  # removing columns I don't think we need
    roi_locations.reset_index(inplace=True)

    session_id = int(path_name.split('/')[-1])
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
            raw_traces = f['data'][()]
            cell_ids = f['roi_names'][()].astype(str)
            f.close()
    roi_locations['cell_id'] = cell_ids

    # eyetracking
    for fn in os.listdir(eye_path):
        if fn.endswith('mapping.h5'):
            dlc_file = os.path.join(eye_path, fn)
    for f in os.listdir(expt_path):
        if f.endswith('time_synchronization.h5'):
            temporal_alignment_file = os.path.join(expt_path, f)
    eye_sync = align_eye_tracking(dlc_file, temporal_alignment_file)
    #    pupil_area = pd.read_hdf(dlc_file, 'raw_pupil_areas')
    #    eye_area = pd.read_hdf(dlc_file, 'raw_eye_areas')
    #    pos = pd.read_hdf(dlc_file, 'raw_screen_coordinates_spherical')
    #
    #    ##temporal alignment
    #    f = h5py.File(temporal_alignment_file, 'r')
    #    eye_frames = f['eye_tracking_alignment'].value
    #    f.close()
    #    eye_frames = eye_frames.astype(int)
    #    eye_frames = eye_frames[np.where(eye_frames>0)]
    #
    #    eye_area_sync = eye_area[eye_frames]
    #    pupil_area_sync = pupil_area[eye_frames]
    #    x_pos_sync = pos.x_pos_deg.values[eye_frames]
    #    y_pos_sync = pos.y_pos_deg.values[eye_frames]
    #
    #    ##correcting dropped camera frames
    #    test = eye_frames[np.isfinite(eye_frames)]
    #    test = test.astype(int)
    #    temp2 = np.bincount(test)
    #    dropped_camera_frames = np.where(temp2>2)[0]
    #    for a in dropped_camera_frames:
    #        null_2p_frames = np.where(eye_frames==a)[0]
    #        eye_area_sync[null_2p_frames] = np.NaN
    #        pupil_area_sync[null_2p_frames] = np.NaN
    #        x_pos_sync[null_2p_frames] = np.NaN
    #        y_pos_sync[null_2p_frames] = np.NaN
    #
    #    eye_sync = pd.DataFrame(data=np.vstack((eye_area_sync, pupil_area_sync, x_pos_sync, y_pos_sync)).T, columns=('eye_area','pupil_area','x_pos_deg','y_pos_deg'))

    # max projection
    mp_path = os.path.join(proc_path, 'max_downsample_4Hz_0.png')
    mp = Image.open(mp_path)
    mp_array = np.array(mp)

    # ROI masks outlines
    boundary_path = os.path.join(roi_path, 'maxInt_boundary.png')
    boundary = Image.open(boundary_path)
    boundary_array = np.array(boundary)

    # stimulus table
    stim_table = create_stim_tables(
        path_name
    )  # returns dictionary. Not sure how to save dictionary so pulling out each dataframe

    # running speed
    dxds, startdate = get_running_data(path_name)
    # pad end with NaNs to match length of dff
    nframes = dff.shape[1] - dxds.shape[0]
    dx = np.append(dxds, np.repeat(np.NaN, nframes))

    # remove traces with NaNs from dff, roi_table, and roi_masks
    roi_locations['roi_mask_id'] = range(len(roi_locations))
    to_keep = np.where(np.isfinite(dff[:, 0]))[0]
    to_del = np.where(np.isnan(dff[:, 0]))[0]
    roi_locations['finite'] = np.isfinite(dff[:, 0])
    roi_trimmed = roi_locations[roi_locations.finite]
    roi_trimmed.reset_index(inplace=True)

    new_dff = dff[to_keep, :]

    for i in to_del:
        boundary_array[np.where(boundary_array == i)] = 0

    # meta data
    meta_data = {}
    meta_data['mouse_id'] = row.Mouse_ID
    meta_data['area'] = row.Area
    meta_data['imaging_depth'] = row.Depth
    meta_data['cre'] = row.Cre
    meta_data['container_ID'] = row.Container_ID
    meta_data['session_ID'] = session_id
    meta_data['startdate'] = startdate

    # Save Data
    save_file = os.path.join(
        save_path, expt_name + '_' + str(session_id) + '_data.h5'
    )
    print("Saving data to: ", save_file)
    store = pd.HDFStore(save_file)
    store['roi_table'] = roi_trimmed
    for key in stim_table.keys():
        store[key] = stim_table[key]
    store['eye_tracking'] = eye_sync

    store.close()
    f = h5py.File(save_file, 'r+')
    dset = f.create_dataset('dff_traces', data=new_dff)
    dset1 = f.create_dataset('raw_traces', data=raw_traces)
    dset2 = f.create_dataset('cell_ids', data=np.array(cell_ids, dtype='S'))
    dset3 = f.create_dataset('max_projection', data=mp_array)
    dset4 = f.create_dataset('roi_outlines', data=boundary_array)
    dset5 = f.create_dataset('running_speed', data=dx)
    dset6 = f.create_dataset('meta_data', data=str(meta_data))
    f.close()
    return


if __name__ == '__main__':
    manifest = pd.read_csv(
        r'/Users/saskiad/Documents/Openscope/2019/Surround suppression/Final dataset/data manifest.csv'
    )
    save_path = r'/Users/saskiad/Documents/Data/Openscope_Multiplex_trim'
    soma = manifest[manifest.Target == 'soma']
    for index, row in soma.iterrows():
        if np.mod(index, 10) == 0:
            print(index)
        expt_id = row.Center_Surround_Expt_ID
        if np.isfinite(expt_id):
            expt_name = 'Center_Surround'
            path_name = os.path.join(r'/Volumes/New Volume', str(int(expt_id)))
            get_all_data(path_name, save_path, expt_name, row)
        expt_id = row.Drifting_Gratings_Grid_Expt_ID
        if np.isfinite(expt_id):
            expt_name = 'DG_Grid'
            path_name = os.path.join(r'/Volumes/New Volume', str(int(expt_id)))
            get_all_data(path_name, save_path, expt_name, row)
        expt_id = row.Size_Tuning_Expt_ID
        if np.isfinite(expt_id):
            expt_name = 'Size_Tuning'
            path_name = os.path.join(r'/Volumes/New Volume', str(int(expt_id)))
            get_all_data(path_name, save_path, expt_name, row)
#
#    row = manifest.loc[27]
#    expt_id = row.Center_Surround_Expt_ID
#    path_name = os.path.join(r'/Volumes/New Volume', str(int(expt_id)))#975348996'
#    expt_name = 'Multiplex'
#    save_path = r'/Users/saskiad/Documents/Data/Openscope_Multiplex_trim'
#    get_all_data(path_name, save_path, expt_name, row)
