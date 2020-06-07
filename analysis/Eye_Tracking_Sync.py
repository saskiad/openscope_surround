# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 18:34:07 2017

@author: saskiad
"""

import numpy as np
import pandas as pd
import h5py
import os


def temporal_alignment_eye_tracking(path_name):
    #get eye position and pupil area
    
    
    
    # f= h5py.File(eye_path, 'r')
    # pupil = f['new_pupil_areas']['values'].value
    # f.close()
    # position = pd.read_hdf(eye_path, 'new_screen_coordinates_spherical')
    # pos_x = position['x_pos_deg'].values
    # pos_y = position['y_pos_deg'].values
    # nea = pd.read_hdf(eye_path, 'new_eye_areas').values
    # npa = pd.read_hdf(eye_path, 'new_pupil_areas').values
    data = pd.read_hdf(eye_path, 'data')
    npa = data.pupil_area.values
    nea = data.eye_area.values
    
    #get temporal alignment data
    for f in os.listdir(path_name):
        if f.startswith('ophys_experiment'):
            expt_path = os.path.join(path_name, f)
    for f in os.listdir(expt_path):
        if f.endswith('time_synchronization.h5'):
            temporal_alignment_file = os.path.join(expt_path, f)           
    f = h5py.File(temporal_alignment_file, 'r')
    eye_frames = f['eye_tracking_alignment'].value
    f.close()
    
    #align
    eye_frames = eye_frames.astype(int)
    eye_frames = eye_frames[np.where(eye_frames>0)]
    # pupil_sync = np.empty((len(eye_frames)))
    
    # pos_x_sync = pos_x[eye_frames]
    # pos_y_sync = pos_y[eye_frames]
    # position_sync = pd.DataFrame(data=np.vstack((pos_x_sync, pos_y_sync)).T, columns=('x_pos_deg','y_pos_deg'))
    # pupil_sync = pupil[eye_frames] 
    nea_sync = nea[eye_frames]
    npa_sync = npa[eye_frames]
    
    
    #correcting dropped camera frames
    test = eye_frames[np.isfinite(eye_frames)]
    test = test.astype(int)
    temp2 = np.bincount(test)
    dropped_camera_frames = np.where(temp2>2)[0]    
    for a in dropped_camera_frames:
        null_2p_frames = np.where(eye_frames==a)[0]
        # position_sync[null_2p_frames,:] = [np.NaN, np.NaN]
        # areas_sync[null_2p_frames,:] = [np.NaN, np.NaN]
        nea_sync[null_2p_frames] = np.NaN
        npa_sync[null_2p_frames] = np.NaN
        # pupil_sync[null_2p_frames] = np.NaN
    
    areas_sync = pd.DataFrame(data=np.vstack((nea_sync, npa_sync)).T, columns=('eye_area','pupil_area'))
    return areas_sync


if __name__ == "__main__":
#    temporal_alignment_file = r'/Users/saskiad/Dropbox/data/692345336_time_synchronization.h5'
#    eye_tracking_path = r'/Users/saskiad/Dropbox/data/692165696_eyetracking_dlc_to_screen_mapping.h5'
#    position_sync, pupil_sync = temporal_alignment_eye_tracking(temporal_alignment_file, eye_tracking_path)
    
    eyepath = r'\\allen\programs\braintv\workgroups\cortexmodels\gocker\boc\pupil_areas_unaligned'


#    all_data = pd.read_csv(r"C:\Users\saskiad\Dropbox\data\20200324_vis_cod_list.csv")
#    fail = []
#    for index, row in all_data.iterrows():
#        if np.mod(index, 20)==0:
#            print(index)
##        try:
#     #    head, tail = os.path.split(row.a_nwb)
#     #     # head = '\\\\'+os.path.join(*head.split('/'))
#     #    for f in os.listdir(head):
#     #        if f.endswith('time_synchronization.h5'):
#     #            temporal_alignment_file = os.path.join(head, f)
#     #    eye_tracking_path = os.path.join(eyepath, str(row.oa_id)+'.h5')
#     # #         head2, _ = os.path.split(row.a_eye_tracking)
#     # #         # head2 = '\\\\'+os.path.join(*head2.split('/'))
#     # #         for f in os.listdir(head2):
#     # #             if f.endswith('dlc_to_screen_mapping.h5'):
#     # #                 eye_tracking_path = os.path.join(head2, f)
#     # # #            eye_tracking_path = '\\\\'+os.path.join(*row.a_eye_tracking.split('/'))
#     #    _ = temporal_alignment_eye_tracking(temporal_alignment_file, eye_tracking_path, session_id=row.oa_id, save_flag=True)
#    
#        if os.path.exists(os.path.join(r'\\allen\programs\braintv\workgroups\nc-ophys\VisualCoding\eye_tracking_505', str(row.ob_id)+'_eye_data.h5'))==False:            
#            head, tail = os.path.split(row.b_nwb)
#            # head = '\\\\'+os.path.join(*head.split('/'))
#            for f in os.listdir(head):
#                if f.endswith('time_synchronization.h5'):
#                    temporal_alignment_file = os.path.join(head, f)
#            eye_tracking_path = os.path.join(eyepath, str(row.ob_id)+'.h5')
#            # head2, _ = os.path.split(row.b_eye_tracking)
#
#            # for f in os.listdir(head2):
#            #     if f.endswith('dlc_to_screen_mapping.h5'):
#            #         eye_tracking_path = os.path.join(head2, f)
#            _ = temporal_alignment_eye_tracking(temporal_alignment_file, eye_tracking_path, session_id=row.ob_id, save_flag=True)
#            
#        if os.path.exists(os.path.join(r'\\allen\programs\braintv\workgroups\nc-ophys\VisualCoding\eye_tracking_505', str(row.oc_id)+'_eye_data.h5'))==False:            
#            head, tail = os.path.split(row.c_nwb)
#            # head = '\\\\'+os.path.join(*head.split('/'))
#            for f in os.listdir(head):
#                if f.endswith('time_synchronization.h5'):
#                    temporal_alignment_file = os.path.join(head, f)
#            eye_tracking_path = os.path.join(eyepath, str(row.oc_id)+'.h5')
#    #         head2, _ = os.path.split(row.c_eye_tracking)
#    #         # head2 = '\\\\'+os.path.join(*head2.split('/'))
#    #         for f in os.listdir(head2):
#    #             if f.endswith('dlc_to_screen_mapping.h5'):
#    #                 eye_tracking_path = os.path.join(head2, f)
#    # #            eye_tracking_path = '\\\\'+os.path.join(*row.a_eye_tracking.split('/'))
#            _ = temporal_alignment_eye_tracking(temporal_alignment_file, eye_tracking_path, session_id=row.oc_id, save_flag=True)
#
#            
##      