#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 16:28:23 2020

@author: saskiad
"""
import pandas as pd
import numpy as np
import h5py


def align_eye_tracking(dlc_file, temporal_alignment_file):
    pupil_area = pd.read_hdf(dlc_file, 'raw_pupil_areas').values
    eye_area = pd.read_hdf(dlc_file, 'raw_eye_areas').values
    pos = pd.read_hdf(dlc_file, 'raw_screen_coordinates_spherical')

    ##temporal alignment
    f = h5py.File(temporal_alignment_file, 'r')
    eye_frames = f['eye_tracking_alignment'][()]
    f.close()
    eye_frames = eye_frames.astype(int)
    eye_frames = eye_frames[np.where(eye_frames > 0)]

    eye_area_sync = eye_area[eye_frames]
    pupil_area_sync = pupil_area[eye_frames]
    x_pos_sync = pos.x_pos_deg.values[eye_frames]
    y_pos_sync = pos.y_pos_deg.values[eye_frames]

    ##correcting dropped camera frames
    test = eye_frames[np.isfinite(eye_frames)]
    test = test.astype(int)
    temp2 = np.bincount(test)
    dropped_camera_frames = np.where(temp2 > 2)[0]
    for a in dropped_camera_frames:
        null_2p_frames = np.where(eye_frames == a)[0]
        eye_area_sync[null_2p_frames] = np.NaN
        pupil_area_sync[null_2p_frames] = np.NaN
        x_pos_sync[null_2p_frames] = np.NaN
        y_pos_sync[null_2p_frames] = np.NaN

    eye_sync = pd.DataFrame(
        data=np.vstack(
            (eye_area_sync, pupil_area_sync, x_pos_sync, y_pos_sync)
        ).T,
        columns=('eye_area', 'pupil_area', 'x_pos_deg', 'y_pos_deg'),
    )
    return eye_sync


if __name__ == '__main__':
    dlc_file = r'/Volumes/New Volume/1010368135/eye_tracking/1010368135_eyetracking_dlc_to_screen_mapping.h5'
    temporal_alignment_file = r'/Volumes/New Volume/1010368135/ophys_experiment_1010535819/1010535819_time_synchronization.h5'
    eye_sync = (dlc_file, temporal_alignment_file)
