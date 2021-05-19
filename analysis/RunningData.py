#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 20:29:20 2020

@author: saskiad
"""

from oscopetools.stim_table import load_stim, load_alignment
import numpy as np


def get_running_data(expt_path):
    '''gets running data from stimulus log and downsamples to match imaging'''
    print("Getting running speed")
    data = load_stim(expt_path)
    dx = data['items']['foraging']['encoders'][0]['dx']
    vsync_intervals = data['intervalsms']
    while len(vsync_intervals) < len(dx):
        vsync_intervals = np.insert(vsync_intervals, 0, vsync_intervals[0])
    vsync_intervals /= 1000
    if len(dx) == 0:
        print("No running data")
    dxcm = (
        (dx / 360) * 5.5036 * np.pi * 2
    ) / vsync_intervals  # 6.5" wheel which mouse at 2/3 r
    twop_frames = load_alignment(expt_path)
    start = np.nanmin(twop_frames)
    endframe = int(np.nanmax(twop_frames) + 1)
    dxds = np.empty((endframe, 1))
    for i in range(endframe):
        try:
            temp = np.where(twop_frames == i)[0]
            dxds[i] = np.mean(dxcm[temp[0] : temp[-1] + 1])
            if np.isinf(dxds[i]):
                dxds[i] = 0
        except:
            if i < start:
                dxds[i] = np.NaN
            else:
                dxds[i] = dxds[i - 1]  # corrects for dropped frames

    startdatetime = data['startdatetime']
    return dxds, startdatetime


if __name__ == '__main__':
    exptpath = r'/Volumes/New Volume/988763069'
    dxds, startdate = get_running_data(exptpath)
