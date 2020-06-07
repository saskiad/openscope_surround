#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 21:59:56 2020

@author: saskiad
"""

import h5py
import pandas as pd


def get_dff_traces(file_path):
    f = h5py.File(file_path)
    dff = f['dff_traces'][()]
    f.close()
    return dff

def get_raw_traces(file_path):
    f = h5py.File(file_path)
    raw = f['raw_traces'][()]
    f.close()
    return raw

def get_running_speed(file_path):
    f = h5py.File(file_path)
    dx = f['running_speed'][()]
    f.close()
    return dx

def get_cell_ids(file_path):
    f = h5py.File(file_path)
    cell_ids = f['cell_ids'][()]
    f.close()
    return cell_ids

def get_max_projection(file_path):
    f = h5py.File(file_path)
    max_proj = f['max_projection'][()]
    f.close()
    return max_proj

def get_metadata(file_path):
    import ast
    f = h5py.File(file_path)
    md = f.get('meta_data')[...].tolist()
    f.close()
    meta_data = ast.literal_eval(md)
    return meta_data

def get_roi_table(file_path):
    rois = pd.read_hdf(file_path, 'roi_table')
    return rois

def get_stimulus_table(file_path, stimulus):
    stim_table = pd.read_hdf(file_path, stimulus)
    return stim_table

def get_pupil_area(file_path):
    
    return

def get_pupil_position(file_path):
    return