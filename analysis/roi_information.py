#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 13:33:26 2019

@author: saskiad

Creates a dataframe of ROI mask information, unique id, valid calls for an imaging session
"""

import os
import json

import pandas as pd


def get_roi_information(storage_directory):
    """Parse ROI information from JSON file into a DataFrame.

    Input:
        storage_directory (str)
            -- Path to folder in which to look for ROI info JSON file.

    Returns:
        DataFrame with ROI mask information with the following columns:
            id     -- ROI number???
            x      -- x coordinate of ROI center
            y      -- y coordinate of ROI center
            width  -- width of ROI
            height -- height of ROI
            valid  -- ???
            mask   -- Boolean mask for pixels included in ROI

    """
    exp_path_head = storage_directory

    #reformat path for mac with local access
    #TODO: might need to adapt this when data is shared via DropBox
    temp = exp_path_head.split('/')
    temp[1] = 'Volumes'
    exp_path_head = '/'.join(temp)

    # Find experiment dir in storage_directory
    exp_path_files = os.listdir(exp_path_head)
    exp_folder_list = [i for i in exp_path_files if 'ophys_experiment' in i]
    if len(exp_folder_list) > 1:
        raise Exception('Multiple experiment folders in '+exp_path_head)
    else:
        exp_folder = exp_folder_list[0]

    # Find file by suffix
    processed_path = os.path.join(exp_path_head, exp_folder)
    for fname in os.listdir(processed_path):
        if fname.endswith('input_extract_traces.json'):
            jsonpath = os.path.join(processed_path, fname)
            with open(jsonpath, 'r') as f:
                jin = json.load(f)
                f.close()
            break

    # Assemble DataFrame.
    roi_locations = pd.DataFrame.from_records(
        data = jin['rois'],
        columns = ['id', 'x', 'y', 'width', 'height', 'valid', 'mask']
    )
    roi_locations['session_id'] = int(
        exp_path_head.split('/')[-2].split('_')[-1]
    )

    return roi_locations
