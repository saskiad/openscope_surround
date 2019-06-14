#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 13:33:26 2019

@author: saskiad

Creates a dataframe of ROI mask information, unique id, valid calls for an imaging session
"""

import pandas as pd
import json
import os

def get_roi_information(storage_directory):    
    exp_path_head = storage_directory
    
    #reformat path for mac with local access 
    #TODO: might need to adapt this when data is shared via DropBox
    temp = exp_path_head.split('/')
    temp[1] = 'Volumes'
    exp_path_head = '/'.join(temp)

    exp_path_files = os.listdir(exp_path_head)
    exp_folder_list = [i for i in exp_path_files if 'ophys_experiment' in i]
    if len(exp_folder_list) > 1:
        raise Exception('Multiple experiment folders in '+exp_path_head)
    else:
        exp_folder = exp_folder_list[0]
    for f in os.listdir(os.path.join(exp_path_head, exp_folder, 'processed')):
        if f.endswith('input_extract_traces.json'):
            jsonpath = os.path.join(exp_path_head, exp_folder, 'processed',f)
            with open(jsonpath, 'r') as w:
                jin = json.load(w)
            rois = jin["rois"]
            roi_locations_list = []
            for i in range(len(rois)):
                roi = rois[i]
                if roi['mask'][0] == '{':
                    mask = parse_mask_string(roi['mask'])
                else:
                    mask = roi["mask"]
                roi_locations_list.append([roi["id"], roi["x"], roi["y"], roi["width"], roi["height"], roi["valid"], mask])
            roi_locations = pd.DataFrame(data=roi_locations_list, columns=['id', 'x', 'y', 'width', 'height', 'valid', 'mask'])
            roi_locations['session_id'] = int(exp_path_head.split('/')[-2].split('_')[-1])
    return roi_locations


