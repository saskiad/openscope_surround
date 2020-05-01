# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 20:37:19 2019

@author: danielm
"""
import numpy as np

import nd2reader

def load_ZStack(stack_path):

    read_obj = nd2reader.Nd2(stack_path)
    
    green_stack = np.zeros((len(read_obj.z_levels),512,512))
    red_stack = np.zeros((len(read_obj.z_levels),512,512))
    for i_z,z in enumerate(read_obj.z_levels):
        green_stack[i_z] = read_obj.get_image(0,0,read_obj.channels[0],z)
        red_stack[i_z] = read_obj.get_image(0,0,read_obj.channels[1],z)

    return green_stack, red_stack