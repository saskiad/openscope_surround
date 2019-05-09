"""
HawkenGratings.py
"""
from psychopy import visual
from camstim import Stimulus, SweepStim
from camstim import Foraging
from camstim import Window, Warp
import numpy as np

# Create display window
window = Window(fullscr=True,
                monitor='Gamma1.Luminance50',
                screen=0,
                warp=Warp.Spherical,
                )

# Paths for stimulus files

center_path = "drifting_gratings_center.stim"
surround_path = "drifting_gratings_surround.stim"
#annulus_path = "annulus.stim"
annulus_1_path = "drifting_gratings_annulus_1.stim"
annulus_2_path = "drifting_gratings_annulus_2.stim"
annulus_3_path = "drifting_gratings_annulus_3.stim"
annulus_4_path = "drifting_gratings_annulus_4.stim"
annulus_5_path = "drifting_gratings_annulus_5.stim"
lsn_path  = "locally_sparse_noise.stim"

path = r"C:\Users\Public\Desktop\pythondev\cam2p_scripts\tests\openscope_surround\grating_order.npy"
stim_order = np.load(path)

lsn = Stimulus.from_file(lsn_path, window)


# Create stimuli
center = Stimulus.from_file(center_path, window)
surround = Stimulus.from_file(surround_path, window)
annulus_1 = Stimulus.from_file(annulus_1_path, window)
annulus_2 = Stimulus.from_file(annulus_2_path, window)
annulus_3 = Stimulus.from_file(annulus_3_path, window)
annulus_4 = Stimulus.from_file(annulus_4_path, window)
annulus_5 = Stimulus.from_file(annulus_5_path, window)

posx = 20    #in degrees
posy = 10    #in degrees

center.stim.pos = (posx, posy)
annulus_1.stim.pos = (posx, posy)
annulus_2.stim.pos = (posx, posy)
annulus_3.stim.pos = (posx, posy)
annulus_4.stim.pos = (posx, posy)
annulus_5.stim.pos = (posx, posy)
surround.stim.pos = (posx, posy)

center.sweep_order = stim_order[:,1].astype(int).tolist()
center._build_frame_list()
surround.sweep_order = stim_order[:,2].astype(int).tolist()
surround._build_frame_list()
annulus_1.sweep_order = stim_order[:,2].astype(int).tolist()
annulus_1._build_frame_list()
annulus_2.sweep_order = stim_order[:,2].astype(int).tolist()
annulus_2._build_frame_list()
annulus_3.sweep_order = stim_order[:,2].astype(int).tolist()
annulus_3._build_frame_list()
annulus_4.sweep_order = stim_order[:,2].astype(int).tolist()
annulus_4._build_frame_list()
annulus_5.sweep_order = stim_order[:,2].astype(int).tolist()
annulus_5._build_frame_list()

# set display sequences
gratings_ds = [(0,1200),(1820,3020)]
lsn_ds=[(1210,1810),(3030,3630)]

center.set_display_sequence(gratings_ds)
surround.set_display_sequence(gratings_ds)
annulus_1.set_display_sequence(gratings_ds)
annulus_2.set_display_sequence(gratings_ds)
annulus_3.set_display_sequence(gratings_ds)
annulus_4.set_display_sequence(gratings_ds)
lsn.set_display_sequence(lsn_ds)

# kwargs
params = {
    'mouseid':'test2',    
    'syncsqr': True,
    'syncsqrloc': (545,340),
    'syncsqrsize': (80,140),
    'syncpulse': True,
    'syncpulseport': 1,
    'syncpulselines': [1, 2],  # frame, start/stop
    'trigger_delay_sec': 5.0,
}

# create SweepStim instance
ss = SweepStim(window,
               stimuli=[surround,annulus_1,annulus_2,annulus_3,annulus_4,annulus_5,center,lsn],
               pre_blank_sec=2,
               post_blank_sec=2,
               params=params,
               )

# add in foraging so we can track wheel, potentially give rewards, etc
f = Foraging(window=window,
             auto_update=False,
             params=params,
             nidaq_tasks={'digital_input': ss.di,
                          'digital_output': ss.do,})  #share di and do with SS
ss.add_item(f, "foraging")

# run it
ss.run()
