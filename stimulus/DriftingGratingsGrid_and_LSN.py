"""
DriftingGratingsGrid.py

Volume Imaging Day 1.

"""
from psychopy import visual
from camstim import Stimulus, SweepStim
from camstim import Foraging
from camstim import Window, Warp

# Create display window
window = Window(
    fullscr=True, monitor='Gamma1.Luminance50', screen=0, warp=Warp.Spherical,
)

# Paths for stimulus files
dg_path = "drifting_gratings_grid_5.stim"
lsn_path = "locally_sparse_noise.stim"
# dg_path = r"C:\Users\Public\Desktop\pythondev\cam2p_scripts\tests\openscope_surround\drifting_gratings_grid_5.stim"

# Create stimuli
dg = Stimulus.from_file(dg_path, window)
lsn = Stimulus.from_file(lsn_path, window)

# set display sequences
dg_ds = [(0, 915), (2735, 3650)]
lsn_ds = [(925, 2725)]

dg.set_display_sequence(dg_ds)
lsn.set_display_sequence(lsn_ds)

# kwargs
params = {
    'syncsqrloc': (510, 360),  # added by DM
    'syncsqrsize': (50, 140),  # added by DM
    'syncpulse': True,
    'syncpulseport': 1,
    'syncpulselines': [1, 2],  # [5, 6],  # frame, start/stop
    'trigger_delay_sec': 5.0,
}

# create SweepStim instance
ss = SweepStim(
    window, stimuli=[dg, lsn], pre_blank_sec=2, post_blank_sec=2, params=params,
)

# add in foraging so we can track wheel, potentially give rewards, etc
f = Foraging(
    window=window,
    auto_update=False,
    params=params,
    nidaq_tasks={'digital_input': ss.di, 'digital_output': ss.do,},
)  # share di and do with SS
ss.add_item(f, "foraging")

# run it
ss.run()
