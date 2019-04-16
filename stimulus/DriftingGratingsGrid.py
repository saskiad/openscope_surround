"""
three_session_A.py
"""
from psychopy import visual
from camstim import Stimulus, SweepStim
from camstim import Foraging
from camstim import Window, Warp

# Create display window
window = Window(fullscr=True,
                monitor='GammaCorrect30',
                screen=1,
                warp=Warp.Spherical,
                )

# Paths for stimulus files
dg_path = r"C:\Users\svc_ncbehavior\Desktop\stimulus\cam2p_scripts\cam_1_0\drifting_gratings.stim"

# Create stimuli
dg = Stimulus.from_file(dg_path, window)

# set display sequences
dg_ds = [(0, 1830)]

dg.set_display_sequence(dg_ds)

# kwargs
params = {
    'syncsqr': True,
    'syncsqrloc': (510,360),
    'syncsqrsize': (50,140),
    'syncpulse': True,
    'syncpulseport': 1,
    'syncpulselines': [1, 2],  # frame, start/stop
    'trigger_delay_sec': 5.0,
}

# create SweepStim instance
ss = SweepStim(window,
               stimuli=[dg],
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
