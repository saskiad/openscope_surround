import os
import argparse

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs

from oscopetools import read_data as rd

parser = argparse.ArgumentParser()
parser.add_argument('DATA_PATH', help='Path to folder with data files.')
parser.add_argument(
    '-o', '--output', help='Path to folder in which to place diagnostic plots.'
)

args = parser.parse_args()

spec = gs.GridSpec(2, 3, width_ratios=[1, 1, 0.5])

# ITERATE OVER FILES
for dfile in tqdm(os.listdir(args.DATA_PATH)):
    if not dfile.endswith('.h5'):
        continue

    eyetracking = rd.get_eye_tracking(os.path.join(args.DATA_PATH, dfile))
    runningspeed = rd.get_running_speed(os.path.join(args.DATA_PATH, dfile))

    plt.figure(figsize=(8, 4))

    plt.subplot(spec[0, 0])
    plt.title('Running speed')
    runningspeed.plot()
    plt.xlabel('')
    plt.xticks([])

    plt.subplot(spec[1, 0])
    runningspeed.plot(robust_range_=True, lw=0.7)

    plt.subplot(spec[0, 1])
    plt.title('Pupil area')
    eyetracking.plot('pupil_area')
    plt.xlabel('')
    plt.xticks([])

    plt.subplot(spec[1, 1])
    eyetracking.plot('pupil_area', robust_range_=True, lw=0.7)

    plt.subplot(spec[0, 2])
    plt.title('Position')
    eyetracking.plot(
        'position',
        marker='o',
        ls='none',
        markeredgecolor='gray',
        markeredgewidth=0.5,
        alpha=0.7,
    )

    plt.subplot(spec[1, 2])
    eyetracking.plot('position', style='density', robust_range_=True)

    plt.tight_layout()
    plt.savefig(
        os.path.join(args.output, dfile.strip('.h5') + '.png'), dpi=600
    )

    plt.close()
