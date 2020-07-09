import os
import argparse

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs

from oscopetools import read_data as rd

parser = argparse.ArgumentParser()
parser.add_argument(
    'DATA_PATH', help='Path to folder with center-surround data files.'
)
parser.add_argument(
    '-o', '--output', help='Path to folder in which to place diagnostic plots.'
)

args = parser.parse_args()


def _complete_circle(ls):
    """Close a circular list. Use to join 0 deg and 360 deg in polar plots."""
    ls.append(ls[0])


# Define placement of plots
## Define gridspec layout
row_spec = gs.GridSpec(
    2,
    1,
    left=0.1,
    top=0.9,
    bottom=0.1,
    right=0.95,
    hspace=0.35,
    height_ratios=[0.7, 0.3],
)
top_spec = gs.GridSpecFromSubplotSpec(
    1, 3, row_spec[0, :], width_ratios=[0.65, 0.2, 0.15], wspace=0.3
)
mean_spec = gs.GridSpecFromSubplotSpec(
    2, 4, top_spec[:, 0], wspace=0.45, hspace=0.5
)
max_resp_spec = gs.GridSpecFromSubplotSpec(1, 3, row_spec[1, :], wspace=0.3)

## Get lists to use for placing plots.
## Gridspecs can be safely ignored from here on.
mean_slots = [mean_spec[i // 4, i % 4] for i in range(8)]
polar_slot = top_spec[:, 1]
waterfall_slot = top_spec[:, 2]
max_resp_slots = [max_resp_spec[0, i] for i in range(3)]

del row_spec, top_spec, mean_spec, max_resp_spec

# Make a set of plots for each file.

STIM_TIME_WINDOW = (0, 2)

# ITERATE OVER FILES
for dfile in os.listdir(args.DATA_PATH):
    if not dfile.endswith('.h5'):
        continue

    stim_table = rd.get_stimulus_table(
        os.path.join(args.DATA_PATH, dfile), 'center_surround',
    )
    dff_fluo = rd.get_dff_traces(os.path.join(args.DATA_PATH, dfile))
    dff_fluo.z_score()  # Convert to Z-score
    trial_fluo = dff_fluo.cut_by_trials(stim_table, num_baseline_frames=30)

    # ITERATE OVER ALL CELLS IN A FILE
    for cell_num, cell_fluo in tqdm(trial_fluo.iter_cells()):

        plt.figure(figsize=(9, 7))
        plt.suptitle('{} cell {}'.format(dfile.strip('.h5'), cell_num))

        # Create a set of axes for plotting the mean response of each cell
        mean_axes = [plt.subplot(mean_slots[0])]
        mean_axes.extend(
            [plt.subplot(spec, sharey=mean_axes[0]) for spec in mean_slots[1:]]
        )

        # Plot the mean response for each surround condition,
        # and collect the max response for each condition at the same time.
        # (Use this later to plot the preferred orientation of each cell.)
        orientations = []
        max_responses = {
            'no_surround': [],
            'ortho_surround': [],
            'iso_surround': [],
        }
        for i, ori in enumerate(rd.Orientation):
            mean_axes[i].set_title(str(int(ori)))
            orientations.append(ori)

            ## Plot NO-SURROUND trials
            no_surround = stim_table['center_surround'].apply(
                lambda x: (x.center_orientation == ori)
                and x.surround_is_empty()
            )
            no_surround_trials = cell_fluo.get_trials(no_surround)
            no_surround_trials.plot(ax=mean_axes[i], alpha=0.7)
            max_responses['no_surround'].append(
                no_surround_trials.get_time_range(*STIM_TIME_WINDOW)
                .trial_mean()
                .data.max()
            )

            ## Plot ORTHOGONAL surround trials
            ortho_surround = stim_table['center_surround'].apply(
                lambda x: (x.center_orientation == ori)
                and (x.surround_orientation in ori.orthogonal())
            )
            ortho_surround_trials = cell_fluo.get_trials(ortho_surround)
            ortho_surround_trials.plot(ax=mean_axes[i], alpha=0.7)
            max_responses['ortho_surround'].append(
                ortho_surround_trials.get_time_range(*STIM_TIME_WINDOW)
                .trial_mean()
                .data.max()
            )

            ## Plot ISO surround trials
            iso_surround = stim_table['center_surround'].apply(
                lambda x: (x.center_orientation == ori)
                and (x.surround_orientation == x.center_orientation)
            )
            iso_surround_trials = cell_fluo.get_trials(iso_surround)
            iso_surround_trials.plot(ax=mean_axes[i], alpha=0.7)
            max_responses['iso_surround'].append(
                iso_surround_trials.get_time_range(*STIM_TIME_WINDOW)
                .trial_mean()
                .data.max()
            )

            mean_axes[i].legend().remove()
            if i < 4:
                mean_axes[i].set_xlabel('')
            if (i % 4) == 0:
                mean_axes[i].set_ylabel('Z-score')

        # Polar plot of angular tuning
        ## Link 0 deg and 360 deg inplace with `_complete_circle`
        _complete_circle(orientations)
        orientations_in_rad = [ori.radians for ori in orientations]
        {_complete_circle(val) for val in max_responses.values()}

        polar_ax = plt.subplot(polar_slot, polar=True)
        polar_ax.set_title(
            'Max resp. in {} window'.format(STIM_TIME_WINDOW),
            pad=25
        )
        for surround_condition in max_responses:
            polar_ax.fill_between(
                orientations_in_rad,
                np.zeros_like(orientations_in_rad),
                np.clip(max_responses[surround_condition], 0, np.inf),
                alpha=0.5,
            )
            polar_ax.plot(
                orientations_in_rad,
                np.clip(max_responses[surround_condition], 0, np.inf),
                label=surround_condition,
            )
        polar_ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2))

        # Plot all trials in chronological order
        water_ax = plt.subplot(waterfall_slot)
        water_ax.set_title('All trials')
        water_ax.imshow(cell_fluo.data[:, 0, :], aspect='auto')
        water_ax.set_yticks([])
        water_ax.set_xticks([])

        # Plot trial-resolved response for preferred orientation
        preferred_orientation = orientations[
            np.argmax(max_responses['iso_surround'])
        ]
        surround_orientations = {
            'no surr': [rd.Orientation(None)],
            'ortho surr': preferred_orientation.orthogonal(),
            'iso surr': [preferred_orientation],
        }

        for (surr_condition, surr_orientations_), plot_slot in zip(
            surround_orientations.items(), max_resp_slots
        ):
            trial_resolved_ax = plt.subplot(plot_slot)
            trial_resolved_ax.set_title(
                'Center {} + {}'.format(int(preferred_orientation), surr_condition)
            )

            trial_mask = stim_table['center_surround'].apply(
                lambda x: (x.center_orientation == preferred_orientation)
                and (x.surround_orientation in surr_orientations_)
            )
            trial_resolved_ax.plot(
                cell_fluo.time_vec,
                cell_fluo.get_trials(trial_mask).data[:, 0, :].T,
                'k-',
                alpha=0.5,
            )
            trial_resolved_ax.set_xlabel('Time (s)')
            trial_resolved_ax.set_ylabel('Z-score')

        plt.savefig(
            os.path.join(
                args.output, '{}_{}.png'.format(dfile.strip('.h5'), cell_num)
            ),
            dpi=600,
        )
        plt.close()
