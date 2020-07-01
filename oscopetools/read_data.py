#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 21:59:56 2020

@author: saskiad
"""
from abc import ABC, abstractmethod
from copy import deepcopy

import matplotlib.pyplot as plt
import h5py
import pandas as pd
import numpy as np

FRAME_RATE = 30.0  # Assumed frame rate in Hz. TODO: load from a file


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
    return pd.read_hdf(file_path, 'roi_table')


def get_stimulus_table(file_path, stimulus):
    return pd.read_hdf(file_path, stimulus)


def get_stimulus_epochs(file_path, session_type):
    if session_type == 'drifting_gratings_grid':
        stim_name_1 = 'drifting_gratings_grid'
    elif session_type == 'center_surround':
        stim_name_1 = 'center_surround'
    elif session_type == 'size_tuning':
        stim_name_1 = np.NaN  # TODO: figure this out

    stim1 = get_stimulus_table(file_path, stim_name_1)
    stim2 = get_stimulus_table(file_path, 'locally_sparse_noise')
    stim_epoch = pd.DataFrame(columns=('Start', 'End', 'Stimulus_name'))
    break1 = np.where(np.ediff1d(stim1.Start) > 1000)[0][0]
    break2 = np.where(np.ediff1d(stim2.Start) > 1000)[0][0]
    stim_epoch.loc[0] = [stim1.Start[0], stim1.End[break1], stim_name_1]
    stim_epoch.loc[1] = [stim1.Start[break1 + 1], stim1.End.max(), stim_name_1]
    stim_epoch.loc[2] = [
        stim2.Start[0],
        stim2.End[break2],
        'locally_sparse_noise',
    ]
    stim_epoch.loc[3] = [
        stim2.Start[break2 + 1],
        stim2.End.max(),
        'locally_sparse_noise',
    ]
    stim_epoch.sort_values(by='Start', inplace=True)
    stim_epoch.loc[4] = [
        0,
        stim_epoch.Start.iloc[0] - 1,
        'spontaneous_activity',
    ]
    for i in range(1, 4):
        stim_epoch.loc[4 + i] = [
            stim_epoch.End.iloc[i - 1] + 1,
            stim_epoch.Start.iloc[i] - 1,
            'spontaneous_activity',
        ]
    stim_epoch.sort_values(by='Start', inplace=True)
    stim_epoch.reset_index(inplace=True)
    stim_epoch['Duration'] = stim_epoch.End - stim_epoch.Start

    return stim_epoch


def get_eye_tracking(file_path):
    raw_eyetracking_dataset = pd.read_hdf(file_path, 'eye_tracking')
    return EyeTracking(raw_eyetracking_dataset, 1.0 / FRAME_RATE)


class Dataset(ABC):
    """A dataset that is interesting to analyze on its own."""

    @abstractmethod
    def __init__(self):
        self._clean = False  # Whether quality control has been applied

    @abstractmethod
    def plot(self, ax=None, **pltargs):
        """Display a diagnostic plot.

        Parameters
        ----------
        ax : matplotlib.Axes object or None
            Axes object onto which to draw the diagnostic plot. Defaults to the
            current Axes if None.
        pltargs
            Parameters passed to `plt.plot()` (or similar) as keyword
            arguments. See `plt.plot` for a list of valid arguments. Examples:
            `color='red'`, `linestyle='dashed'`.

        Returns
        -------
        axes : Axes
            Axes object containing the diagnostic plot.

        """
        # Suggested implementation for derived classes:
        # def plot(self, type_specific_arguments, ax=None, **pltargs):
        #     ax = super().plot(ax=ax, **pltargs)
        #     ax.plot(relevant_data, **pltargs)  # pltargs might include color, linestyle, etc
        #     return ax  # ax should be returned so the user can change axis labels, etc

        # Default to the current Axes if none are supplied.
        if ax is None:
            ax = plt.gca()

        return ax

    @abstractmethod
    def apply_quality_control(self, inplace=False):
        """Clean up the dataset.

        Parameters
        ----------
        inplace : bool, default False
            Whether to clean up the current Dataset instance (ie, self) or
            a copy. In either case, a cleaned Dataset instance is returned.

        Returns
        -------
        dataset : Dataset
            A cleaned dataset.

        """
        # Suggested implementation for derived classes:
        # def apply_quality_control(self, type_specific_arguments, inplace=False):
        #     dset_to_clean = super().apply_quality_control(inplace)
        #     # Do stuff to `dset_to_clean` to clean it.
        #     dset_to_clean._clean = True
        #     return dset_to_clean

        # Get a reference to the dataset to be cleaned. Might be the current
        # dataset or a copy of it.
        if inplace:
            dset_to_clean = self
        else:
            dset_to_clean = self.copy()

        return dset_to_clean

    def copy(self):
        """Get a deep copy of the current Dataset."""
        return deepcopy(self)


class TimeseriesDataset(Dataset):
    """Abstract base class for Datasets containing timeseries."""

    def __init__(self, timestep_width):
        self._timestep_width = timestep_width

    def __len__(self):
        return self.num_timesteps

    @property
    @abstractmethod
    def num_timesteps(self):
        """Number of timesteps in timeseries."""
        raise NotImplementedError

    @property
    def timestep_width(self):
        """Width of each timestep in seconds."""
        return self._timestep_width

    @property
    def duration(self):
        """Duration of the timeseries in seconds."""
        return self.num_timesteps * self.timestep_width

    @property
    def time_vec(self):
        """A vector of timestamps the same length as the timeseries."""
        time_vec = np.arange(
            0, self.duration - 0.5 * self.timestep_width, self.timestep_width
        )
        assert len(time_vec) == len(
            self
        ), 'Length of time_vec ({}) does not match instance length ({})'.format(
            len(time_vec), len(self)
        )
        return time_vec

    def get_time_range(self, start: float, stop: float = None):
        """Extract a time window from the timeseries by time in seconds.

        Parameters
        ----------
        start, stop : float
            Beginning and end of the time window to extract in seconds. If
            `stop` is omitted, only the frame closest to `start` is returned.

        Returns
        -------
        windowed_timeseries : TimeseriesDataset
            A timeseries of the same type as the current instance containing
            only the frames in the specified window. Note that the `time_vec`
            of `windowed_timeseries` will start at 0, not `start`.

        """
        frame_range = [
            self._get_nearest_frame(t_)
            for t_ in (start, stop)
            if t_ is not None
        ]
        return self.get_frame_range(*frame_range)

    @abstractmethod
    def get_frame_range(self, start: int, stop: int = None):
        """Extract a time window from the timeseries by frame number.

        Parameters
        ----------
        start, stop : int
            Beginning and end of the time window to extract in frames. If
            `stop` is omitted, only the `start` frame is returned.

        Returns
        -------
        windowed_timeseries : TimeseriesDataset
            A timeseries of the same type as the current instance containing
            only the frames in the specified window. Note that the `time_vec`
            of `windowed_timeseries` will start at 0, not `start`.

        """
        raise NotImplementedError

    def _get_nearest_frame(self, time_: float):
        """Round a timestamp to the nearest integer frame number."""
        if time_ <= 0.0:
            raise ValueError(
                'Expected `time_` to be >= 0, got {}'.format(time_)
            )

        frame_num = int(np.round(time_ / self.timestep_width))
        assert frame_num <= len(self)

        return min(frame_num, len(self) - 1)


class RawFluorescence(TimeseriesDataset):
    pass


class DeltaFluorescence(TimeseriesDataset):
    # This cannot be instantiated until all of the methods of its parents have
    # been implemented.
    pass


class EyeTracking(TimeseriesDataset):
    _x_pos_name = 'x_pos_deg'
    _y_pos_name = 'y_pos_deg'

    def __init__(
        self, tracked_attributes: pd.DataFrame, timestep_width: float
    ):
        super().__init__(timestep_width)
        self._dframe = pd.DataFrame(tracked_attributes)

    @property
    def num_timesteps(self):
        """Number of timesteps in EyeTracking dataset."""
        return self._dframe.shape[0]

    def get_frame_range(self, start: int, stop: int = None):
        window = self.copy()
        if stop is not None:
            window._dframe = window._dframe.iloc[start:stop, :]
        else:
            window._dframe = window._dframe.iloc[start, :]

        return window

    def plot(self, channel='position', ax=None, **pltargs):
        """Make a diagnostic plot of eyetracking data."""
        ax = super().plot(ax, **pltargs)

        # Check whether the `channel` argument is valid
        if channel not in self._dframe.columns and channel != 'position':
            raise ValueError(
                'Got unrecognized channel `{}`, expected one of '
                '{} or `position`'.format(
                    channel, self._dframe.columns.tolist()
                )
            )

        if channel in self._dframe.columns:
            ax.plot(self.time_vec, self._dframe[channel], **pltargs)
        elif channel == 'position':
            ax.plot(
                self._dframe[self._x_pos_name],
                self._dframe[self._y_pos_name],
                **pltargs
            )
        else:
            raise NotImplementedError(
                'Plotting for channel {} is not implemented.'.format(channel)
            )

        return ax

    def apply_quality_control(self, inplace=False):
        super().apply_quality_control(inplace)
        raise NotImplementedError


class RunningSpeed(TimeseriesDataset):
    pass
