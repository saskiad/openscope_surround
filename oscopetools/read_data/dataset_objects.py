"""Classes for interacting with OpenScope datasets."""
__all__ = ('RawFluorescence', 'TrialFluorescence', 'EyeTracking')

from abc import ABC, abstractmethod
from copy import deepcopy
import warnings

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


class SliceParseError(Exception):
    pass


def _try_parse_positionals_as_slice_like(*args):
    """Try to parse positional arguments as a slice-like int or pair of ints.

    Output can be treated as a `(start, stop)` range (where `stop` is optional)
    on success, and can be treated as a boolean mask if a `SliceParseError` is
    raised.

    Returns
    -------
    slice_like : [int] or [int, int]

    Raises
    ------
    SliceParseError
        If positional arguments are a boolean mask, not a slice.
    TypeError
        If positional arguments are not bool-like or int-like.
    ValueError
        If positional arguments are empty or have more than two entries.

    """
    flattened_args = np.asarray(args).flatten()
    if len(flattened_args) == 0:
        raise ValueError('Empty positional arguments')
    elif _is_bool(flattened_args[0]):
        raise SliceParseError('Cannot parse bool positionals as slice.')
    elif int(flattened_args[0]) != flattened_args[0]:
        raise TypeError(
            'Expected positionals to be bool-like or int-like, '
            'got type {} instead'.format(flattened_args.dtype)
        )
    elif (len(flattened_args) > 0) and (len(flattened_args) <= 2):
        # Positional arguments are a valid slice-like int or pair of ints
        return flattened_args.tolist()
    else:
        # Case: positionals are not bool and are of the wrong length
        raise ValueError(
            'Positionals of length {} cannot be parsed as slice-like'.format(
                len(flattened_args)
            )
        )


def _is_bool(x):
    return isinstance(x, (bool, np.bool, np.bool8, np.bool_))


def _validate_vector_mask_length(mask, expected_length):
    if np.ndim(mask) != 1:
        raise ValueError(
            'Expected mask to be vector-like, got '
            '{}D array instead'.format(np.ndim(mask))
        )

    mask = np.asarray(mask).flatten()
    if len(mask) != expected_length:
        raise ValueError(
            'Expected mask of length {}, got mask of '
            'length {} instead.'.format(len(mask), expected_length)
        )

    return mask


def _get_vector_mask_from_range(values_to_mask, start, stop=None):
    """Unmask all values within a range."""
    if stop is not None:
        mask = values_to_mask >= start
        mask &= values_to_mask < stop
    else:
        mask = values_to_mask == start
    return mask


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

    def get_time_range(self, start, stop=None):
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
    def get_frame_range(self, start, stop=None):
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

    def _get_nearest_frame(self, time_):
        """Round a timestamp to the nearest integer frame number."""
        frame_num = int(np.round(time_ / self.timestep_width))
        assert frame_num <= len(self)

        return min(frame_num, len(self) - 1)


class TrialDataset(Dataset):
    """Abstract base class for datasets that are divided into trials.

    All children should have a list-like `_trial_num` attribute.

    """

    @property
    def num_trials(self):
        """Number of trials."""
        return len(self._trial_num)

    @property
    def trial_vec(self):
        """Trial numbers."""
        return self._trial_num

    def get_trials(self, *args):
        """Get a subset of the trials in TrialDataset.

        Parameters
        ----------
        start, stop : int
            Get a range of trials from `start` to an optional `stop`.
        mask : bool vector-like
            A boolean mask used to select trials.

        Returns
        -------
        trial_subset : TrialDataset
            A new `TrialDataset` object containing only the specified trials.

        """
        # Implementation note:
        # This function tries to convert positional arguments to a boolean
        # trial mask. `_get_trials_from_mask` is reponsible for actually
        # getting the `trial_subset` to be returned.
        try:
            # Try to parse positional arguments as a range of trials
            trial_range = _try_parse_positionals_as_slice_like(*args)
            mask = _get_vector_mask_from_range(self.trial_vec, *trial_range)
        except SliceParseError:
            # Couldn't parse pos args as a range of trials. Try parsing as
            # a boolean trial mask.
            if len(args) == 1:
                mask = _validate_vector_mask_length(args[0], self.num_trials)
            else:
                raise ValueError(
                    'Expected a single mask argument, got {}'.format(len(args))
                )

        return self._get_trials_from_mask(mask)

    def iter_trials(self):
        """Get an iterator over all trials.

        Yields
        ------
        (trial_num, trial_contents): (int, TrialDataset)
            Yields a tuple containing the trial number and a `TrialDataset`
            containing that trial for each trial in the original
            `TrialDataset`.

        Example
        -------
        >>> trials = TrialDataset()
        >>> for trial_num, trial in trials.iter_trials():
        >>>     print(trial_num)
        >>>     trial.plot()

        """
        for trial_num in self.trial_vec:
            yield (trial_num, self.get_trials(trial_num))

    @abstractmethod
    def _get_trials_from_mask(self, mask):
        """Get a subset of trials using a boolean mask.

        Subclasses are required to implement this method to get the rest of
        TrialDataset functionality.

        Parameters
        ----------
        mask : bool vector-like
            A boolean trial mask, the length of which is guaranteed to match
            the number of trials.

        Returns
        -------
        trial_subset : TrialDataset
            A new `TrialDataset` object containing only the specified trials.

        """
        raise NotImplementedError

    @abstractmethod
    def trial_mean(self):
        """Get the mean across all trials.

        Returns
        -------
        trial_mean : TrialDataset
            A new `TrialDataset` object containing the mean of all trials in
            the current one.

        """
        raise NotImplementedError

    @abstractmethod
    def trial_std(self):
        """Get the standard deviation across all trials.

        Returns
        -------
        trial_std : TrialDataset
            A new `TrialDataset` object containing the standard deviation of
            all trials in the current one.

        """
        raise NotImplementedError


class Fluorescence(TimeseriesDataset):
    """A fluorescence timeseries.

    Any fluorescence timeseries. May have one or more cells and one or more
    trials.

    """

    def __init__(self, fluorescence_array, timestep_width):
        super().__init__(timestep_width)

        self.fluo = np.asarray(fluorescence_array)
        self.cell_vec = np.arange(0, self.num_cells)
        self.is_z_score = False
        self.is_dff = False

    @property
    def num_timesteps(self):
        """Number of timesteps."""
        return self.fluo.shape[-1]

    @property
    def num_cells(self):
        """Number of ROIs."""
        return self.fluo.shape[-2]

    def get_cells(self, *args):
        # Implementation note:
        # This function tries to convert positional arguments to a boolean
        # cell mask. `_get_cells_from_mask` is reponsible for actually
        # getting the `cell_subset` to be returned.
        try:
            # Try to parse positional arguments as a range of cells
            cell_range = _try_parse_positionals_as_slice_like(*args)
            mask = _get_vector_mask_from_range(self.cell_vec, *cell_range)
        except SliceParseError:
            # Couldn't parse pos args as a range of cells. Try parsing as
            # a boolean cell mask.
            if len(args) == 1:
                mask = _validate_vector_mask_length(args[0], self.num_cells)
            else:
                raise ValueError(
                    'Expected a single mask argument, got {}'.format(len(args))
                )

        return self._get_cells_from_mask(mask)

    def iter_cells(self):
        """Get an iterator over all cells in the fluorescence dataset.

        Yields
        ------
        (cell_num, cell_fluorescence) : (int, Fluorescence)
            Yields a tuple of the cell number and fluorescence for each cell.

        Example
        -------
        >>> fluo_dset = Fluorescence()
        >>> for cell_num, cell_fluorescence in fluo_dset.iter_cells():
        >>>     print('Cell number {}'.format(cell_num))
        >>>     cell_fluorescence.plot()

        """
        for cell_num in self.cell_vec:
            yield (cell_num, self.get_cells(cell_num))

    def get_frame_range(self, start, stop=None):
        """Get a time window by frame number."""
        fluo_copy = self.copy(read_only=True)

        if stop is None:
            time_slice = self.fluo[..., start][..., np.newaxis]
        else:
            time_slice = self.fluo[..., start:stop]

        fluo_copy.fluo = time_slice.copy()
        return fluo_copy

    def copy(self, read_only=False):
        """Get a deep copy.

        Parameters
        ----------
        read_only : bool, default False
            Whether to get a read-only copy of the underlying `fluo` array.
            Getting a read-only copy is much faster and should be used if a
            large number of copies need to be created.

        """
        if read_only:
            # Get a read-only view of the fluo array
            # This is much faster than creating a full copy
            read_only_fluo = self.fluo.view()
            read_only_fluo.flags.writeable = False

            deepcopy_memo = {id(self.fluo): read_only_fluo}
            copy_ = deepcopy(self, deepcopy_memo)
        else:
            copy_ = deepcopy(self)

        return copy_

    def _get_cells_from_mask(self, mask):
        cell_subset = self.copy(read_only=False)
        cell_subset.cell_vec = self.cell_vec[mask].copy()
        cell_subset.fluo = self.fluo[..., mask, :].copy()

        assert cell_subset.fluo.ndim == self.fluo.ndim
        assert cell_subset.num_cells == np.sum(mask)

        return cell_subset


class RawFluorescence(Fluorescence):
    """Fluorescence timeseries from a full imaging session.

    Not divided into trials.

    """

    def __init__(self, fluorescence_array, timestep_width):
        fluorescence_array = np.asarray(fluorescence_array)
        assert fluorescence_array.ndim == 2

        super().__init__(fluorescence_array, timestep_width)

    def z_score(self):
        """Convert to Z-score."""
        if self.is_z_score:
            raise ValueError('Instance is already a Z-score')
        else:
            z_score = self.fluo - self.fluo.mean(axis=1)[:, np.newaxis]
            z_score /= z_score.std(axis=1)[:, np.newaxis]
            self.fluo = z_score
            self.is_z_score = True

    def cut_by_trials(self, trial_timetable, num_baseline_frames=None):
        """Divide fluorescence traces up into equal-length trials.

        Parameters
        ----------
        trial_timetable : pd.DataFrame-like
            A DataFrame-like object with 'Start' and 'End' items for the start
            and end frames of each trial, respectively.

        Returns
        -------
        trial_fluorescence : TrialFluorescence

        """
        if ('Start' not in trial_timetable) or ('End' not in trial_timetable):
            raise ValueError(
                'Could not find `Start` and `End` in trial_timetable.'
            )

        if (num_baseline_frames is None) or (num_baseline_frames < 0):
            num_baseline_frames = 0

        # Slice the RawFluorescence up into trials.
        trials = []
        num_frames = []
        for start, end in zip(
            trial_timetable['Start'], trial_timetable['End']
        ):
            # Coerce `start` and `end` to ints if possible
            if (int(start) != start) or (int(end) != end):
                raise ValueError(
                    'Expected trial start and end frame numbers'
                    ' to be ints, got {} and {} instead'.format(start, end)
                )
            start = max(int(start) - num_baseline_frames, 0)
            end = int(end)

            trials.append(self.fluo[..., start:end])
            num_frames.append(end - start)

        # Truncate all trials to the same length if necessary
        min_num_frames = min(num_frames)
        if not all([dur == min_num_frames for dur in num_frames]):
            warnings.warn(
                'Truncating all trials to shortest duration {} '
                'frames (longest trial is {} frames)'.format(
                    min_num_frames, max(num_frames)
                )
            )
            for i in range(len(trials)):
                trials[i] = trials[i][..., :min_num_frames]

        # Try to get a vector of trial numbers
        try:
            trial_num = trial_timetable['trial_num']
        except KeyError:
            try:
                trial_num = trial_timetable.index.tolist()
            except AttributeError:
                warnings.warn(
                    'Could not get trial_num from trial_timetable. '
                    'Falling back to arange.'
                )
                trial_num = np.arange(0, len(trials))

        # Construct TrialFluorescence and return it.
        trial_fluorescence = TrialFluorescence(
            np.asarray(trials), trial_num, self.timestep_width,
        )
        trial_fluorescence.is_z_score = self.is_z_score
        trial_fluorescence.is_dff = self.is_dff
        trial_fluorescence._baseline_duration = (
            num_baseline_frames * self.timestep_width
        )

        # Check that trial_fluorescence was constructed correctly.
        assert trial_fluorescence.num_cells == self.num_cells
        assert trial_fluorescence.num_timesteps == min_num_frames
        assert trial_fluorescence.num_trials == len(trials)

        return trial_fluorescence

    def plot(self, ax=None, **pltargs):
        if ax is not None:
            ax = plt.gca()

        ax.imshow(self.fluo, **pltargs)

        return ax

    def apply_quality_control(self, inplace=False):
        raise NotImplementedError


class TrialFluorescence(Fluorescence, TrialDataset):
    """Fluorescence timeseries divided into trials."""

    def __init__(self, fluorescence_array, trial_num, timestep_width):
        fluorescence_array = np.asarray(fluorescence_array)
        assert fluorescence_array.ndim == 3
        assert fluorescence_array.shape[0] == len(trial_num)

        super().__init__(fluorescence_array, timestep_width)

        self._baseline_duration = 0
        self._trial_num = np.asarray(trial_num)

    @property
    def time_vec(self):
        time_vec_without_baseline = super().time_vec
        return time_vec_without_baseline - self._baseline_duration

    def plot(self, ax=None, **pltargs):
        if ax is None:
            ax = plt.gca()

        if self.num_cells == 1:
            # If there is only one cell, make a line plot
            alpha = pltargs.pop('alpha', 1)

            fluo_mean = self.trial_mean().fluo[0, 0, :]
            fluo_std = self.trial_std().fluo[0, 0, :]
            ax.fill_between(
                self.time_vec,
                fluo_mean - fluo_std,
                fluo_mean + fluo_std,
                label='Mean $\pm$ SD',
                alpha=alpha * 0.6,
                **pltargs
            )
            ax.plot(self.time_vec, fluo_mean, alpha=alpha, **pltargs)
            ax.set_xlabel('Time (s)')
            ax.legend()
        else:
            # If there are many cells, just show the mean as a matrix.
            ax.imshow(self.trial_mean().fluo[0, ...], **pltargs)

        return ax

    def apply_quality_control(self, inplace=False):
        raise NotImplementedError

    def _get_trials_from_mask(self, mask):
        trial_subset = self.copy(read_only=True)
        trial_subset._trial_num = trial_subset._trial_num[mask].copy()
        trial_subset.fluo = trial_subset.fluo[mask, ...].copy()

        return trial_subset

    def trial_mean(self, ignore_nan=False):
        """Get the mean fluorescence for each cell across all trials.

        Parameters
        ----------
        ignore_nan : bool, default False
            Whether to return the `mean` or `nanmean` for each cell.

        Returns
        -------
        trial_mean : TrialFluoresence
            A new `TrialFluorescence` object with the mean across trials.

        See Also
        --------
        `trial_std()`

        """
        trial_mean = self.copy(read_only=True)
        trial_mean._trial_num = np.asarray([np.nan])

        if ignore_nan:
            trial_mean.fluo = np.nanmean(self.fluo, axis=0)[np.newaxis, :, :]
        else:
            trial_mean.fluo = self.fluo.mean(axis=0)[np.newaxis, :, :]

        return trial_mean

    def trial_std(self, ignore_nan=False):
        """Get the standard deviation of the fluorescence for each cell across trials.

        Parameters
        ----------
        ignore_nan : bool, default False
            Whether to return the `std` or `nanstd` for each cell.

        Returns
        -------
        trial_std : TrialFluorescence
            A new `TrialFluorescence` object with the standard deviation across
            trials.

        See Also
        --------
        `trial_mean()`

        """
        trial_std = self.copy(read_only=True)
        trial_std._trial_num = np.asarray([np.nan])

        if ignore_nan:
            trial_std.fluo = np.nanstd(self.fluo, axis=0)[np.newaxis, :, :]
        else:
            trial_std.fluo = self.fluo.std(axis=0)[np.newaxis, :, :]

        return trial_std


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
            window._dframe = window._dframe.iloc[start:stop, :].copy()
        else:
            window._dframe = window._dframe.iloc[start, :].copy()

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
            if pltargs.pop('style', None) in ['contour', 'density']:
                x = self._dframe[self._x_pos_name]
                y = self._dframe[self._y_pos_name]
                mask = np.isnan(x) | np.isnan(y)
                if any(mask):
                    warnings.warn(
                        'Dropping {} NaN entries in order to estimate '
                        'density.'.format(sum(mask))
                    )
                sns.kdeplot(x[~mask], y[~mask], ax=ax, **pltargs)
            else:
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
