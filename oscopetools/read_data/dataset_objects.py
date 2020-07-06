"""Classes for interacting with OpenScope datasets."""
__all__ = ('RawFluorescence', 'TrialFluorescence', 'EyeTracking')

from abc import ABC, abstractmethod
from copy import deepcopy
import warnings

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


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
        if time_ <= 0.0:
            raise ValueError(
                'Expected `time_` to be >= 0, got {}'.format(time_)
            )

        frame_num = int(np.round(time_ / self.timestep_width))
        assert frame_num <= len(self)

        return min(frame_num, len(self) - 1)


class SliceParseError(Exception):
    pass


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
            trial_range = self._try_parse_positionals_as_slice_like(*args)
            mask = self._trial_range_to_mask(*trial_range)
        except SliceParseError:
            # Couldn't parse pos args as a range of trials. Try parsing as
            # a boolean trial mask.
            if len(args) == 1:
                mask = self._validate_trial_mask_shape(args[0])
            else:
                raise ValueError(
                    'Expected a single mask argument, got {}'.format(len(args))
                )

        return self._get_trials_from_mask(mask)

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

    def _try_parse_positionals_as_slice_like(self, *args):
        if len(args) == 0:
            # Case: Positional arguments are empty
            raise ValueError('Empty positional arguments')
        elif len(args) == 1:
            # Case: Positional arguments contain a single element.
            # If it's an integer, use that as the value for slice `start`
            # If it's a tuple, try to use it as a `(start, stop)` pair
            try:
                # Check if args contains a single integer scalar
                if int(args[0]) == args[0]:
                    return [args[0]]
                else:
                    raise SliceParseError(
                        'Positional argument {} is not int-like'.format(
                            args[0]
                        )
                    )
            except TypeError:
                if (len(args[0]) == 1) or (len(args[0]) == 2):
                    return np.atleast_1d(args[0])
                else:
                    raise SliceParseError(
                        'Found more than two elements in tuple {}'.format(
                            args[0]
                        )
                    )
        elif len(args) == 2:
            return args
        else:
            raise SliceParseError

    def _validate_trial_mask_shape(self, mask):
        if np.ndim(mask) != 1:
            raise ValueError(
                'Expected mask to be vector-like, got '
                '{}D array instead'.format(np.ndim(mask))
            )

        mask = np.asarray(mask).flatten()
        if len(mask) != self.num_trials:
            raise ValueError(
                'len of mask {} does not match number of '
                'trials {}'.format(len(mask), self.num_trials)
            )

        return mask

    def _trial_range_to_mask(self, start, stop=None):
        """Convert a range of trials to a boolean trial mask."""
        if stop is not None:
            mask = self.trial_vec >= start
            mask &= self.trial_vec < stop
        else:
            mask = self.trial_vec == start
        return mask


class Fluorescence(TimeseriesDataset):
    """A fluorescence timeseries.

    Any fluorescence timeseries. May have one or more cells and one or more
    trials.

    """

    @property
    def num_timesteps(self):
        """Number of timesteps."""
        return self.fluo.shape[-1]

    @property
    def num_cells(self):
        """Number of ROIs."""
        return self.fluo.shape[-2]

    def get_frame_range(self, start, stop=None):
        """Get a time window by frame number."""
        fluo_copy = self.copy()

        if stop is None:
            time_slice = self.fluo[..., start][..., np.newaxis]
        else:
            time_slice = self.fluo[..., start:stop]

        fluo_copy.fluo = time_slice
        return fluo_copy


class RawFluorescence(Fluorescence):
    """Fluorescence timeseries from a full imaging session.

    Not divided into trials.

    """

    def __init__(self, fluorescence_array, timestep_width):
        fluorescence_array = np.asarray(fluorescence_array)
        assert fluorescence_array.ndim == 2

        super().__init__(timestep_width)

        self.fluo = fluorescence_array
        self.is_z_score = False
        self.is_dff = False

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
                    ' to be ints, got {} and {} instead'.format(
                        start, end
                    )
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
            np.asarray(trials),
            trial_num,
            self.timestep_width,
        )
        trial_fluorescence.is_z_score = self.is_z_score
        trial_fluorescence.is_dff = self.is_dff
        trial_fluorescence._baseline_duration = num_baseline_frames * self.timestep_width

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


class TrialFluorescence(TrialDataset, Fluorescence):
    """Fluorescence timeseries divided into trials."""

    def __init__(self, fluorescence_array, trial_num, timestep_width):
        fluorescence_array = np.asarray(fluorescence_array)
        assert fluorescence_array.ndim == 3
        assert fluorescence_array.shape[0] == len(trial_num)

        self._timestep_width = timestep_width

        self._baseline_duration = 0
        self.fluo = fluorescence_array
        self._trial_num = np.asarray(trial_num)
        self.is_z_score = False
        self.is_dff = False

    @property
    def time_vec(self):
        time_vec_without_baseline = super().time_vec
        return time_vec_without_baseline - self._baseline_duration

    def _get_trials_from_mask(self, mask):
        trial_subset = self.copy()
        trial_subset._trial_num = trial_subset._trial_num[mask]
        trial_subset.fluo = trial_subset.fluo[mask, ...]

        return trial_subset

    def plot(self, ax=None, **pltargs):
        if ax is None:
            ax = plt.gca()

        ax.imshow(self.fluo.mean(axis=0), **pltargs)

        return ax

    def apply_quality_control(self, inplace=False):
        raise NotImplementedError


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
