"""Types for representing experimental conditions."""

__all__ = (
    'Orientation',
    'TemporalFrequency',
    'SpatialFrequency',
    'Contrast',
    'CenterSurroundStimulus',
)

import warnings

import numpy as np


class _IterableNamedOrderedSet(type):
    def __iter__(cls):
        for member in cls._MEMBERS:
            yield cls(member)


class SetMembershipError(Exception):
    pass


class _NamedOrderedSet(metaclass=_IterableNamedOrderedSet):
    _MEMBERS = ()

    def __init__(self, member_value):
        if member_value in self._MEMBERS:
            self._member_value = member_value
        elif np.isnan(member_value):
            self._member_value = None
        else:
            raise SetMembershipError(
                'Unrecognized member {}, expected '
                'one of {}'.format(member_value, self._MEMBERS)
            )

    def __eq__(self, other):
        raise NotImplementedError

    def __lt__(self, other):
        raise NotImplementedError

    def __le__(self, other):
        return self.__lt__(other) or self.__eq__(other)

    def __gt__(self, other):
        return not self.__le__(other)

    def __ge__(self, other):
        return not self.__lt__(other)

    def __repr__(self):
        return '_NamedOrderedSet({})'.format(self._member_value)


class Orientation(_NamedOrderedSet):
    """Orientation of part of a CenterSurroundStimulus."""

    _MEMBERS = (None, 0, 45, 90, 135, 180, 225, 270, 315)

    def __init__(self, orientation):
        if issubclass(type(orientation), Orientation):
            member_value = orientation._member_value
        else:
            member_value = orientation

        super().__init__(member_value)

    @property
    def orientation(self):
        """Orientation in degrees."""
        return self._member_value

    def __lt__(self, other):
        other_as_ori = Orientation(other)

        if (self._member_value is not None) and (
            other_as_ori.orientation is not None
        ):
            result = self._member_value < other_as_ori.orientation
        elif (self._member_value is None) and (
            other_as_ori.orientation is not None
        ):
            result = True
        else:
            result = False

        return result

    def __eq__(self, other):
        other_as_ori = Orientation(other)
        return other_as_ori.orientation == self._member_value

    def __repr__(self):
        return 'Orientation({})'.format(self._member_value)


class Contrast(_NamedOrderedSet):
    """Contrast of a CenterSurroundStimulus."""

    _MEMBERS = [0.8]

    def __init__(self, contrast):
        if issubclass(type(contrast), Contrast):
            member_value = contrast._member_value
        else:
            member_value = contrast

        super().__init__(member_value)

    def __lt__(self, other):
        other_as_contrast = Contrast(other)

        if (self._member_value is not None) and (
            other_as_contrast._member_value is not None
        ):
            result = self._member_value < other_as_contrast._member_value
        elif (self._member_value is None) and (
            other_as_contrast._member_value is not None
        ):
            result = True
        else:
            result = False

        return result

    def __eq__(self, other):
        other_as_contrast = Contrast(other)
        return other_as_contrast._member_value == self._member_value

    def __repr__(self):
        return 'Contrast({})'.format(self._member_value)


class TemporalFrequency(_NamedOrderedSet):
    """Temporal frequency of a CenterSurroundStimulus."""

    _MEMBERS = (1, 2)

    def __init__(self, temporal_frequency):
        if issubclass(type(temporal_frequency), TemporalFrequency):
            member_value = temporal_frequency._member_value
        else:
            member_value = temporal_frequency

        super().__init__(member_value)

    def __lt__(self, other):
        other_as_tf = TemporalFrequency(other)

        if (self._member_value is not None) and (
            other_as_tf._member_value is not None
        ):
            result = self._member_value < other_as_tf._member_value
        elif (self._member_value is None) and (
            other_as_tf._member_value is not None
        ):
            result = True
        else:
            result = False

        return result

    def __eq__(self, other):
        other_as_tf = TemporalFrequency(other)
        return other_as_tf._member_value == self._member_value

    def __repr__(self):
        return 'TemporalFrequency({})'.format(self._member_value)


class SpatialFrequency(_NamedOrderedSet):
    """Spatial frequency of a CenterSurroundStimulus."""

    _MEMBERS = [0.04]

    def __init__(self, spatial_frequency):
        if issubclass(type(spatial_frequency), SpatialFrequency):
            member_value = spatial_frequency._member_value
        else:
            member_value = spatial_frequency

        super().__init__(member_value)

    def __lt__(self, other):
        other_as_tf = SpatialFrequency(other)

        if (self._member_value is not None) and (
            other_as_tf._member_value is not None
        ):
            result = self._member_value < other_as_tf._member_value
        elif (self._member_value is None) and (
            other_as_tf._member_value is not None
        ):
            result = True
        else:
            result = False

        return result

    def __eq__(self, other):
        other_as_tf = SpatialFrequency(other)
        return other_as_tf._member_value == self._member_value

    def __repr__(self):
        return 'SpatialFrequency({})'.format(self._member_value)


class CenterSurroundStimulus:
    """A center-surround stimulus with possibly empty components.

    Methods
    -------
    is_empty()
        Returns True if the stimulus is completely empty.
    center_is_empty()
        Returns True if the center of the visual field is empty.
    surround_is_empty()
        Returns True if the surround part of the visual field is empty.

    Attributes
    ----------
    temporal_frequency : TemporalFrequency
    spatial_frequency : SpatialFrequency
    contrast : Contrast
    center_orientation, surround_orientation : Orientation
        Orientation of center and surround part of the visual field. Can be
        empty if this part of the visual field is omitted.

    Notes
    -----
    Please use this class instead of a DataFrame with NaN entries. NaN is not
    equal to itself, is not greater or less than other quantities, and is not
    equal to zero (coercing it to zero using np.nan_to_num could cause bugs by
    mixing empty stimuli with eg. stimulus with 0 deg orientation), whereas
    `CenterSurroundStimulus` and its attributes are always guaranteed to be
    well-defined.

    """

    def __init__(
        self,
        temporal_frequency,
        spatial_frequency,
        contrast,
        center_orientation,
        surround_orientation,
    ):
        """Initialize CenterSurroundStimulus.

        Parameters
        ----------
        temporal_frequency : int, float, or TemporalFrequency
        spatial_frequency : int, float, or SpatialFrequency
        contrast : int, float, or Contrast
        center_orientation, surround_orientation : int, float, None, NaN, or Orientation
            Orientation in degrees, or None or NaN if absent.

        Returns
        -------
        center_surround_stimulus : CenterSurroundStimulus

        Raises
        ------
        SetMembershipError
            If one of the parameters has an invalid value.

        """
        if (center_orientation is None) or np.isnan(center_orientation):
            warnings.warn(
                'Constructing a CenterSurroundStimulus with an empty center.'
            )
        self._stimulus_attributes = {
            'temporal_frequency': TemporalFrequency(temporal_frequency),
            'spatial_frequency': SpatialFrequency(spatial_frequency),
            'contrast': Contrast(contrast),
            'center_orientation': Orientation(center_orientation),
            'surround_orientation': Orientation(surround_orientation),
        }

    @property
    def temporal_frequency(self):
        return self._stimulus_attributes['temporal_frequency']

    @property
    def spatial_frequency(self):
        return self._stimulus_attributes['spatial_frequency']

    @property
    def contrast(self):
        return self._stimulus_attributes['contrast']

    @property
    def center_orientation(self):
        """Center orientation in degrees."""
        return self._stimulus_attributes['center_orientation']

    @property
    def surround_orientation(self):
        """Surround orientation in degrees."""
        return self._stimulus_attributes['surround_orientation']

    def is_empty(self):
        """Check whether the stimulus is completely empty."""
        return self == CenterSurroundStimulus(None, None, None, None, None)

    def center_is_empty(self):
        """Check whether the center of the stimulus is empty."""
        return self.center_orientation == Orientation(None)

    def surround_is_empty(self):
        """Check whether the surround portion of the stimulus is empty."""
        return self.surround_orientation == Orientation(None)

    def __repr__(self):
        return (
            f'CenterSurroundStimulus('
            f'{self.temporal_frequency._member_value}, '
            f'{self.spatial_frequency._member_value}, '
            f'{self.contrast._member_value}, '
            f'{self.center_orientation._member_value}, '
            f'{self.surround_orientation._member_value})'
        )

    def __str__(self):
        return (
            '\rCenterSurroundStimulus with attributes'
            f'\n    temporal_frequency   {str(self.temporal_frequency._member_value):>5}'
            f'\n    spatial_frequency    {str(self.spatial_frequency._member_value):>5}'
            f'\n    contrast             {str(self.contrast._member_value):>5}'
            f'\n    center_orientation   {str(self.center_orientation._member_value):>5}'
            f'\n    surround_orientation {str(self.surround_orientation._member_value):>5}\n'
        )

    def __eq__(self, other):
        """Test equality."""
        if not issubclass(type(other), CenterSurroundStimulus):
            raise TypeError(
                '`==` is not supported between types '
                '`CenterSurroundStimulus` and `{}`'.format(type(other))
            )

        # Two CenterSurroundStimulus objects are equal if all attrs are equal.
        if all(
            [
                getattr(self, name) == getattr(other, name)
                for name in self._stimulus_attributes
            ]
        ):
            result = True
        else:
            result = False

        return result
