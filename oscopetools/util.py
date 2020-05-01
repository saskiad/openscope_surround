"""Utility functions, mainly for stim_tables.

Should probably put these all in a StimTable class eventually.

"""
#%% IMPORT MODULES

import os
import sys
import warnings
import copy

import numpy as np
import pandas as pd


#%% MISC

class gagProcess(object):
    """Class to forcibly gag verbose methods.

    Temporarily redirects stdout to block print commands.

    >>> with gagProcess:
    >>>     print 'Things that will not be printed.'

    """

    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._original_stdout


#%% UTILITIES FOR STIM_TABLES

@np.vectorize
def nanequal(a, b):
    """Return true if a==b or a and b are both nan.

    Works only for numeric types.

    """
    if a == b or np.isclose(a, b) or (np.isnan(a) and np.isnan(b)):
        return True
    else:
        return False


def nanuniquerows(data, columns):
    """Find unique rows of data based on specified columns.

    Differs from np.unique(data, axis = 0) in that NaNs are considered equal.

    Inputs:
        data (pd.DateFrame-like)
        columns (list of strings)

    Returns:
        Dict with inds of sample unique rows and vals of columns in those rows.

    """
    uniquerows = {'inds': [], 'vals': []}
    for i in range(data.shape[0]):
        i_isunique = True
        for j in range(i + 1, data.shape[0]):
            if all(nanequal(data.loc[i, columns], data.loc[j, columns])):
                i_isunique = False
                break
            else:
                continue
        if i_isunique:
            uniquerows['inds'].append(i)
            uniquerows['vals'].append(data.loc[i, columns])

    uniquerows['vals'] = pd.DataFrame(uniquerows['vals'])
    return uniquerows


def content_rowmask(data, **filter_conds):
    """Find rows of data matching all filter conditions.

    Inputs:
        data (pd.DataFrame-like)
            -- DataFrame from which to generate a mask based on contents.
        filter_conds (arbitrary types)
            -- Conditions on which to filter. For example, to find rows where
            column 'colname' of data has the value 'somevalue', pass
            'colname = somevalue' as an argument.

    Returns:
        Boolean vector that is True for rows of data when all filter_conds are
        True.

    """
    row_mask = np.ones(data.shape[0], dtype = np.bool)
    for key, val in filter_conds.items():
        row_mask = np.logical_and(row_mask, nanequal(data[key], val))
    return row_mask


def populate_columns(data, Mean_Gray = True, No_Surround = True, Ortho = True, inplace = False):
    if not inplace:
        data = copy.deepcopy(data)

    if Mean_Gray:
        data['Mean_Gray'] = np.logical_and(
            np.isnan(data['Surround_Ori']),
            np.isnan(data['Center_Ori'])
        )
    if No_Surround:
        data['No_Surround'] = np.logical_and(
            np.isnan(data['Surround_Ori']),
            ~np.isnan(data['Center_Ori'])
        )
    if Ortho:
        data['Ortho'] = np.logical_and(
            np.logical_and(~data['Mean_Gray'], ~data['No_Surround']),
            np.isclose(
                np.abs(data['Center_Ori'] - data['Surround_Ori']) % 180.,
                90.
            )
        )

    return data


#%% UTILITIES FOR FINDING PATHS

def find_by_suffix(directory, suffix, return_full = False):
    """Search for fnames with a matching suffix, returning first match.

    Inputs:
        directory (str)
        suffix (str)
        return_full (bool)
            -- Return rel/path/to/directory/match if True.

    Returns:
        Matched name in directory or full path to match, depending on value of
        return_full.

    """
    matches = []
    for name in os.listdir(directory):
        if name.endswith(suffix):
            matches.append(name)

    if len(matches) > 1:
        warnings.warn('More than one match. Returning first one.')

    if len(matches) > 0:
        if return_full:
            return os.path.join(directory, matches[0])
        else:
            return matches[0]
    else:
        return None


def find_by_prefix(directory, prefix, return_full = False):
    """Search for fnames with a matching prefix, returning first match.

    Inputs:
        directory (str)
        prefix (str)
        return_full (bool)
            -- Return rel/path/to/directory/match if True.

    Returns:
        Matched name in directory or full path to match, depending on value of
        return_full.

    """
    matches = []
    for name in os.listdir(directory):
        if name.startswith(prefix):
            matches.append(name)

    if len(matches) > 1:
        warnings.warn('More than one match. Returning first one.')

    if len(matches) > 0:
        if return_full:
            return os.path.join(directory, matches[0])
        else:
            return matches[0]
    else:
        return None


def find_anywhere(directory, pattern, return_full = False):
    """Search for fnames containing pattern, returning first match.

    Inputs:
        directory (str)
        pattern (str)
        return_full (bool)
            -- Return rel/path/to/directory/match if True.

    Returns:
        Matched name in directory or full path to match, depending on value of
        return_full.

    """
    matches = []
    for name in os.listdir(directory):
        if pattern in name:
            matches.append(name)

    if len(matches) > 1:
        warnings.warn('More than one match. Returning first one.')

    if len(matches) > 0:
        if return_full:
            return os.path.join(directory, matches[0])
        else:
            return matches[0]
    else:
        return None
