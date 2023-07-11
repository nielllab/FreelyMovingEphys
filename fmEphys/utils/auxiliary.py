"""
fmEphys/utils/auxiliary.py

Miscellaneous helper functions.

Functions
---------
write_dummy_cfg
    Create an empty dict with default parameters.
str_to_bool
    Parse strings to read argparse flag entries in as bool.
flatten_series
    Given a Series containing lists as objects, format as a numpy array.
find_index_in_list
    Search for a string in a list of strings and return the index.
show_xr_objs
    Return a list of columns in a DataFrame that contain xarray objects.
replace_xr_obj
    Replace xarray objects in a DataFrame with their values.
fill_NaNs
    Fill NaNs in a 1D array with linear interpolation.
drop_nan_along
    Drop NaNs in an array along an axis.
z_score
    Calculate the z-score of an array.
std_err
    Calculate the standard error of an array.
blank_col
    Create a blank column to be appended to a DataFrame.


Written by DMM, 2021
"""


import os
import yaml
import xarray as xr
import numpy as np
import pandas as pd

import fmEphys as fme


def write_dummy_cfg():
    """ Create an empty dict with default parameters.

    Returns
    --------
    dict
        Dictionary with default parameters.
    """

    # Read the pipeline_cfg.yml file saved in this repository.
    cfg_path = os.path.join(fme.up_dir(__file__, 3), 'pipeline_cfg.yml')
    
    with open(cfg_path, 'r') as infile:
        cfg_dict = yaml.load(infile, Loader=yaml.FullLoader)

    # Read the internals.yml file saved inside of this repository.
    internals_path = os.path.join(fme.up_dir(__file__, 1), 'internals.yml')
    
    with open(internals_path, 'r') as infile:
        internals_dict = yaml.load(infile, Loader=yaml.FullLoader)

    # Merge the dictionaries and return.
    return {**internals_dict, **cfg_dict}


def str_to_bool(value):
    """ Parse strings to read argparse flag entries in as bool.
    
    Parameters
    ----------
    value : str
        Input value.

    Returns
    -------
    bool
        Input value as a boolean.
    """

    if isinstance(value, bool):
        return value
    
    if value.lower() in {'False', 'false', 'f', '0', 'no', 'n'}:
        return False
    
    elif value.lower() in {'True', 'true', 't', '1', 'yes', 'y'}:
        return True
    
    raise ValueError(f'{value} is not a valid boolean value')


def flatten_series(s):
    """ Given a Series contining lists as objects, format as a numpy array.

    Parameters
    ----------
    s : pd.Series
        Series in which each index is a list as an object. Every index
        must have a list of the same length.
    
    Returns
    -------
    a : np.array
        2D array with axis 0 matching the number of indexes in the Series and
        axis 1 matching the length of the object lists in the input Series.
    """

    a = np.zeros([np.size(s,0), len(s.iloc[0])])

    count = 0
    for ind, data in s.iteritems():
        a[count,:] = data
        count += 1

    return a


def find_index_in_list(a, subset):
    """ Search for a subset of values in a list and return the index.

    Parameters
    ----------
    a : list
        List to search through
    subset : list
        List of values which may exist in a. This must be
        shorter than a.
    
    Returns
    -------
    generator
        Generator object containing the index of each instance
        of subset in a.
    """

    if not subset:
        return
    
    subset_len = len(subset)
    first_val = subset[0]

    for idx, item in enumerate(a):

        if item == first_val:

            if a[idx:idx+subset_len] == subset:

                yield tuple(range(idx, idx+subset_len))


def show_xr_objs(df):
    """ Return a list of columns in a DataFrame that contain xarray objects.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to search through.
    
    Returns
    -------
    ret : list
        List of column names that contain xarray objects.
    """

    ret = []

    for col, ser in df.iteritems():
        for i in ser.index.values:

            if type(ser.loc[i]) == xr.core.dataarray.DataArray:
                if col not in ret:
                    ret.append(col)

    return ret


def replace_xr_obj(df):
    """ Replace xarray objects in a DataFrame with their values.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to search through.

    Returns
    -------
    df : pd.DataFrame
        DataFrame with xarray objects replaced by their values as a list.
    """
    for x in show_xr_objs(df):
        for i, val in df[x].iteritems():
            if type(val) == xr.core.dataarray.DataArray:
                df.at[i,x] = val.values
    return df


def fill_NaNs(x):
    """ Fill NaNs in a 1D array with linear interpolation.

    Parameters
    ----------
    x : np.array
        1D array to fill NaNs in.

    Returns
    -------
    x : np.array
        1D array with NaNs filled.
    """

    nans = np.isnan(x)
    f = lambda z: z.nonzero()[0]
    x[nans]= np.interp(f(nans), f(~nans), x[~nans])

    return x


def drop_nan_along(x, axis=1):
    """ Drop NaNs in an array along an axis.
    
    This will remove any index along the chosen axis 

    Parameters
    ----------
    x : np.array
        Array.
    axis : int
        Axis of `x` to drop along. Default is 1, which
        will drop along columns, so that any rows (axis=0)
        containing NaNs will be dropped.
        that will have any NaNs removed.
    """

    x = x[~np.isnan(x).any(axis=axis)]

    return x


def z_score(A):
    """ Calculate z-score of an array.

    Parameters
    ----------
    A : np.array
        Array to calculate z-score of.
    
    Returns
    -------
    z : float
        Z-score of A.
    
    """

    z = (np.max(np.abs(A))-np.mean(A)) / np.std(A)

    return z


def stderr(A):
    """ Calculate standard error of an array.

    Parameters
    ----------
    A : np.array
        Array to calculate standard error of.
    
    Returns
    -------
    err : float
        Standard error of A.
    
    """

    err = np.std(A) / np.sqrt(len(A))

    return err


def blank_col(length, size):
    """ Create a blank column to be appended to a DataFrame.

    An empty series is initialized, where each index contains
    an array of values with the size specified by `size`. The
    number of indexes (i.e. cells) is specified by `length`.
    
    This can be appended to an existing DataFrame with a matching
    length and the data can be overwritten for each index (i.e.,
    cell).

    Parameters
    ----------
    length : int
        Number of indexes in the series. One for each cell.
    size : int
        Size of the array in each index (e.g., number of
        timepoints, bins of a tuning curve, etc.).

    Returns
    -------
    tmp_series : pd.Series
        Series with `length` indexes, each containing an array
        of `size` values. Will be initialized with all values
        as a NaN.
    """

    tmp_arr = np.zeros(size)*np.nan
    tmp_series = pd.Series([])

    for i in range(length):
        tmp_series.at[i] = tmp_arr.astype(object)

    return tmp_series

