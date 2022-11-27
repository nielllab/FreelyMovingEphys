"""Basic str and array opreations, logging errors.
"""
import pandas as pd
import itertools
import numpy as np
import xarray as xr
from datetime import datetime

class Log:
    def __init__(self, f, name="", PRINT=True, retrain=False):
        text = ""
        if type(name) == list:
            text = "{}".format(name[0])
            for x in name[1:]:
                text += ",{}".format(x)
        elif type(text) == str:
            text = name
        self.FNAME = f
        if retrain:
            F = open(self.FNAME,"a")
        else:
            F = open(self.FNAME,"w+")
        if len(text) != 0:
            F.write(text + "\n")
        F.close()
        if PRINT:
            print(text)

    def log(self, data, PRINT=True):
        text = ""
        if type(data) == list:
            text = "{}".format(data[0])
            for x in data[1:]:
                text += ",{}".format(x) 
        elif type(data) == str:
            text = data
        if PRINT:
            print(text)
        F = open(self.FNAME,"a")
        F.write(str(text) + "\n")
        F.close()

def flatten_dict(ndict):
    """
    nested dict to flat dict
    https://stackoverflow.com/questions/71460721/best-way-to-get-nested-dictionary-items
    """
    def key_value_pairs(d, key=[]):
        if not isinstance(d, dict):
            yield tuple(key), d
        else:
            for level, d_sub in d.items():
                key.append(level)
                yield from key_value_pairs(d_sub, key)
                key.pop()
    flat = dict(key_value_pairs(ndict))
    return flat

def nest_dict(d):
    """
    return a nested version of the existin dict
    https://stackoverflow.com/questions/62709639/how-to-convert-dict-in-which-key-is-in-tuple-to-nested-dictionary
    
    if self.d is currently
    {('a', 'b', 'c'): 7}
    
    this will return
    {'a': {
        'b': {
            'c': 7
        }
    }}
    
    """
    n = {}
    for k in d.keys():
        n[k[0]] = {k[1]: {k[2]: d[k]}}
    return n

def calc_list_index(a, subset):
    """Find the indexes of values in a list.

    Parameters
    --------
    a : list
        Values
    subset : list
        Values that may appear in `a`. This should have an equal or
        smaller number of values than `a`. It is okay for values in
        `subset` to not appear in `a` (these will not be given an index).
    
    Returns
    --------
    (idx, )
    """
    if not subset:
        return
    subset_len = len(subset)
    first_val = subset[0]
    for idx, item in enumerate(a):
        if item == first_val:
            if a[idx:idx+subset_len] == subset:
                yield tuple(range(idx, idx+subset_len))

def stderr(A):
    """Standard error.
    """
    return np.std(A) / np.sqrt(np.size(A))

def z_score(A):
    """Z-score.
    """
    return (np.max(np.abs(A))-np.mean(A)) / np.std(A)

def modind(a, b):
    """Modulation index.

    A positive modulation index indicates a preference
    for `a` over `b`. If one of the inputs is by definition
    preferred, it should be used as `a` and the non-preferred
    value should be `b`.
    """
    modind_val = (a - b) / (a + b)
    return modind_val

def nearest_ind(val, a):
    """Approximate the position of a value in an array.

    This is useful if `a` is an array of timestamps and you want the
    frame closest to the time `val`.
    
    Parameters
    ----------
    val : float
        Single value which falls somewhere between minimum and maximum
        values in `a`.
    a : np.array
        Array of values.
    
    Returns
    -------
    ind : int
        Position of a value in `a` which is the closest to `val` of all
        values in `a`.

    """
    ind = np.argmin(np.abs(a - val))
    return ind

def str_to_bool(value):
    """Parse str as bool type.

    Can be helpful when turning an argparse input into a bool.
    """
    if isinstance(value, bool):
        return value
    if value.lower() in {'False', 'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'True', 'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')

def probe_to_ch(probe):
    """Get channel info from probe name.

    Parameters
    ----------
    probe : str
        Probe name. This must contain the number of channels/sites in the
        probe as a str (e.g. 64, 128).
    
    Returns
    -------
    numCh : int
        Number of channels/sites.
    chSpacing : int
        Vertical distance between sites in microns.
    """
    if '16' in probe:
        return 16, 25
    if '64' in probe:
        if probe=='DB_P64-8':
            return 64, 25/2
        else:
            return 64, 25
    if '128' in probe:
        return 128, 25

def get_all_cap_combs(s):
    """Get every unique pattern of character capitalization.

    Example
    -------
    For s='word', this would return [word, Word, wOrd, woRd, ... WORD] each
    as a string.
    
    """
    return map(''.join, itertools.product(*zip(s.upper(), s.lower())))

def drop_NaNs(x, axis=1):
    """Drop all NaNs along one axis of an array.

    Parameters
    ----------
    x : np.array
        Array
    axis : int
        Axis to drop NaNs along.

    Returns
    x : np.array
        Same as input `x` but with all NaNs removed.
    
    """
    # axis=1 will drop along columns (i.e. any rows with NaNs will be dropped)
    x = x[~np.isnan(x).any(axis=axis)]
    return x

def add_jitter(center, size, scale=0.2):
    """Jitter values around a center point.
    
    Parameters
    ----------
    center : int or float
        Values
    size : int
        Number of points to create which are scattered around `center`
    scale : float
        Maximum distance that points can jitter relative to `center` in
        positive or negative direction.

    Returns
    -------
    jittered : np.array
        Jittered values
    """
    jittered = np.ones(size) + np.random.uniform(center-scale, center+scale, size)
    return jittered

def series_to_arr(s):
    """Collapse Series of lists to 2D array.

    This is for a pd.Series where each index contains a list of values. The list
    should be stored in each index as an object. The list in each index must be
    of equal length. This function will return a 2D array with a shape determined
    by the length of the Series and the length shared by all lists in each index.

    Parameters
    ----------
    s : pd.Series
        Series in which each index is a list as an object. Every index
        must have a list of the same length.

    Returns
    -------
    a : np.array
        2D array with axis 0 matching the number of indexes in the Series and
        axis 1 matching the length of the lists in the input Series.
    """
    # Only check the length of the list in index 0...
    # If the list lendths are not consistant, this will error
    a = np.zeros([np.size(s,0), len(s.iloc[0])])
    for i, vals in enumerate(s):
        a[i,:] = vals
    return a

def series_to_list(s, flat=False):
    """Collapse Series of lists to a list of lists
    This is necessary (instead of to a numpy array) if they can all have different values
    """
    out = []
    for i, vals in enumerate(s):
        if not flat:
            out.append(out)
        else:
            out.extend(vals)
    return out
    
def merge_uneven_xr(objs, dim_name='frame'):
    """Merge DataArrays of unequal lengths.

    For two or more xarray DataArrays with the same named dimension which
    is of nearly the same length. This will compare their lengths and
    shorten the length to match the shortest dimention length of the
    DataArrays that were compared.

    Do not use this if the lengths are off by significant size. This is
    best used for cases when the length is rarely different and in those
    cases is different by a small amount.

    Once the lengths are corrected, they will be merged into a
    single xr.Dataset

    Parameters
    ----------
    objs : list
        List of xr.DataArray
    dim_name : str
        Name of the DataArray dimension along which lengths should
        be compared and shortened.

    Returns
    -------
    mergered_objs : xr.Dataset
        All data mergered together into a single Dataset.
    """
    # Check lengths
    max_lens = []
    for obj in objs:
        max_lens.append(dict(obj.frame.sizes)[dim_name])
    
    # Use the smallest
    set_len = np.min(max_lens)

    # Shorten everything to `set_len`
    even_objs = []
    for obj in objs:
        obj_len = dict(obj.frame.sizes)[dim_name]

        if obj_len > set_len: # if this one is too long

            # Find how much it needs to be shortened by
            diff = obj_len - set_len
            good_inds = range(0, obj_len-diff)

            # Set the new end
            obj = obj.sel(frame=good_inds)

            even_objs.append(obj)

        # If it is the smallest length or all objects have the same length,
        # just append it to the list of objects to merge
        else:
            even_objs.append(obj)

    # Merge the xr which now have equal lengths
    merged_objs = xr.merge(even_objs)

    return merged_objs

def split_xyl(pts):

    x_pos = pd.Series([])
    y_pos = pd.Series([])
    likeli = pd.Series([])

    for col in pts.columns.values:
        if '_x' in col:
            x_pos = pd.concat([x_pos, pts[col]], axis=1)
        elif '_y' in col:
            y_pos = pd.concat([y_pos, pts[col]], axis=1)
        elif 'likeli' in col:
            likeli = pd.concat([likeli, pts[col]], axis=1)

    x_pos.drop(columns=[0], inplace=True)
    y_pos.drop(columns=[0], inplace=True)
    likeli.drop(columns=[0], inplace=True)
    
    return x_pos, y_pos, likeli

def empty_obj_col(n_cells, sz):
    """
    df is the dataframe
    sz is the length of values for each 
    it is okay for the values to be different sizes for differnet cells
    add an empty column to a pandas dataframe, for columns that will store arrays as an object

    after running this, you can add the new column as
        df['NewColumnName'] = empty_obj_col(115, 2001)
    for a recording with 115 cells and where the array of data for each cell
    has the length 2001 (as is the case for KDE PSTHs)
    """

    empty_arr = np.zeros(sz)*np.nan

    # Create an empty array of NaNs with 
    _sdata = np.zeros(n_cells)*np.nan
    empty_series = pd.Series(_sdata.astype(object))

    for i in range(empty_series.index.values):
        empty_series[i] = empty_arr.copy().astype(object)

    return empty_series


