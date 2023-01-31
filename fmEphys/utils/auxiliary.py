"""
FreelyMovingEphys/src/utils/auxiliary.py
"""
import os
import sys
import yaml
import xarray as xr
import numpy as np

import fmEphys

def write_dummy_cfg():
    cfg_path = os.path.join(fmEphys.up_dir(__file__, 3), 'pipeline_cfg.yml')
    internals_path = os.path.join(fmEphys.up_dir(__file__, 1), 'internals.yml')
        
    with open(cfg_path, 'r') as infile:
        cfg_dict = yaml.load(infile, Loader=yaml.FullLoader)

    with open(internals_path, 'r') as infile:
        internals_dict = yaml.load(infile, Loader=yaml.FullLoader)

    return {**internals_dict, **cfg_dict}

def str_to_bool(value):
    """ Parse strings to read argparse flag entries in as bool.
    
    Parameters
    --------
    value : str
        Input value.

    Returns
    --------
    bool
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
    --------
    s : pd.Series
        Series in which each index is a list as an object. Every index
        must have a list of the same length.
    
    Returns
    --------
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
    """
    Parameters
    --------
    a : list
        list of values
    subset : list
        list of values shorter than a, which may exist in a
    
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

def show_xr_objs(df):
    ret = []
    for col, ser in df.iteritems():
        if type(ser.iloc[0]) == xr.core.dataarray.DataArray:
            ret.append(col)
    return ret

def replace_xr_obj(df):
    for x in show_xr_objs(df):
        for i, val in df[x].iteritems():
            df.at[i,x] = val.values
    return df

def fill_NaNs(x):

    nans = np.isnan(x)
    f = lambda z: z.nonzero()[0]
    x[nans]= np.interp(f(nans), f(~nans), x[~nans])

    return x
