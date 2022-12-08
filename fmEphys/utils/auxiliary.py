"""
FreelyMovingEphys/src/utils/auxiliary.py
"""
import os
import sys
import numpy as np

import fmEphys

def start_log(head):
    date_str, time_str = fmEphys.fmt_now()

    log_path = os.path.join(head,
                    'errlog_{}_{}.out'.format(date_str, time_str))

    with open(log_path, 'w') as f:
        sys.stdout = f

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