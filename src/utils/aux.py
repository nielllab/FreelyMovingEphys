"""
FreelyMovingEphys/core/utils/aux.py
"""
import numpy as np

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