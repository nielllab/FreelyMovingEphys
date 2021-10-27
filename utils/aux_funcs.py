"""
aux_funcs.py
"""
import pandas as pd
import numpy as np
import os, fnmatch

def nanxcorr(x, y, maxlag=25):
    """ Cross correlation ignoring NaNs.

    Parameters:
    x (np.array): array of values
    y (np.array): array of values to shift, must be same length as x
    maxlag (int): number of lags to shift y prior to testing correlation (default 25)
    
    Returns:
    cc_out (np.array): cross correlation
    lags (range): lag vector
    """
    lags = range(-maxlag, maxlag)
    cc = []
    for i in range(0,len(lags)):
        # shift data
        yshift = np.roll(y, lags[i])
        # get index where values are usable in both x and yshift
        use = ~pd.isnull(x + yshift)
        # some restructuring
        x_arr = np.asarray(x, dtype=object); yshift_arr = np.asarray(yshift, dtype=object)
        x_use = x_arr[use]; yshift_use = yshift_arr[use]
        # normalize
        x_use = (x_use - np.mean(x_use)) / (np.std(x_use) * len(x_use))
        yshift_use = (yshift_use - np.mean(yshift_use)) / np.std(yshift_use)
        # get correlation
        cc.append(np.correlate(x_use, yshift_use))
    cc_out = np.hstack(np.stack(cc))
    
    return cc_out, lags

def smooth_convolve(y, box_pts=10):
    """ Smooth values in an array using a convolutional window.

    Parameters:
    y (np.array): array to smooth
    box_pts (int): window size to use for convolution
    
    Returns
    y_smooth (np.array): smoothed y values
    """
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def str_to_bool(value):
    """ Parse strings to read argparse flag entries in as bool.
    
    Parameters:
    value (str): input value
    
    Returns:
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
    a = np.zeros([np.size(s,0), len(s.iloc[0])])
    count = 0
    for ind, data in s.iteritems():
        a[count,:] = data
        count += 1
    return a

def find(pattern, path):
    """ Glob for subdirectories

    Parameters:
    pattern (str): str with * for missing sections of characters
    path (str): path to search, including subdirectories
    
    Returns:
    result (list): list of files matching pattern
    """
    result = [] # initialize the list as empty
    for root, _, files in os.walk(path): # walk though the path directory, and files
        for name in files:  # walk to the file in the directory
            if fnmatch.fnmatch(name,pattern):  # if the file matches the filetype append to list
                result.append(os.path.join(root,name))
    return result # return full list of file of a given type

def check_path(basepath, path):
    """ Check if path exists, if not then create directory
    """
    if path in basepath:
        return basepath
    elif not os.path.exists(os.path.join(basepath, path)):
        os.makedirs(os.path.join(basepath, path))
        print('Added Directory:'+ os.path.join(basepath, path))
        return os.path.join(basepath, path)
    else:
        return os.path.join(basepath, path)

def list_subdirs(root_dir):
    """ List subdirectories in a root directory
    """
    dirnames = []
    for _, dirs, _ in os.walk(root_dir):
        for rec_dir in dirs:
            dirnames.append(rec_dir)
    return dirnames