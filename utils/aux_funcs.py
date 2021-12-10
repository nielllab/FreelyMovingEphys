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
    """ List paths of subdirectories in a root directory
    """
    dirnames = []
    for _, dirs, _ in os.walk(root_dir):
        for rec_dir in dirs:
            dirnames.append(rec_dir)
    return dirnames

def list_subdirs_nonrecursive(root_dir):
    """ List names of subdirectories (not paths) in a root directory, non-recursively
    """
    dirnames = []
    for _, _, filenames in os.walk(root_dir):
        for name in filenames:
            dirnames.append(name)
        break
    return dirnames

def sub2ind(array_shape, rows, cols):
    ind = rows*array_shape[1] + cols
    ind[ind < 0] = -1
    ind[ind >= array_shape[0]*array_shape[1]] = -1
    return ind

def nanmedfilt(A, sz=5):
    """ Median filtering of 1D or 2D array, A, while ignoring NaNs using tile size of sz
    adapted from https://www.mathworks.com/matlabcentral/fileexchange/41457-nanmedfilt2
    """
    if type(sz)==int:
        sz = np.array([sz,sz])
    if any(sz%2 == 0):
        print('kernel size must be odd')
    margin = np.array((sz-1)//2)
    if len(np.shape(A))==1:
        A = np.expand_dims(A, axis=0)
    AA = np.zeros(np.squeeze(np.array(np.shape(A))+2*np.expand_dims(margin,0)))
    AA[:] = np.nan
    AA[margin[0]:-margin[0], margin[1]:-margin[1]] = A
    iB, jB = np.mgrid[0:sz[0],0:sz[1]]
    isB = sub2ind(np.shape(AA.T),jB,iB)+1
    iA, jA = np.mgrid[0:np.size(A,0),0:np.size(A,1)]
    iA += 1
    isA = sub2ind(np.shape(AA.T),jA,iA)
    idx = isA + np.expand_dims(isB.flatten('F')-1,1)
    
    B = np.sort(AA.T.flatten()[idx-1],0)
    j = np.any(np.isnan(B),0)
    last = np.zeros([1,np.size(B,1)])+np.size(B,0)
    last[:,j] = np.argmax(np.isnan(B[:,j]),0)
    
    M = np.zeros([1,np.size(B,1)])
    M[:] = np.nan
    valid = np.where(last>0)[1]
    mid = (1+last)/2
    i1 = np.floor(mid[:,valid])
    i2 = np.ceil(mid[:,valid])
    i1 = sub2ind(np.shape(B.T),valid,i1)
    i2 = sub2ind(np.shape(B.T),valid,i2)
    M[:,valid] = 0.5*(B.flatten('F')[i1.astype(int)-1] + B.flatten('F')[i2.astype(int)-1])
    M = np.reshape(M, np.shape(A))
    
    return M