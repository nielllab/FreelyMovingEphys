"""
fmEphys/utils/filter.py

Filtering functions.

Functions
---------
convfilt
    Smooth values in an array using a convolutional window.
sub2ind
    Convert subscripts to linear indices.
nanmedfilt
    Median filtering of 1D or 2D array while ignoring NaNs.
butterfilt
    Apply filter to ephys LFP along time dimension, axis=0.


Written by DMM, 2021
"""


import numpy as np
import scipy.signal


def convfilt(y, box_pts=10):
    """ Smooth values in an array using a convolutional window.

    Parameters
    ----------
    y : array
        Array to smooth.
    box_pts : int
        Window size to use for convolution.
    
    Returns
    -------
    y_smooth : array
        Smoothed y values.

    """

    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode='same')

    return y_smooth


def sub2ind(array_shape, rows, cols):
    """ Convert subscripts to linear indices.

    Equivalent to Matlab's sub2ind function
    https://www.mathworks.com/help/matlab/ref/sub2ind.html

    Parameters
    ----------
    array_shape : tuple
        Shape of the array.
    rows : np.array
        Row subscripts.
    columns : np.array
        Column subscripts.
    
    Returns
    -------
    ind : np.array
        Multidimensional subscripts.

    """

    ind = rows*array_shape[1] + cols

    ind[ind < 0] = -1

    ind[ind >= array_shape[0]*array_shape[1]] = -1

    return ind


def nanmedfilt(A, sz=5):
    """ Median filtering of 1D or 2D array while ignoring NaNs.

    Parameters
    ----------
    A : np.array
        1D or 2D array.
    sz : int
        Kernel size for median filter. Must be an odd integer.

    Returns
    -------
    M : np.array
        Array matching shape of input, A, with median filter applied.

    Notes
    -----
    Adapted from https://www.mathworks.com/matlabcentral/fileexchange/41457-nanmedfilt2

    """

    if type(sz) == int:
        sz = np.array([sz, sz])

    if any(sz % 2 == 0):
        print('kernel size must be odd')

    margin = np.array((sz-1) // 2)

    if len(np.shape(A)) == 1:
        A = np.expand_dims(A, axis=0)

    AA = np.zeros(np.squeeze(np.array(np.shape(A)) + 2*np.expand_dims(margin, 0)))
    AA[:] = np.nan
    AA[margin[0]:-margin[0], margin[1]:-margin[1]] = A

    iB, jB = np.mgrid[0:sz[0], 0:sz[1]]
    isB = sub2ind(np.shape(AA.T), jB, iB) + 1

    iA, jA = np.mgrid[0:np.size(A,0), 0:np.size(A,1)]
    iA += 1
    isA = sub2ind(np.shape(AA.T), jA, iA)

    idx = isA + np.expand_dims(isB.flatten('F')-1, 1)
    
    B = np.sort(AA.T.flatten()[idx-1], 0)
    j = np.any(np.isnan(B), 0)

    last = np.zeros([1, np.size(B,1)]) + np.size(B,0)
    last[:, j] = np.argmax(np.isnan(B[:, j]),0)
    
    M = np.zeros([1, np.size(B,1)])
    M[:] = np.nan

    valid = np.where(last > 0)[1]
    mid = (1 + last) / 2

    i1 = np.floor(mid[:, valid])
    i2 = np.ceil(mid[:, valid])
    i1 = sub2ind(np.shape(B.T), valid, i1)
    i2 = sub2ind(np.shape(B.T), valid, i2)

    M[:,valid] = 0.5*(B.flatten('F')[i1.astype(int)-1] + B.flatten('F')[i2.astype(int)-1])

    M = np.reshape(M, np.shape(A))

    return M


def butterfilt(arr, lowcut, highcut, fs, order):
    """ Apply filter to ephys LFP along time dimension, axis=0.

    Parameters
    ----------
    lfp : np.array
        Ephys LFP with shape (time, channel).
    filt : str
        Lowpass filter.
    lowcut : int
        Low end of frequency cut off.
    highcut : int
        High end of frequency cut off.
    fs : int
        Sample rate.
    order : int
        Order of filter.

    Returns
    -------
    filt: np.array
        Filtered data with shape (time, channel).

    """

    # Nyquist frequency
    nyq = 0.5 * fs

    # low cutoff
    low = lowcut / nyq

    # high cutoff
    high = highcut / nyq

    sos = scipy.signal.butter(order,
                              [low, high],
                              btype='bandpass', output='sos')
    
    filt = scipy.signal.sosfiltfilt(sos, arr, axis=0)
    
    return filt

