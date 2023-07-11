"""
fmEphys/utils/correlation.py

Functions for calculating correlations.

Functions
---------
nanxcorr
    Cross correlation ignoring NaNs.


Written by DMM, 2021
"""


import numpy as np
import pandas as pd


def nanxcorr(x, y, maxlag=25):
    """ Cross correlation ignoring NaNs.

    Parameters
    ----------
    x : array
        Array of values.
    y : array
        Array of values to shift. Must be same length as x.
    maxlag : int
        Number of lags to shift y prior to testing correlation.
    
    Returns
    -------
    cc_out : array
        Cross correlation.
    lags : range
        Lag vector.

    """

    lags = range(-maxlag, maxlag)
    cc = []

    for i in range(0,len(lags)):
        
        # shift data
        yshift = np.roll(y, lags[i])
        
        # get index where values are usable in both x and yshift
        use = ~pd.isnull(x + yshift)
        
        # some restructuring
        x_arr = np.asarray(x, dtype=object)
        yshift_arr = np.asarray(yshift, dtype=object)

        x_use = x_arr[use]
        yshift_use = yshift_arr[use]
        
        # normalize
        x_use = (x_use - np.mean(x_use)) / (np.std(x_use) * len(x_use))

        yshift_use = (yshift_use - np.mean(yshift_use)) / np.std(yshift_use)
        
        # get correlation
        cc.append(np.correlate(x_use, yshift_use))

    cc_out = np.hstack(np.stack(cc))
    
    return cc_out, lags

