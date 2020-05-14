#####################################################################################
"""
interpolate_nan.py of FreelyMovingEphys

Interpolates across NaNs, but only for runs less than length 'r'. Leaves NaNs at
start/end since we don't want to extrapolate. Takes option input 'method' to specify
method to pass to interp1d.

Adapted from /niell-lab-analysis/freely moving/interpNan.m

last modified: May 08, 2020
"""
#####################################################################################

from scipy.interpolate import interp1d
import numpy as np
import math

def interpolate_across_nans(x, r, method='linear'):

    # initial interpolation
    y = interp1d(np.where(math.isnan(x)), x(np.where(math.isnan(x))), kind=method)

    # find start and end of NaN runs
    bad = math.isnan(x)
    starts = np.where(np.diff(bad)>0) + 1

    # starts can either be row or columns, depending on input
    if math.isnan(x[1]):
        try:
            starts = [1, starts]
        except:
            starts = [1, np.transpose(starts)]

    ends = np.where(np.diff(bad)<0)

    # leave runs > r as bad, mark runs < r as not bad
    for i in range(1, len(ends)):
        if (ends(i) - starts(i)) < r & starts(i) != 1:
            bad[starts(i):ends(i)] = 0

    # put NaNs back in NaNs for bad interpolation
    y[bad] = math.nan

    return y