"""
fmEphys/utils/plot_helpers.py

Figure plotting helper functions.

Functions
---------
props
    Return a dictionary of plotting properties.


Written by DMM, 2022
"""


import numpy as np
import matplotlib.pyplot as plt


def props():
    """ Return a dictionary of plotting properties.
    
    Returns
    -------
    props : dict
        A dictionary of plotting properties.
    
    Notes
    -----
    Keys in the `props` dict are...
    colors : dict
        A dictionary of colors.
    psth_bins : np.array
        An array of time bins for PSTHs.
    psth_bins_long : np.array
        An array of time bins for long PSTHs
        from -1500 ms to +1500 ms.

    """

    plasma_map = plt.cm.plasma(np.linspace(0,1,15))
    colors = {
        'early': plasma_map[10,:],
        'late': plasma_map[8,:],
        'biphasic': plasma_map[5,:],
        'negative': plasma_map[2,:],
        'unresponsive': 'dimgrey',
        'gaze': 'firebrick',
        'comp': 'mediumblue',
        'rc': 'indigo',
    }
    psth_bins = np.arange(-1,1.001,1/1000)
    psth_bins_long = np.arange(-1.5,1.501,1/1000)

    props = {
        'colors': colors,
        'psth_bins': psth_bins,
        'psth_bins_long': psth_bins_long
    }

    return props


def jitter(c, sz, maxdist=0.2):
    """ Jitter x values.

    This is useful for scatter plots of categorical data,
    when the distribution along y is more clear when the
    x position of that category is jittered slightly.

    Parameters
    ----------
    c : int or float
        Center position of the data.
    sz : int
        Number of data points.
    maxdist : float
        Maximum distance that a value can be jittered
        from their center point, `c`.

    Returns
    -------
    j_x : np.array
        Jittered x values in a 1D array of length `sz`.

    """

    j_x = np.ones(sz) + np.random.uniform(c-maxdist, c+maxdist, sz)

    return j_x

