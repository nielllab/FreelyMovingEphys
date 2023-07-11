"""
fmEphys/plot/axs.py

Functions for plotting on axes.


Written by DMM, 2023
"""


import numpy as np
import matplotlib.pyplot as plt

import fmEphys as fme


def tuning(ax, bins, tuning, error,
              label=None, unum=None, ylim=None):
    """ Plot the tuning curve for one cell.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to plot on.
    bins : np.array
        The bin centers for the tuning curve.
    tuning : np.array
        The tuning curve values. Should be 1D and have the same
        length as `bins`.
    error : np.array
        Error bars for the tuning curve. Should be 1D and have
        the same length as `bins`.
    label : str, optional
        The label for the x-axis.
    ylim : float, optional
        The y-axis limit. If None, the limit is set to 1.2 times
        the maximum value of `tuning`.
    
    Example use
    -----------
    fme.plot.tuning_ax(axs[i], bins, tuning[i,:], error[i,:])

    """

    ax.errorbar(bins, tuning, yerr=error)

    if ylim is None:
        try:
            ax.set_ylim(0, np.nanmax(tuning*1.2))
        except ValueError:
            ax.set_ylim(0,1)
    elif ylim is not None:
        ax.set_ylim([0,ylim])

    # Set x-axis limits
    bin_sz = np.nanmedian(np.diff(bins))
    llim = bins[0]-(bin_sz/2)
    ulim = bins[-1]+(bin_sz/2)

    ax.set_xlim([llim, ulim])
    ax.set_title('unit{}'.format(unum))
    ax.set_ylabel('sp/sec')

    if label is not None:
        ax.set_xlabel(label)


def cat_scatter(ax, df, prop, cat=None, cats=None,
                colors=None, use_median=False):
    """ Categorical scatter plot.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to plot on.

    
    """


    if cats is None and cat is None:

        cats = ['early','late','biphasic','negative']
        cat = 'gazecluster'

        _props = fme.props()
        colors = _props['colors']

    for c_i, c in enumerate(cats.items()):

        cdata = df[prop][df[cat]==c]

        x_jitter = fme.plot.jitter(c_i, np.size(cdata,0))

        ax.plot(x_jitter, cdata,
                '.', color=colors[c], markersize=2)
        
        # Either use median or mean of the data
        if use_median:
            hline = np.nanmedian(cdata)
        elif not use_median:
            hline = np.nanmean(cdata)
        
        ax.hlines(hline, c-0.2, c+0.2,
                  color='k', linewidth=2)

        err = fme.stderr(cdata)
        
        ax.vlines(c, hline-err, hline+err,
                  color='k', linewidth=2)
        
    ax.set_xticks(range(len(cats)), cats)


def PSTH_heatmap(ax, tseq,
                cscale=0.75):
    """
    with shape (n_cells, n_timepoints)
    
    """

    ax.set_xlabel('time (msec)')
    ax.set_ylim('cells')
    ax.set_ylim([np.size(tseq,0), 0])

    img = ax.imshow(tseq, cmap='coolwarm', vmin=-cscale,
                    vmax=cscale)
    
    if np.size(tseq,1)==2001:
        psth_bins = np.linspace(-1., 1., 1/1000)
        winStart = 800 # 1000-200
        winEnd = 1400 # 1000+400
        cent = 1000

    elif np.size(tseq,1)==3001:
        psth_bins = np.linspace(-1.5, 1.5, 1/1000)
        winStart = 1300 # 1500-200
        winEnd = 1900 # 1500+400
        cent = 1500

    ax.set_xlim([winStart,winEnd])

    ax.set_xticks(np.linspace(winStart, winEnd, 4),
                  labels=np.linspace(-200, 400, 4).astype(int))
    
    ax.vlines(cent, 0, np.size(tseq,0), color='k',
              linestyle='dashed', linewidth=1)

    return img

