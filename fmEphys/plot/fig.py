"""
fmEphys/plot/fig.py

Figure plotting functions.


Written by DMM, 2023
"""


import numpy as np
import matplotlib.pyplot as plt

import fmEphys as fme


def tuning_fig(bins, tuning, error, pdf=None, label=None):

    n_cells = np.size(tuning, 0)

    rows = int(np.ceil(n_cells*2))
    cols = int(np.ceil(n_cells/2))

    bin_sz = np.nanmedian(np.diff(bins))
    llim = bins[0]-(bin_sz/2)
    ulim = bins[-1]+(bin_sz/2)

    fig, axs_ = plt.subplots(rows, cols, figsize=(11,8.5), dpi=300)
    axs = axs_.flatten()

    for i in range(n_cells):

        fme.plot.tuning(bins, tuning[i,:], error[i,:])
        
        axs[i].errorbar(bins, tuning[i,:], yerr=error[i,:])

        try:
            axs[i].set_ylim(0, np.nanmax(tuning[i,:]*1.2))
        except ValueError:
            axs[i].set_ylim(0,1)

        axs[i].set_xlim([llim, ulim])
        axs[i].set_title('unit{}'.format(i))
        axs[i].set_ylabel('sp/sec')

        if label is not None:
            axs[i].set_xlabel(label)

    plt.tight_layout()

    if pdf is not None:
        pdf.savefig(); plt.close()
    elif pdf is None:
        fig.show()


def psth_plot(right, left, title_str, psth_bins):
    """ Plot PSTHs for all units.

    For PSTHs calculated with KDE (not hist based)
    """

    n_cells = np.size(right, 0)

    fig, axs = plt.subplots(np.ceil(n_cells/7).astype('int'), 7,
                            figsize=(35, np.int(np.ceil(n_cells/3))),
                            dpi=50)
    axs = axs.flatten()

    for i in range(np.size(right,0)):

        axs[i].plot(psth_bins, right[i,:],
                    color='tab:blue', label='right')
        
        axs[i].plot(psth_bins, left[i,:],
                    color='tab:red', label='left')
        maxval = np.max(np.maximum(right[i,:],
                                    left[i,:]))
        
        if (not np.isfinite(maxval)) or (maxval == 0):
            maxval = 1
            
        axs[i].vlines(0, 0, maxval*1.5, linestyles='dotted', colors='k')
        axs[i].set_xlim([-500, 500])
        axs[i].set_ylim([0, maxval*1.2])
        axs[i].set_ylabel('sp/sec')
        axs[i].set_xlabel('ms')
        axs[i].set_title(i)
        
    axs[0].legend()
    fig.suptitle(title_str)
    fig.tight_layout()

    return fig


def STA_fig(sta, pdf=None, lag=2):

    # Get lag index
    lag_range = np.arange(-2,8,2)
    li = np.argwhere(lag_range==2)
    lagname = lag*0.025*1000

    n_cells = np.size(sta,0)

    rows = int(np.ceil(n_cells*2))
    cols = int(np.ceil(n_cells/2))

    fig, axs_ = plt.subplots(rows, cols, figsize=(11,8.5), dpi=300)
    axs = axs_.flatten()
    for i in range(n_cells):

        axs[i].imshow(sta[i,li,:,:], vmin=-0.3, vmax=0.3, cmap='seismic')
        axs[i].set_title('unit{} lag{}ms'.format(i,lagname))
        axs[i].axis('off')

    plt.tight_layout()
    
    if pdf is not None:
        pdf.savefig(); plt.close()

    elif pdf is None:
        fig.show()


def plot_all_STV(stv, pdf=None, use_cmap='STA'):

    if use_cmap == 'STA':
        use_cmap == 'seismic'
    elif use_cmap == 'STV':
        use_cmap == 'cividis'

    n_cells = np.size(stv,0)

    rows = int(np.ceil(n_cells*2))
    cols = int(np.ceil(n_cells/2))

    fig, axs_ = plt.subplots(rows, cols, figsize=(11,8.5), dpi=300)
    axs = axs_.flatten()
    for i in range(n_cells):

        axs[i].imshow(stv[i,:,:], vmin=-1, vmax=1, cmap='cividis')
        axs[i].set_title('unit{}'.format(i))
        axs[i].axis('off')

    plt.tight_layout()
    
    if pdf is not None:
        pdf.savefig(); plt.close()

    elif pdf is None:
        fig.show()

