

import numpy as np
import matplotlib.pyplot as plt

def jitter(c, sz, maxdist=0.2):
    """

    Args:
        c: int of center.
        sz: int of size.
        maxdist: Maximum distance that a value can be jittered
            from the center point, `c`.

    Returns:
    """
    return np.ones(sz)+np.random.uniform(c-maxdist, c+maxdist, sz)

def plot_all_tuning(bins, tuning, error, pdf=None, label=None):

    n_cells = np.size(tuning, 0)

    rows = int(np.ceil(n_cells*2))
    cols = int(np.ceil(n_cells/2))

    bin_sz = np.nanmedian(np.diff(bins))
    llim = bins[0]-(bin_sz/2)
    ulim = bins[-1]+(bin_sz/2)

    fig, axs_ = plt.subplots(rows, cols, figsize=(11,8.5), dpi=300)
    axs = axs_.flatten()

    for i in range(n_cells):
        
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

def plot_all_STA(sta, pdf=None, lag=2):

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

def plot_all_STV(stv, pdf=None):

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
