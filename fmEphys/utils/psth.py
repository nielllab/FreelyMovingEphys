"""
fmEphys/utils/psth.py


Written by DMM, 2022
"""


import numpy as np
import matplotlib.pyplot as plt
import sklearn.neighbors
import sklearn.linear_model


def drop_nearby_events(x, preserve, win=0.25):
    """Drop events that fall near others.

    When eliminating compensatory eye/head movements which fall right after
    gaze-shifting eye/head movements, `x` should be the compensatory event
    times. `preserve` should be the gaze-shifting event times.

    Parameters
    ----------
    thin : np.array
        Array of timestamps (as float in units of seconds) that
        should be thinned out, removing any timestamps that fall
        within `win` seconds of timestamps in `avoid`.
    avoid : np.array
        Timestamps to avoid being near.
    win : np.array
        Time (in seconds) that times in `thin` must fall before or
        after items in `avoid` by.
    
    """

    to_drop = np.array([c for c in x for g in preserve if ((g>(c-win)) & (g<(c+win)))])

    thinned_x = np.delete(x, np.isin(x, to_drop))

    return thinned_x


def drop_repeat_events(eventT, onset=True, win=0.020):
    """Eliminate saccades repeated over sequential camera frames.

    Saccades sometimes span sequential camera frames, so that two or
    three sequential camera frames are labaled as saccade events, despite
    only being a single eye/head movement. This function keeps only a
    single frame from the sequence, either the first or last in the
    sequence.

    Parameters
    ----------
    eventT : np.array
        Array of saccade times (in seconds as float).
    onset : bool
        If True, a sequence of back-to-back frames labeled as a saccade will
        be reduced to only the first/onset frame in the sequence. If false, the
        last in the sequence will be used.
    win : float
        Distance in time (in seconds) that frames must follow each other to be
        considered repeating. Frames are 0.016 ms, so the default value, 0.020
        requires that frames directly follow one another.

    Returns
    -------
    thinned : np.array
        Array of saccade times, with repeated labels for single events removed.

    """

    duplicates = set([])
    for t in eventT:
        if onset:
            # keep first
            new = eventT[((eventT-t)<win) & ((eventT-t)>0)]
        else:
            # keep last
            new = eventT[((t-eventT)<win) & ((t-eventT)>0)]
        duplicates.update(list(new))

    thinned = np.sort(np.setdiff1d(eventT,
                                   np.array(list(duplicates)),
                                   assume_unique=True))
    
    return thinned


def calc_PSTH(spikeT, eventT, bandwidth=10, resample_size=1, edgedrop=15, win=1000):
    """Calculate PSTH for a single unit.

    The Peri-Stimulus Time Histogram (PSTH) will be calculated using Kernel
    Density Estimation by sliding a gaussian along the spike times centered
    on the event time.

    Because the gaussian filter will create artifacts at the edges (i.e. the
    start and end of the time window), it's best to add extra time to the start
    and end and then drop that time from the PSTH, leaving the final PSTH with no
    artifacts at the start and end. The time (in msec) set with `edgedrop` pads
    the start and end with some time which is dropped from the final PSTH before
    the PSTH is returned.

    Parameters
    ----------
    spikeT : np.array
        Array of spike times in seconds and with the type float. Should be 1D and be
        the spike times for a single ephys unit.
    eventT : np.array
        Array of event times (e.g. presentation of stimulus or the time of a saccade)
        in seconds and with the type float.
    bandwidth : int
        Bandwidth of KDE filter in units of milliseconds.
    resample_size : int
        Size of binning when resampling spike rate, in units of milliseconds.
    edgedrop : int
        Time to pad at the start and end, and then dropped, to eliminate edge artifacts.
    win : int
        Window in time to use in positive and negative directions. For win=1000, the
        PSTH will start -1000 ms before the event and end +1000 ms after the event.

    Returns
    -------
    psth : np.array
        Peri-Stimulus Time Histogram

    """
    
    # Unit conversions
    bandwidth = bandwidth / 1000
    resample_size = resample_size / 1000
    win = win / 1000
    edgedrop = edgedrop / 1000
    edgedrop_ind = int(edgedrop / resample_size)

    bins = np.arange(-win-edgedrop, win+edgedrop+resample_size, resample_size)

    # Timestamps of spikes (`sps`) relative to `eventT`
    sps = []
    for i, t in enumerate(eventT):
        sp = spikeT-t
        # Only keep spikes in this window
        sp = sp[(sp <= (win+edgedrop)) & (sp >= (-win-edgedrop))] 
        sps.extend(sp)
    sps = np.array(sps)

    if len(sps) < 10:
        n_bins = int((win * 1000 * 2) + 1)
        return np.zeros(n_bins)*np.nan

    kernel = sklearn.neighbors.KernelDensity(kernel='gaussian',
                                             bandwidth=bandwidth).fit(sps[:, np.newaxis])
    
    density = kernel.score_samples(bins[:, np.newaxis])

    # Multiply by the # spikes to get spike count per point. Divide
    # by # events for rate/event.
    psth = np.exp(density) * (np.size(sps ) / np.size(eventT))

    # Drop padding at start & end to eliminate edge effects.
    psth = psth[edgedrop_ind:-edgedrop_ind]

    return psth
    

def calc_hist_PSTH(cells, right, left, n_cells, label, trange, trange_x):
    """ Histogram-based PSTH calculation.

    No longer used in pipeline.
    
    """

    rightavg = np.zeros((n_cells, trange.size-1))
    leftavg = np.zeros((n_cells, trange.size-1))

    fig = plt.subplots(np.ceil(n_cells/7).astype('int'), 7,
                       figsize=(35, np.int(np.ceil(n_cells/3))),
                       dpi=50)
    
    for i, ind in enumerate(cells.index):
        
        for s in np.array(right):

            hist, _ = np.histogram(cells.at[ind,'spikeT']-s, trange)

            rightavg[i,:] = rightavg[i,:] + hist / (right.size*np.diff(trange))
        
        for s in np.array(left):

            hist, _ = np.histogram(cells.at[ind,'spikeT']-s, trange)

            leftavg[i,:] = leftavg[i,:]+ hist/(left.size*np.diff(trange))
        

        plt.subplot(np.ceil(n_cells/7).astype('int'), 7, i+1)

        plt.plot(trange_x, rightavg[i,:], color='tab:blue')
        plt.plot(trange_x, leftavg[i,:], color='tab:red')

        maxval = np.max(np.maximum(rightavg[i,:], leftavg[i,:]))

        plt.vlines(0, 0, maxval*1.5, linestyles='dotted', colors='k')
        plt.xlim([-0.5, 0.5])
        plt.ylim([0, maxval*1.2])
        plt.ylabel('sp/sec')
        plt.xlabel('sec')
        plt.title(str(ind)+' '+label)

    plt.tight_layout()

    return rightavg, leftavg, fig

