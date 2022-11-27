

import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt

import cv2
import scipy.interpolate
import sklearn.linear_model
import sklearn.neighbors

import fmEphys

def calc_kde_PSTH(spikeT, eventT, bandwidth=10, resample_size=1, edgedrop=15, win=1000):
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
    for t in eventT:
        sp = spikeT-t
        # Only keep spikes in this window
        sp = sp[(sp <= (win+edgedrop)) & (sp >= (-win-edgedrop))] 
        sps.extend(sp)
    sps = np.array(sps)

    kernel = sklearn.neighbors.KernelDensity(kernel='gaussian',
                                bandwidth=bandwidth).fit(sps[:, np.newaxis])
    density = kernel.score_samples(bins[:, np.newaxis])

    # Multiply by the # spikes to get spike count per point. Divide
    # by # events for rate/event.
    psth = np.exp(density) * (np.size(sps ) / np.size(eventT))

    # Drop padding at start & end to eliminate edge effects.
    psth = psth[edgedrop_ind:-edgedrop_ind]

    return psth


def calc_all_PSTH(spike_data, eventT, bandwidth=10, resample_size=1, edgedrop=15, win=1000):
    """
    
    spike_data should be a series of the col 'spikeT'

    This is a wrapper function for `calc_kde_PSTH` that makes it easier to
    calculate a PSTH for all cells in a dataset.
    """

    all_psth = np.empty([len(spike_data.index.values),
                            int((win*2)+1)])

    for i, ind in tqdm(enumerate(spike_data.index.values)):
        all_psth[i,:] = calc_kde_PSTH(spike_data[ind], eventT,
                                            bandwidth, resample_size,
                                            edgedrop, win).astype(object)

    return all_psth


def calc_all_tuning(spike_rates, variable, variable_range, time, model_dT=0.025):
    """
    
    time can be worldT or eyeT. For worldcma contrast, use worldT,
    for pupil radius use eyeT, etc.

    spike_rates is a series, the column 'rate' from ephys_data
    This will calculate a tuning curve for every cell in the dataset

    returns centered bins, tuning curve, and the error at each bin

    """
    # Set up a timebase that the spike rate will be in
    modelT = np.arange(0, np.max(time), model_dT)

    n_cells = len(spike_rates.index.values)

    # Empty arrays to store outputs in
    scatter = np.zeros((n_cells, len(variable)))
    tuning = np.zeros((n_cells, len(variable_range)-1))
    tuning_err = tuning.copy()
    var_cent = np.zeros(len(variable_range)-1)

    # Set up bins relative to the center point
    for j in range(len(variable_range)-1):

        var_cent[j] = 0.5 * (variable_range[j] + variable_range[j+1])

    # Calculate tuning in each bin
    for i, ind in enumerate(spike_rates.index.values):

        rateInterp = scipy.interpolate.interp1d(modelT[:-1],
                                                spike_rates[ind],
                                                bounds_error=False)
        scatter[i,:] = rateInterp(time)

        for j in range(len(variable_range)-1):
            usePts = (variable >= variable_range[j]) & (variable < variable_range[j+1])
            tuning[i, j] = np.nanmean(scatter[i, usePts])
            tuning_err[i ,j] = np.nanstd(scatter[i, usePts]) / np.sqrt(np.count_nonzero(usePts))

    return var_cent, tuning, tuning_err

def calc_all_STA(model_nsp, model_vid):
    """ Calculate multi-lag spike-triggered average

    returns array of STAs with shape:
        (cell, lag, x, y)
    """

    lag_range = np.arange(-2,8,2)

    nks = np.shape(model_vid[0,:,:])
    all_sta = np.zeros([np.size(model_nsp, 0),
                        len(lag_range),
                        np.size(model_vid, 1),
                        np.size(model_vid, 2)]) * np.nan      

    for i in tqdm(range(np.size(model_nsp, 0))):

        for l, lag in enumerate(lag_range):

            sp = model_nsp[i,:].copy()
            sp = np.roll(sp, -lag)

            sta = model_vid.T @ sp
            sta = np.reshape(sta, nks)
            nsp = np.sum(sp)

            if nsp > 0:

                sta = sta / nsp
                sta = sta - np.mean(sta)

            all_sta[i,l,:,:] = sta

    return all_sta

def calc_all_STV(model_nsp, model_vid):

    lag = 2

    nks = np.shape(model_vid[0,:,:])
    sq_model_vid = model_vid.copy()**2

    sq_mdlvid_T = np.nan_to_num(model_vid,0).T
    mean_sq_img = np.mean(sq_model_vid,0)

    all_stv = np.zeros([np.size(model_nsp, 0),
                        np.size(model_vid, 1),
                        np.size(model_vid, 2)]) * np.nan   
    
    mean_sq_img_norm = np.mean(model_vid**2, axis=0)
    
    for i in tqdm(range(np.size(model_nsp, 0))):

        sp = model_nsp[i,:].copy()
        sp = np.roll(sp, -lag)

        stv = sq_mdlvid_T @ sp
        stv = np.reshape(stv, nks)

        nsp = np.sum(sp)

        if nsp > 0:
            stv = stv / nsp
            stv = stv - mean_sq_img

        all_stv[i,:,:] = stv

    return all_stv


def drop_nearby_events(thin, avoid, win=0.25):
    """Drop events that fall near others.

    When eliminating compensatory eye/head movements which fall right after
    gaze-shifting eye/head movements, `thin` should be the compensatory event
    times.

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
    to_drop = np.array([c for c in thin for g in avoid if ((g>(c-win)) & (g<(c+win)))])
    thinned = np.delete(thin, np.isin(thin, to_drop))
    return thinned

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

    thinned = np.sort(np.setdiff1d(eventT, np.array(list(duplicates)), assume_unique=True))
    
    return thinned

def calc_saccade_times(dHead, dGaze, eyeT):
    """
    dHead is the horizontal head yaw
    dEye can be calculated from ` np.diff(theta) `
    eyeT is the eyecam timestamps

    Input arrays must all be the same length. if this is not
    the case, it will likley be because eyeT is one value longer
    than dEye and dHead. In this case, drop the last value
    of eyeT (i.e., `eyeT[:-1]`).

    These thresholds are applied directionally.

    leftward gaze shifting movements
        dHead > 60 deg/sec
        dGaze > 240 deg/sec

    rightward gaze shift movements
        dHead < -60 deg/sec
        dGaze < -240 deg/sec

    leftward compensatory movements
        dHead > 60 deg/sec
        -120 deg/sec < dGaze < 120 deg/sec

    rightward compensatory movements
        dHead < 60 deg/sec
        -120 deg/sec > dGaze > 120 deg/sec

    """

    # thresholds for eye/head movements
    shifted_head = 60 # deg/sec
    still_gaze = 120 # deg/sec
    shifted_gaze = 240 # deg/sec

    gazeL = eyeT[(dHead > shifted_head) & (dGaze > shifted_gaze)]
    gazeR = eyeT[(dHead < -shifted_head) & (dGaze < -shifted_gaze)]

    compL = eyeT[(dHead > shifted_head) & (dGaze < still_gaze) & (dGaze > -still_gaze)]
    compR = eyeT[(dHead < -shifted_head) & (dGaze > -still_gaze) & (dGaze < still_gaze)]

    # Remove compensatory movements that fall right after a gaze shifting movement
    # in the same direction
    compL = drop_nearby_events(compL, gazeL)
    compR = drop_nearby_events(compR, gazeR)

    # Events that span multiple camera frames will count towards the data more than
    # once. Here, we drop single events that repeat in the dataset as back-to-back
    # samples (i.e., camera frames). This is only keeping a single timepoint, which
    # by default is only the onset of the event (the first timestamp in the sequence).
    compL = drop_repeat_events(compL)
    compR = drop_repeat_events(compR)
    gazeL = drop_repeat_events(gazeL)
    gazeR = drop_repeat_events(gazeR)

    saccade_events = {
        'compL': compL,
        'compR': compR,
        'gazeL': gazeL,
        'gazeR': gazeR
    }

    return saccade_events
