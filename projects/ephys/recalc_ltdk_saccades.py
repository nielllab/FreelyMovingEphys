import sys, os
sys.path.insert(0, '/home/niell_lab/Documents/github/FreelyMovingEphys/')
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.stats import linregress
from tqdm import tqdm
from sklearn.neighbors import KernelDensity

from projects.ephys.population import Population
from src.utils.path import find
from src.utils.auxiliary import flatten_series

def calc_kde_sdf(spikeT, eventT, bandwidth=10, resample_size=1, edgedrop=15, win=1000):
    """
    bandwidth (in msec)
    resample_size (msec)
    edgedrop (msec to drop at the start and end of the window so eliminate artifacts of filtering)
    win = 1000msec before and after
    """
    # some conversions
    bandwidth = bandwidth/1000 # msec to sec
    resample_size = resample_size/1000 # msec to sec
    win = win/1000 # msec to sec
    edgedrop = edgedrop/1000
    edgedrop_ind = int(edgedrop/resample_size)

    # setup time bins
    bins = np.arange(-win-edgedrop, win+edgedrop+resample_size, resample_size)

    # get timestamp of spikes relative to events in eventT
    sps = []
    for i, t in enumerate(eventT):
        sp = spikeT-t
        sp = sp[(sp <= (win+edgedrop)) & (sp >= (-win-edgedrop))] # only keep spikes in this window
        sps.extend(sp)
    sps = np.array(sps) # all values in here are between -1 and 1

    # kernel density estimation
    kernel = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(sps[:,np.newaxis])
    density = kernel.score_samples(bins[:,np.newaxis])
    sdf = np.exp(density)*(np.size(sps)/np.size(eventT)) # convert back to spike rate
    sdf = sdf[edgedrop_ind:-edgedrop_ind]
    bins = bins[edgedrop_ind:-edgedrop_ind]

    return bins, sdf

def apply_win_to_comp_sacc(comp, gazeshift, win=0.25):
    bad_comp = np.array([c for c in comp for g in gazeshift if ((g>(c-win)) & (g<(c+win)))])
    comp_times = np.delete(comp, np.isin(comp, bad_comp))
    return comp_times

def main():
    # Load data
    ltdk_savepath = '/home/niell_lab/data/freely_moving_ephys/batch_files/021022/ltdk'
    ltdk = Population(savepath=ltdk_savepath)
    ltdk.load(fname='ltdk_031822')

    saccthresh = { # deg/sec
        'head_moved': 60,
        'gaze_stationary': 120,
        'gaze_moved': 240
    }

    # only use dHead at this point, don't even calculate using dEye
    for stim in ['FmLt','FmDk']:
        for s, name in enumerate(ltdk.data['session'].unique()):
            print('{} stim of {} recording'.format(stim, name))
            dHead = ltdk.data[stim+'_dHead'][ltdk.data['session']==name].iloc[0]
            dGaze = ltdk.data[stim+'_dGaze'][ltdk.data['session']==name].iloc[0]
            eyeT = ltdk.data[stim+'_eyeT'][ltdk.data['session']==name].iloc[0][:-1]
            
            gazeL = eyeT[(dHead > saccthresh['head_moved']) & (dGaze > saccthresh['gaze_moved'])]
            gazeR = eyeT[(dHead < -saccthresh['head_moved']) & (dGaze < -saccthresh['gaze_moved'])]

            compL = eyeT[(dHead > saccthresh['head_moved']) & (dGaze < saccthresh['gaze_stationary']) & (dGaze > -saccthresh['gaze_stationary'])]
            compR = eyeT[(dHead < -saccthresh['head_moved']) & (dGaze > -saccthresh['gaze_stationary']) & (dGaze < saccthresh['gaze_stationary'])]
            
            compL = apply_win_to_comp_sacc(compL, gazeL)
            compR = apply_win_to_comp_sacc(compR, gazeR)
            
            # SDFs
            for ind in tqdm(ltdk.data[ltdk.data['session']==name].index.values):
                spikeT = ltdk.data.loc[ind,stim+'_spikeT']
                
                movements = [gazeL, gazeR, compL, compR]
                movkeys = [stim+'_gazeshift_left_saccPSTH_dHead',
                        stim+'_gazeshift_right_saccPSTH_dHead',
                        stim+'_comp_left_saccPSTH_dHead',
                        stim+'_comp_right_saccPSTH_dHead']
                timekeys = [stim+'_gazeshift_left_saccTimes_dHead',
                        stim+'_gazeshift_right_saccTimes_dHead',
                        stim+'_comp_left_saccTimes_dHead',
                        stim+'_comp_right_saccTimes_dHead']
                for x in range(4):
                    movkey = movkeys[x]; timekey = timekeys[x]
                    eventT = movements[x]
                    
                    # save the spike density function
                    _, sdf = calc_kde_sdf(spikeT, eventT)
                    ltdk.data.at[ind, movkey] = sdf.astype(object)
                    
                    # save the saccade times
                    ltdk.data.at[ind, timekey] = eventT.astype(object)

    # Save new pickle file
    ltdk.save(fname='ltdk_0411822_sdf', savedir='/home/niell_lab/Desktop/')

if __name__ == '__main__':
    main()