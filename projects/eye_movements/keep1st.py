import sys, os
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.neighbors import KernelDensity

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

    return sdf

def apply_win_to_comp_sacc(comp, gazeshift, win=0.25):
    bad_comp = np.array([c for c in comp for g in gazeshift if ((g>(c-win)) & (g<(c+win)))])
    comp_times = np.delete(comp, np.isin(comp, bad_comp))
    return comp_times

def keep_first_saccade(eventT, win=0.020):
    duplicates = set([])
    for t in eventT:
        new = eventT[((eventT-t)<win) & ((eventT-t)>0)]
        duplicates.update(list(new))
    out = np.sort(np.setdiff1d(eventT, np.array(list(duplicates)), assume_unique=True))
    return out

def main(pickle_path, save_path, stim_list):
    data = pd.read_pickle(pickle_path)

    for stim in stim_list:
        print(stim)
        for name in data['session'].unique():
            print(name)
            gazeL = data[stim+'_gazeshift_left_saccTimes_dHead'][data['session']==name].iloc[0]
            gazeR = data[stim+'_gazeshift_right_saccTimes_dHead'][data['session']==name].iloc[0]
            compL = data[stim+'_comp_left_saccTimes_dHead'][data['session']==name].iloc[0]
            compR = data[stim+'_comp_right_saccTimes_dHead'][data['session']==name].iloc[0]

            for ind in tqdm(data[data['session']==name].index.values):
                spikeT = data.loc[ind,stim+'_spikeT']

                movements = [gazeL, gazeR, compL, compR]
                movkeys = [stim+'_gazeshift_left_saccPSTH_dHead1',
                        stim+'_gazeshift_right_saccPSTH_dHead1',
                        stim+'_comp_left_saccPSTH_dHead1',
                        stim+'_comp_right_saccPSTH_dHead1']
                timekeys = [stim+'_gazeshift_left_saccTimes_dHead1',
                        stim+'_gazeshift_right_saccTimes_dHead1',
                        stim+'_comp_left_saccTimes_dHead1',
                        stim+'_comp_right_saccTimes_dHead1']
                for x in range(4):
                    movkey = movkeys[x]
                    timekey = timekeys[x]
                    eventT = keep_first_saccade(movements[x])

                    sdf = calc_kde_sdf(spikeT, eventT)
                    data.at[ind, movkey] = sdf.astype(object)

                    data.at[ind, timekey] = eventT.astype(object)

    data.to_pickle(save_path)

if __name__ == '__main__':
    pickle_path = 'E:/Dylan/ltdk_051822.pickle'
    save_path = 'E:/Dylan/ltdk_051822_keep1st.pickle'
    stim_list = ['FmLt', 'FmDk']

    main(pickle_path, save_path, stim_list)