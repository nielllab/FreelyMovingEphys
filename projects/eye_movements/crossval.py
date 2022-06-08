from sklearn.neighbors import KernelDensity
from tqdm import tqdm
import numpy as np
import pandas as pd

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

def main():

    df = pd.read_pickle('/home/niell_lab/Desktop/hffm_050922_sn_update.pickle')

    train_psth = np.zeros([len(df.index.values), 2001])
    test_psth = np.zeros([len(df.index.values), 2001])
    for i, ind in tqdm(enumerate(df.index.values)):
        if df.loc[ind, 'pref_gazeshift_direction']=='L':
            fullT = df.loc[ind, 'FmLt_gazeshift_left_saccTimes_dHead'].copy().astype(float)
        elif df.loc[ind, 'pref_gazeshift_direction']=='R':
            fullT = df.loc[ind, 'FmLt_gazeshift_right_saccTimes_dHead'].copy().astype(float)
        
        train_inds = np.random.choice(np.arange(0, fullT.size), size=int(np.floor(fullT.size/2)), replace=False)
        test_inds = np.arange(0, fullT.size)
        test_inds = np.delete(test_inds, train_inds)

        train = fullT[train_inds].copy()
        test = fullT[test_inds].copy()
        
        spikeT = df.loc[ind,'FmLt_spikeT']
        
        train_psth[i,:] = calc_kde_sdf(spikeT, train)
        test_psth[i,:] = calc_kde_sdf(spikeT, test)

    np.save('/home/niell_lab/Desktop/train_psth1.npy', train_psth)
    np.save('/home/niell_lab/Desktop/test_psth1.npy', test_psth)

if __name__ == '__main__':
    main()