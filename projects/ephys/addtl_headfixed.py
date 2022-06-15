import os, cv2, warnings, sys
sys.path.insert(0, '/home/niell_lab/Documents/GitHub/FreelyMovingEphys/')
import pandas as pd
import numpy as np
from scipy.signal import argrelmax
import xarray as xr
from scipy.interpolate import interp1d
from tqdm import tqdm
warnings.filterwarnings('ignore')
from sklearn.neighbors import KernelDensity

from src.utils.path import find, list_subdirs
from src.utils.auxiliary import flatten_series

def calc_kde_sdf(spikeT, eventT, bandwidth=10, resample_size=1, edgedrop=15, win=1000, shift_half=False):
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
 
    if shift_half:
        # shift everything forward so that t=0 is centered between frame 0 and frame 1
        eventT = eventT + (1/120)

    # get timestamp of spikes relative to events in eventT
    sps = []
    for i, t in enumerate(eventT):
        sp = spikeT-t
        sp = sp[(sp <= (win+edgedrop)) & (sp >= (-win-edgedrop))] # only keep spikes in this window
        sps.extend(sp)
    sps = np.array(sps) # all values in here are between -1 and 1

    # set minimum number of spikes
    if len(sps) < 5:
        return np.zeros(2001)*np.nan, np.zeros(2001)*np.nan

    # kernel density estimation
    kernel = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(sps[:,np.newaxis])
    density = kernel.score_samples(bins[:,np.newaxis])
    sdf = np.exp(density)*(np.size(sps)/np.size(eventT)) # convert back to spike rate
    sdf = sdf[edgedrop_ind:-edgedrop_ind]
    bins = bins[edgedrop_ind:-edgedrop_ind]

    return sdf
class AddtlHF:
    def __init__(self, base_path):
        self.savepath = os.path.join(base_path,'addtlhf_props2.npz')

        # self.Wn_ephys = pd.read_hdf(find('*hf1_wn*_ephys_props.h5',base_path)[0])
        self.Sn_ephys = pd.read_hdf(find('*hf2_*_ephys_props.h5',base_path)[0])
        self.Rc_ephys = pd.read_hdf(find('*hf4_revchecker*_ephys_props.h5',base_path)[0])
        # self.FmLt_ephys = pd.open_hdf(find('*fm1*_ephys_props.h5',base_path)[0])
        self.Sn_world = xr.open_dataset(find('*hf2_*_world.nc', base_path)[0])
        self.Rc_world = xr.open_dataset(find('*hf4_revchecker*_world.nc', base_path)[0])

        self.Sn_dStim_thresh = 1e5
        self.Sn_rf_change_thresh = 30
        self.frameshift = 4

        model_dt = 0.025
        self.trange = np.arange(-1, 1.1, model_dt)
        self.trange_x = 0.5*(self.trange[0:-1]+ self.trange[1:])

    def calc_psth(self, spikeT, eventT):
        psth = np.zeros(self.trange.size-1)
        for s in np.array(eventT):
            hist, _ = np.histogram(spikeT-s, self.trange)
            psth = psth + hist / (eventT.size*np.diff(self.trange))
        return psth

    def calc_RF_stim(self, unit_sta, vid):
        flat_unit_sta = unit_sta.copy().flatten()
        on_y, on_x = np.unravel_index(np.argmax(flat_unit_sta), unit_sta.shape)
        off_y, off_x = np.unravel_index(np.argmin(flat_unit_sta), unit_sta.shape)
        on_stim_history = vid[:,on_y*2,on_x*2]
        off_stim_history = vid[:,off_y*2,off_x*2]
        return on_stim_history, (on_x, on_y), off_stim_history, (off_x, off_y)

    def sort_lum(self, unit_stim, eventT, eyeT, flips):
        event_eyeT = np.zeros(len(eventT))
        for i, t in enumerate(eventT):
            event_eyeT[i] = eyeT[np.argmin(np.abs(t-eyeT))]
        gray = np.nanmedian(unit_stim)
        
        shifted_flips = flips+self.frameshift
        if np.max(shifted_flips) > (unit_stim.size-self.frameshift):
            shifted_flips = shifted_flips[:-1]
            event_eyeT = event_eyeT[:-1]
            
        rf_off = event_eyeT.copy(); rf_on = event_eyeT.copy(); only_global = event_eyeT.copy()

        off_bool = unit_stim[shifted_flips]<(gray-self.Sn_rf_change_thresh)
        offT = rf_off[off_bool] # light-to-dark transitions, as a timestamp in ephys eyeT timebase
        offInds = flips[np.where(off_bool)[0]]
        
        on_bool = unit_stim[shifted_flips]>(gray+self.Sn_rf_change_thresh)
        onT = rf_on[on_bool] # same for dark-to-light transitions
        onInds = flips[np.where(on_bool)[0]]
        
        background_bool = (unit_stim[shifted_flips]>(gray-self.Sn_rf_change_thresh)) & (unit_stim[shifted_flips]<(gray+self.Sn_rf_change_thresh))
        backgroundT = only_global[background_bool] # stim did not change from baseline enoguh
        backgroundInds = flips[np.where(background_bool)[0]]
        
        return event_eyeT, offT, onT, backgroundT
    
    def calc_Sn_psth(self):
        vid = self.Sn_world.WORLD_video.values.astype(np.uint8).astype(float)
        worldT = self.Sn_world.timestamps.values
        eyeT = self.Sn_ephys['Sn_eyeT'].iloc[0]
        ephysT0 = self.Sn_ephys['t0'].iloc[0]

        # when does the stimulus change?
        dStim = np.sum(np.abs(np.diff(vid, axis=0)), axis=(1,2))
        flips = np.argwhere((dStim[1:]>self.Sn_dStim_thresh) * (dStim[:-1]<self.Sn_dStim_thresh)).flatten()

        eventT = worldT[flips+1] - ephysT0

        rf_xy = np.zeros([len(self.Sn_ephys.index.values),4]) # [unit#, on x, on y, off x, off y]
        on_Sn_psth = np.zeros([len(self.Sn_ephys.index.values), 2001, 4]) # shape = [unit#, time, all/ltd/on/not_rf]
        off_Sn_psth = np.zeros([len(self.Sn_ephys.index.values), 2001, 4])
        for i, ind in tqdm(enumerate(self.Sn_ephys.index.values)):
            unit_sta = self.Sn_ephys.loc[ind, 'Sn_spike_triggered_average']
            on_stim_history, on_xy, off_stim_history, off_xy = self.calc_RF_stim(unit_sta, vid)
            rf_xy[i,0] = on_xy[0]; rf_xy[i,1] = on_xy[1]
            rf_xy[i,2] = off_xy[0]; rf_xy[i,3] = off_xy[1]
            # spikes
            unit_spikeT = self.Sn_ephys.loc[ind, 'spikeT']
            if len(unit_spikeT)<10: # if a unit never fired during revchecker
                continue
            # on subunit
            all_eventT, offT, onT, backgroundT = self.sort_lum(on_stim_history, eventT, eyeT, flips)
            if len(offT)==0 or len(onT)==0:
                on_Sn_psth[i,:,:] = np.nan
                continue
            # print('all={} off={}, on={}, background={}'.format(len(all_eventT), len(offT), len(onT), len(backgroundT)))
            on_Sn_psth[i,:,0] = calc_kde_sdf(unit_spikeT, all_eventT, shift_half=True)
            on_Sn_psth[i,:,1] = calc_kde_sdf(unit_spikeT, offT, shift_half=True)
            on_Sn_psth[i,:,2] = calc_kde_sdf(unit_spikeT, onT, shift_half=True)
            on_Sn_psth[i,:,3] = calc_kde_sdf(unit_spikeT, backgroundT, shift_half=True)
            # off subunit
            all_eventT, offT, onT, backgroundT = self.sort_lum(off_stim_history, eventT, eyeT, flips)
            if len(offT)==0 or len(onT)==0:
                on_Sn_psth[i,:,:] = np.nan
                continue
            # print('all={} off={}, on={}, background={}'.format(len(all_eventT), len(offT), len(onT), len(backgroundT)))
            off_Sn_psth[i,:,0] = calc_kde_sdf(unit_spikeT, all_eventT, shift_half=True)
            off_Sn_psth[i,:,1] = calc_kde_sdf(unit_spikeT, offT, shift_half=True)
            off_Sn_psth[i,:,2] = calc_kde_sdf(unit_spikeT, onT, shift_half=True)
            off_Sn_psth[i,:,3] = calc_kde_sdf(unit_spikeT, backgroundT, shift_half=True)

        self.on_Sn_psth = on_Sn_psth
        self.off_Sn_psth = off_Sn_psth
        self.rf_xy = rf_xy

    def calc_Rc_psth(self):
        vid = self.Rc_world.WORLD_video.values.astype(np.uint8)
        worldT = self.Rc_world.timestamps.values
        eyeT = self.Rc_ephys['Rc_eyeT'].iloc[0]
        ephysT0 = self.Rc_ephys['t0'].iloc[0]

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)
        num_frames = np.size(vid, 0); vid_width = np.size(vid, 1); vid_height = np.size(vid, 2)
        kmeans_input = vid.reshape(num_frames, vid_width*vid_height)
        _, labels, _ = cv2.kmeans(kmeans_input.astype(np.float32), 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        label_diff = np.diff(np.ndarray.flatten(labels))

        stim_state = interp1d(worldT[:-1]-ephysT0, label_diff, bounds_error=False)(eyeT)
        eventT = eyeT[np.where((stim_state<-0.1)+(stim_state>0.1))]

        Rc_psth = np.zeros([len(self.Rc_ephys.index.values), 2001]) # shape = [unit#, time]
        for i, ind in tqdm(enumerate(self.Rc_ephys.index.values)):
            unit_spikeT = self.Rc_ephys.loc[ind, 'spikeT']
            if len(unit_spikeT)<10: # if a unit never fired during revchecker
                continue
            Rc_psth[i,:] = calc_kde_sdf(unit_spikeT, eventT)

        self.Rc_psth = Rc_psth

    def save(self):
        print('saving '+self.savepath)
        np.savez(self.savepath, sn_on=self.on_Sn_psth, sn_off=self.off_Sn_psth, rf=self.rf_xy, rc=self.Rc_psth)

def main():
    base_path = '/home/niell_lab/Mounts/Goeppert/nlab-nas/Dylan/freely_moving_ephys/ephys_recordings/'
    recordings = [
        # '062921/G6HCK1ALTRN',
        # '070921/J553RT'#,
        '100821/J559TT',
        '101521/J559NC',
        '101621/J559NC',
        '102621/J558NC',
        '102721/J558NC',
        '102821/J570LT',
        '110321/J558LT',
        '110421/J558LT',
        '110421/J569LT',
        '110521/J569LT',
        '122021/J581RT',
        '020222/J577TT',
        '020322/J577TT',
        '020422/J577RT',
        '020522/J577RT'
    ]
    for i, r in enumerate(recordings):
        recpath = os.path.join(base_path, r)
        recnames = list_subdirs(recpath)
        if 'hf1_wn' not in recnames:
            continue
        print(r)
        ahf = AddtlHF(recpath)
        print('sparse noise')
        ahf.calc_Sn_psth()
        # print('reversing checkerboard')
        # ahf.calc_Rc_psth()
        ahf.save()

if __name__ == '__main__':
    main()