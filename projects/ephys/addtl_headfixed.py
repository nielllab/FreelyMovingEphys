import os, cv2, warnings
import pandas as pd
import numpy as np
from scipy.signal import argrelmax
import xarray as xr
from scipy.interpolate import interp1d
from tqdm import tqdm
warnings.filterwarnings('ignore')

from src.utils.path import find
from src.utils.auxiliary import flatten_series

class AddtlHF:
    def __init__(self, base_path):
        self.savepath = os.path.join(base_path,'addtlhf_props.npz')

        self.Wn_ephys = pd.read_hdf(find('*hf1_wn*_ephys_props.h5',base_path)[0])
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
        y, x = np.unravel_index(np.argmax(flat_unit_sta), unit_sta.shape)
        stim_history = vid[:,y*2,x*2]
        return stim_history, x, y

    def sort_lum(self, unit_stim, eventT, eyeT, flips, ):
        event_eyeT = np.zeros(len(eventT))
        for i, t in enumerate(eventT):
            event_eyeT[i] = eyeT[np.argmin(np.abs(t-eyeT))]
        rf_off = event_eyeT.copy(); rf_on = event_eyeT.copy(); only_global = event_eyeT.copy()
        gray = np.nanmedian(unit_stim)
        
        off_bool = unit_stim[flips+self.frameshift]<(gray-self.Sn_rf_change_thresh)
        offT = rf_off[off_bool] # light-to-dark transitions, as a timestamp in ephys eyeT timebase
        offInds = flips[np.where(off_bool)[0]]
        
        on_bool = unit_stim[flips+self.frameshift]>(gray+self.Sn_rf_change_thresh)
        onT = rf_on[on_bool] # same for dark-to-light transitions
        onInds = flips[np.where(on_bool)[0]]
        
        background_bool = (unit_stim[flips+self.frameshift]>(gray-self.Sn_rf_change_thresh)) & (unit_stim[flips+self.frameshift]<(gray+self.Sn_rf_change_thresh))
        backgroundT = only_global[background_bool] # stim did not change from baseline enoguh
        backgroundInds = flips[np.where(background_bool)[0]]
        
        return event_eyeT, offT, offInds, onT, onInds, backgroundT, backgroundInds
    
    def calc_Sn_psth(self):
        vid = self.Sn_world.WORLD_video.values.astype(np.uint8).astype(float)
        worldT = self.Sn_world.timestamps.values
        eyeT = self.Sn_ephys['Sn_eyeT'].iloc[0]
        ephysT0 = self.Sn_ephys['t0'].iloc[0]

        # when does the stimulus change?
        dStim = np.sum(np.abs(np.diff(vid, axis=0)), axis=(1,2))
        flips = np.argwhere((dStim[1:]>self.Sn_dStim_thresh) * (dStim[:-1]<self.Sn_dStim_thresh)).flatten()

        flipT = worldT[flips]
        eventT = flipT - ephysT0

        rf_xy = np.zeros([len(self.Sn_ephys.index.values),2]) # [unit#, x/y]
        Sn_psth = np.zeros([len(self.Sn_ephys.index.values), len(self.trange_x), 4]) # shape = [unit#, time, all/ltd/on/not_rf]
        for i, ind in tqdm(enumerate(self.Sn_ephys.index.values)):
            unit_sta = self.Wn_ephys.loc[ind, 'Wn_spike_triggered_average']
            unit_stim, rf_xy[i,0], rf_xy[i,1] = self.calc_RF_stim(unit_sta, vid)
            all_eventT, offT, _, onT, _, backgroundT, _ = self.sort_lum(unit_stim, eventT, eyeT, flips)
            unit_spikeT = self.Sn_ephys.loc[ind, 'spikeT']
            Sn_psth[i,:,0] = self.calc_psth(unit_spikeT, all_eventT)
            Sn_psth[i,:,1] = self.calc_psth(unit_spikeT, offT)
            Sn_psth[i,:,2] = self.calc_psth(unit_spikeT, onT)
            Sn_psth[i,:,3] = self.calc_psth(unit_spikeT, backgroundT)

        self.Sn_psth = Sn_psth
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

        Rc_psth = np.zeros([len(self.Rc_ephys.index.values), len(self.trange_x)]) # shape = [unit#, time]
        for i, ind in tqdm(enumerate(self.Rc_ephys.index.values)):
            unit_spikeT = self.Rc_ephys.loc[ind, 'spikeT']
            Rc_psth[i,:] = self.calc_psth(unit_spikeT, eventT)

        self.Rc_psth = Rc_psth

    def save(self):
        print('saving '+self.savepath)
        np.savez(self.savepath, sn=self.Sn_psth, rc=self.Rc_psth, rf=self.rf_xy)