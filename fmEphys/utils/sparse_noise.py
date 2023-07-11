"""
fmEphys/utils/sparse_noise.py


Written by DMM, 2021
"""


import os
import cv2
import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import scipy.interpolate
import scipy.signal
import sklearn.cluster

import fmEphys as fme


class HeadFixedSparseNoise(fme.Ephys):


    def __init__(self, cfg, recording_name, recording_path):

        fme.Ephys.__init__(self, cfg, recording_name, recording_path)

        self.fm = False
        self.stim = 'sn'

        self.Sn_dStim_thresh = 1e5
        self.Sn_rf_change_thresh = 30
        self.frameshift = 4


    def calc_RF_stim(self, unit_sta, vid):

        flat_unit_sta = unit_sta.copy().flatten()

        on_y, on_x = np.unravel_index(np.argmax(flat_unit_sta), unit_sta.shape)
        off_y, off_x = np.unravel_index(np.argmin(flat_unit_sta), unit_sta.shape)

        on_stim_history = vid[:, on_y*2, on_x*2]
        off_stim_history = vid[:, off_y*2, off_x*2]
        
        return on_stim_history, (on_x, on_y), off_stim_history, (off_x, off_y)


    def sort_lum(self, unit_stim, eventT, eyeT, flips):

        event_eyeT = np.zeros(len(eventT))

        for i, t in enumerate(eventT):
            event_eyeT[i] = eyeT[np.nanargmin(np.abs(t-eyeT))]

        gray = np.nanmedian(unit_stim)
        
        shifted_flips = flips+self.frameshift

        if np.max(shifted_flips) > (unit_stim.size-self.frameshift):

            shifted_flips = shifted_flips[:-1]
            event_eyeT = event_eyeT[:-1]
            
        rf_off = event_eyeT.copy()
        rf_on = event_eyeT.copy()
        only_global = event_eyeT.copy()

        off_bool = unit_stim[shifted_flips]<(gray-self.Sn_rf_change_thresh)
        offT = rf_off[off_bool] # light-to-dark transitions, as a timestamp in ephys eyeT timebase
        # offInds = flips[np.where(off_bool)[0]]
        
        on_bool = unit_stim[shifted_flips]>(gray+self.Sn_rf_change_thresh)
        onT = rf_on[on_bool] # same for dark-to-light transitions
        # onInds = flips[np.where(on_bool)[0]]
        
        background_bool = (unit_stim[shifted_flips]>(gray-self.Sn_rf_change_thresh)) &                 \
                            (unit_stim[shifted_flips]<(gray+self.Sn_rf_change_thresh))
        backgroundT = only_global[background_bool] # stim did not change from baseline enoguh
        # backgroundInds = flips[np.where(background_bool)[0]]
        
        return event_eyeT, offT, onT, backgroundT


    def calc_Sn_psth(self):

        _offset_time=(1/120)

        # Read it in again, since the origional will not still be held in memory
        self.Sn_world = xr.open_dataset(self.world_path)
        vid = self.Sn_world.WORLD_video.values.astype(np.uint8).astype(float)

        self.unit_stim_eventT = {}

        # when does the stimulus change?
        dStim = np.sum(np.abs(np.diff(vid, axis=0)), axis=(1,2))

        flips = np.argwhere((dStim[1:]>self.Sn_dStim_thresh) *              \
                            (dStim[:-1]<self.Sn_dStim_thresh)).flatten()

        # interpolate if there are any NaNs in the timestamps
        _worldT = fme.fill_NaNs(self.worldT.values)
        _eyeT = fme.fill_NaNs(self.eyeT)

        eventT = _worldT[flips+1] # - self.ephysT0

        # [unit#, on x, on y, off x, off y]
        rf_xy = np.zeros([len(self.cells.index.values),4])
        
        # shape = [unit#, time, all/ltd/on/not_rf]
        on_Sn_psth = np.zeros([len(self.cells.index.values), 2001, 4])
        off_Sn_psth = np.zeros([len(self.cells.index.values), 2001, 4])
        
        for cell_i, ind in tqdm(enumerate(self.cells.index.values)):

            unit_sta = self.sta[cell_i]

            on_stim_history, on_xy, off_stim_history, off_xy = self.calc_RF_stim(unit_sta, vid)

            rf_xy[cell_i,0] = on_xy[0]
            rf_xy[cell_i,1] = on_xy[1]
            rf_xy[cell_i,2] = off_xy[0]
            rf_xy[cell_i,3] = off_xy[1]

            # Spikes
            _event_names = ['allT', 'darkT', 'lightT', 'bckgndT']

            unit_spikeT = self.cells.loc[ind, 'spikeT']

            # If a unit never fired during revchecker
            if len(unit_spikeT)<10:

                on_Sn_psth[cell_i,:,:] = np.empty([2001,4])*np.nan
                off_Sn_psth[cell_i,:,:] = np.empty([2001,4])*np.nan

                continue

            # On subunit
            all_eventT, offT, onT, backgroundT = self.sort_lum(on_stim_history, eventT, _eyeT, flips)

            # if len(offT)==0 or len(onT)==0:
            #     on_Sn_psth[cell_i,:,:] = np.empty([2001,4])*np.nan
            #     continue

            unit_stim_eventT = {}

            for i, n in enumerate(_event_names):

                unit_stim_eventT['onSubunit_eventT_{}'.format(n)] = [all_eventT,
                                                                     offT,
                                                                     onT,
                                                                     backgroundT][i] + _offset_time

            # print('all={} off={}, on={}, background={}'.format(len(all_eventT), len(offT),
            #           len(onT), len(backgroundT)))

            on_Sn_psth[cell_i,:,0] = self.calc_kde_PSTH(unit_spikeT, all_eventT+_offset_time)
            on_Sn_psth[cell_i,:,1] = self.calc_kde_PSTH(unit_spikeT, offT+_offset_time)
            on_Sn_psth[cell_i,:,2] = self.calc_kde_PSTH(unit_spikeT, onT+_offset_time)
            on_Sn_psth[cell_i,:,3] = self.calc_kde_PSTH(unit_spikeT, backgroundT+_offset_time)
            
            # off subunit
            all_eventT, offT, onT, backgroundT = self.sort_lum(off_stim_history, eventT,_eyeT, flips)

            # if len(offT)==0 or len(onT)==0:
            #     off_Sn_psth[cell_i,:,:] = np.empty([2001,4])*np.nan
            #     continue

            for i, n in enumerate(_event_names):
                unit_stim_eventT['offSubunit_eventT_{}'.format(n)] = [all_eventT,
                                                                      offT,
                                                                      onT,
                                                                      backgroundT][i] + _offset_time

            # print('all={} off={}, on={}, background={}'.format(len(all_eventT),
            #               len(offT), len(onT), len(backgroundT)))
            off_Sn_psth[cell_i,:,0] = self.calc_kde_PSTH(unit_spikeT, all_eventT+_offset_time)
            off_Sn_psth[cell_i,:,1] = self.calc_kde_PSTH(unit_spikeT, offT+_offset_time)
            off_Sn_psth[cell_i,:,2] = self.calc_kde_PSTH(unit_spikeT, onT+_offset_time)
            off_Sn_psth[cell_i,:,3] = self.calc_kde_PSTH(unit_spikeT, backgroundT+_offset_time)

            self.unit_stim_eventT[ind] = unit_stim_eventT

        _tmp_dict = {
            'onSubunit_eventT_allT': np.nan,
            'onSubunit_eventT_darkT': np.nan,
            'onSubunit_eventT_lightT': np.nan,
            'onSubunit_eventT_bckgndT': np.nan,
            'offSubunit_eventT_allT': np.nan,
            'offSubunit_eventT_darkT': np.nan,
            'offSubunit_eventT_lightT': np.nan,
            'offSubunit_eventT_bckgndT': np.nan
        }

        for ind in self.cells.index.values:
            if ind not in self.unit_stim_eventT.keys():
                self.unit_stim_eventT[ind] = _tmp_dict

        self.on_Sn_psth = on_Sn_psth
        self.off_Sn_psth = off_Sn_psth
        self.rf_xy = rf_xy
        self.all_stimT = eventT+_offset_time


    def save_as_df(self):

        if type(self.ball_speed) != np.ndarray:
            self.ball_speed = self.ball_speed.values

        unit_data = pd.DataFrame([])
        for unit_num, ind in enumerate(self.cells.index):
            cols = [
                'contrast',
                'contrast_tuning_bins',
                'contrast_tuning',
                'contrast_tuning_err',
                'spike_triggered_average',
                'spike_triggered_variance',
                'saccade_rightT',
                'saccade_leftT',
                'saccade_rightPSTH',
                'saccade_leftPSTH',
                'pupilradius_tuning_bins',
                'pupilradius_tuning',
                'pupilradius_tuning_err',
                'theta_tuning_bins',
                'theta_tuning',
                'theta_tuning_err',
                'phi_tuning_bins',
                'phi_tuning',
                'phi_tuning_err',
                'eyeT',
                'dEye_dpf',
                'dEye_dps',
                'theta',
                'phi',
                'ballspeed',
                'ballspeed_tuning_bins',
                'ballspeed_tuning',
                'ballspeed_tuning_err',
                'stim_PSTH_onSub_all',
                'stim_PSTH_onSub_darkRF',
                'stim_PSTH_onSub_lightRF',
                'stim_PSTH_onSub_bckgndRF',
                'stim_PSTH_offSub_all',
                'stim_PSTH_offSub_darkRF',
                'stim_PSTH_offSub_lightRF',
                'stim_PSTH_offSub_bckgndRF',
                'approx_onSub_RF_coords',
                'approx_offSub_RF_coords',
                'stimT_all_shared',
                'stimT_unit_offSub_allT',
                'stimT_unit_offSub_darkT',
                'stimT_unit_offSub_lightT',
                'stimT_unit_offSub_bckgndT',
                'stimT_unit_onSub_allT',
                'stimT_unit_onSub_darkT',
                'stimT_unit_onSub_lightT',
                'stimT_unit_onSub_bckgndT',
                'worldT'
            ]

            unit_df = pd.DataFrame(pd.Series([
                self.contrast,
                self.contrast_tuning_bins,
                self.contrast_tuning[unit_num],
                self.contrast_tuning_err[unit_num],
                self.sta[unit_num],
                self.stv[unit_num],
                self.all_eyeR,
                self.all_eyeL,
                self.rightsacc_avg[unit_num],
                self.leftsacc_avg[unit_num],
                self.pupilradius_tuning_bins,
                self.pupilradius_tuning[unit_num],
                self.pupilradius_tuning_err[unit_num],
                self.theta_tuning_bins,
                self.theta_tuning[unit_num],
                self.theta_tuning_err[unit_num],
                self.phi_tuning_bins,
                self.phi_tuning[unit_num],
                self.phi_tuning_err[unit_num],
                self.eyeT,
                self.dEye,
                self.dEye_dps,
                self.theta,
                self.phi,
                self.ball_speed,
                self.ballspeed_tuning_bins,
                self.ballspeed_tuning[unit_num],
                self.ballspeed_tuning_err[unit_num],
                self.on_Sn_psth[unit_num,:,0],
                self.on_Sn_psth[unit_num,:,1],
                self.on_Sn_psth[unit_num,:,2],
                self.on_Sn_psth[unit_num,:,3],
                self.off_Sn_psth[unit_num,:,0],
                self.off_Sn_psth[unit_num,:,1],
                self.off_Sn_psth[unit_num,:,2],
                self.off_Sn_psth[unit_num,:,3],
                self.rf_xy[unit_num,:2],
                self.rf_xy[unit_num,2:],
                self.all_stimT,
                self.unit_stim_eventT[ind]['offSubunit_eventT_allT'],
                self.unit_stim_eventT[ind]['offSubunit_eventT_darkT'],
                self.unit_stim_eventT[ind]['offSubunit_eventT_lightT'],
                self.unit_stim_eventT[ind]['offSubunit_eventT_bckgndT'],
                self.unit_stim_eventT[ind]['onSubunit_eventT_allT'],
                self.unit_stim_eventT[ind]['onSubunit_eventT_darkT'],
                self.unit_stim_eventT[ind]['onSubunit_eventT_lightT'],
                self.unit_stim_eventT[ind]['onSubunit_eventT_bckgndT'],
                self.worldT.values
            ]),dtype=object).T

            unit_df.columns = cols
            unit_df.index = [ind]

            unit_data = pd.concat([unit_data, unit_df], axis=0)

        data_out = pd.concat([self.cells, unit_data], axis=1)

        # Add a prefix to all column names
        data_out = data_out.add_prefix('Sn_')

        data_out['session'] = self.session_name

        data_out.to_hdf(os.path.join(self.recording_path,
                                     (self.recording_name+'_ephys_props.h5')), 'w')


    def save_as_dict(self):

        if type(self.ball_speed) != np.ndarray:
            self.ball_speed = self.ball_speed.values

        stim = 'Sn'
        save_dict = {
            'CRF': self.contrast_tuning_bins,
            'CRF_err': self.contrast_tuning,
            'CRF_bins':self.contrast_tuning_err,
            'STA': self.sta,
            'STV': self.stv,
            'saccade_rightT': self.all_eyeR,
            'saccade_leftT': self.all_eyeL,
            'saccade_rightPSTH': self.rightsacc_avg,
            'saccade_leftPSTH': self.leftsacc_avg,
            'PupilRadius_tuning': self.pupilradius_tuning,
            'PupilRadius_bins': self.pupilradius_tuning_bins,
            'PupilRadius_err': self.pupilradius_tuning_err,
            'Theta_tuning': self.theta_tuning,
            'Theta_bins': self.theta_tuning_bins,
            'Theta_err': self.theta_tuning_err,
            'Phi_tuning': self.phi_tuning,
            'Phi_bins': self.phi_tuning_bins,
            'Phi_err': self.phi_tuning_err,
            'eyeT': self.eyeT,
            'dEye': self.dEye_dps,
            'dEye_dpf': self.dEye,
            'theta': self.theta,
            'phi': self.phi,
            'ballspeed': self.ball_speed, 
            'ballspeed_tuning_bins': self.ballspeed_tuning_bins,
            'ballspeed_tuning': self.ballspeed_tuning,
            'ballspeed_tuning_err': self.ballspeed_tuning_err,
            'stim_PSTH_onSubReg': self.on_Sn_psth,
            'stim_PSTH_offSubReg': self.off_Sn_psth,
            'stim_PSTH_center_RF_used': self.rf_xy,
            'session': self.session_name,
            'stim': stim,
            'worldT': self.worldT.values
        }

        for key, val in save_dict.items():
            if type(val)==xr.DataArray:
                save_dict[key] = val.values
        
        # Merge Phy2 cell data in
        save_cells = self.cells.to_dict(orient='list')
        save_dict = {**save_cells, **save_dict}

        # Add 'FmLt' to the start of each key
        for old_key in save_dict.keys():
            save_dict['{}_{}'.format(stim,old_key)] = save_dict.pop(old_key)

        savepath = os.path.join(self.recording_path,
                                '{}_ephys_props.h5'.format(self.recording_name))

        fme.write_h5(savepath, save_dict)


    def analyze(self):
        # delete the existing h5 file, so that a new one can be written
        if os.path.isfile(os.path.join(self.recording_path,
                                       (self.recording_name+'_ephys_props.h5'))):
            os.remove(os.path.join(self.recording_path,
                                   (self.recording_name+'_ephys_props.h5')))

        self.detail_pdf = PdfPages(os.path.join(self.recording_path,
                                        (self.recording_name + '_detailed_analysis_figures.pdf')))
        self.diagnostic_pdf = PdfPages(os.path.join(self.recording_path,
                                        (self.recording_name + '_diagnostic_analysis_figures.pdf')))
        
        print('starting ephys analysis for '+self.recording_name)
        self.base_ephys_analysis()

        self.calc_Sn_psth()

        print('closing pdfs')

        self.detail_pdf.close()
        self.diagnostic_pdf.close()

        print('saving ephys file')
        self.save_as_df()

