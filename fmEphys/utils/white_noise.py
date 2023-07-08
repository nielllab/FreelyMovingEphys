"""
fmEphys/utils/white_noise.py

Analysis of neural responses of the head-fixed
white noise stimulus.

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


class HeadFixedWhiteNoise(fme.Ephys):


    def __init__(self, cfg, recording_name, recording_path):

        fme.Ephys.__init__(self, cfg, recording_name, recording_path)

        self.fm = False
        self.stim = 'wn'


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
                'dEye',
                'theta',
                'phi',
                'ballspeed',
                'ballspeed_tuning_bins',
                'ballspeed_tuning',
                'ballspeed_tuning_err',
                'lfp_power',
                'layer5cent_from_lfp',
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
                self.lfp_power_profiles,
                self.lfp_layer5_centers,
                self.worldT.values
            ]),dtype=object).T

            unit_df.columns = cols
            unit_df.index = [ind]

            unit_data = pd.concat([unit_data, unit_df], axis=0)

        data_out = pd.concat([self.cells, unit_data], axis=1)

        # Add a prefix to all column names
        data_out = data_out.add_prefix('Wn_')

        data_out['session'] = self.session_name

        data_out.to_hdf(os.path.join(self.recording_path,
                       (self.recording_name+'_ephys_props.h5')), 'w')


    def save_as_dict(self):

        if type(self.ball_speed) != np.ndarray:
            self.ball_speed = self.ball_speed.values

        stim = 'Wn'
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
            'lfp_power_profiles': self.lfp_power_profiles,
            'laminar_position_layer5': self.lfp_layer5_centers,
            'session': self.session_name,
            'worldT': self.worldT.values,
            'stim': stim
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

        print('closing pdfs')
        self.detail_pdf.close()
        self.diagnostic_pdf.close()

        print('saving ephys file')
        self.save_as_df()

