"""
FreelyMovingEphys/src/freelymoving.py
"""
import os
from tqdm import tqdm

import numpy as np
import pandas as pd
import xarray as xr

import cv2

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import fmEphys

class FreelyMovingLight(fmEphys.Ephys):
    def __init__(self, cfg, recording_name, recording_path):
        fmEphys.Ephys.__init__(self, cfg, recording_name, recording_path)

        self.fm = True
        self.stim = 'lt'

    def save_as_df(self):
        unit_data = pd.DataFrame([])
        for unit_num, ind in enumerate(self.cells.index):
            cols = [
                'contrast_tuning_bins',
                'contrast_tuning',
                'contrast_tuning_err',
                'spike_triggered_average',
                'spike_triggered_variance',
                'saccade_rightT',
                'saccade_leftT',
                'compensatory_rightT',
                'compensatory_leftT',
                'gazeshift_leftT',
                'gazeshift_rightT',
                'saccade_rightPSTH',
                'saccade_leftPSTH',
                'compensatory_rightPSTH',
                'compensatory_leftPSTH',
                'gazeshift_leftPSTH',
                'gazeshift_rightPSTH',
                'pupilradius_tuning_bins',
                'pupilradius_tuning',
                'pupilradius_tuning_err',
                'theta_tuning_bins',
                'theta_tuning',
                'theta_tuning_err',
                'phi_tuning_bins',
                'phi_tuning',
                'phi_tuning_err',
                'gyroz_tuning_bins',
                'gyroz_tuning',
                'gyroz_tuning_err',
                'gyrox_tuning_bins',
                'gyrox_tuning',
                'gyrox_tuning_err',
                'gyroy_tuning_bins',
                'gyroy_tuning',
                'gyroy_tuning_err',
                'imuT',
                'gyro_x',
                'gyro_y',
                'gyro_z',
                'dHead',
                'eyeT',
                'dEye_dpf',
                'dEye',
                'theta',
                'phi',
                'dGaze',
                'roll_tuning_bins',
                'roll_tuning',
                'roll_tuning_err',
                'pitch_tuning_bins',
                'pitch_tuning',
                'pitch_tuning_err',
                'roll',
                'pitch',
                'topT',
                'top_speed',
                'top_forward_run',
                'top_fine_motion',
                'top_backward_run',
                'top_immobility',
                'top_head_yaw',
                'top_body_yaw',
                'top_movement_yaw'
            ]
            unit_df = pd.DataFrame(pd.Series([
                self.contrast_tuning_bins,
                self.contrast_tuning[unit_num],
                self.contrast_tuning_err[unit_num],
                self.sta[unit_num],
                self.stv[unit_num],
                self.all_eyeR,
                self.all_eyeL,
                self.compR,
                self.compL,
                self.gazeL,
                self.gazeR,
                self.rightsacc_avg[unit_num],
                self.leftsacc_avg[unit_num],
                self.rightsacc_avg_comp[unit_num],
                self.leftsacc_avg_comp[unit_num],
                self.leftsacc_avg_gaze_shift[unit_num],
                self.rightsacc_avg_gaze_shift[unit_num],
                self.pupilradius_tuning_bins,
                self.pupilradius_tuning[unit_num],
                self.pupilradius_tuning_err[unit_num],
                self.theta_tuning_bins,
                self.theta_tuning[unit_num],
                self.theta_tuning_err[unit_num],
                self.phi_tuning_bins,
                self.phi_tuning[unit_num],
                self.phi_tuning_err[unit_num],
                self.gyroz_tuning_bins,
                self.gyroz_tuning[unit_num],
                self.gyroz_tuning_err[unit_num],
                self.gyrox_tuning_bins,
                self.gyrox_tuning[unit_num],
                self.gyrox_tuning_err[unit_num],
                self.gyroy_tuning_bins,
                self.gyroy_tuning[unit_num],
                self.gyroy_tuning_err[unit_num],
                self.imuT.values,
                self.gyro_x,
                self.gyro_y,
                self.gyro_z,
                self.dHead,
                self.eyeT,
                self.dEye,
                self.dEye_dps,
                self.theta,
                self.phi,
                self.dGaze,
                self.roll_tuning_bins,
                self.roll_tuning[unit_num],
                self.roll_tuning_err[unit_num],
                self.pitch_tuning_bins,
                self.pitch_tuning[unit_num],
                self.pitch_tuning_err[unit_num],
                self.roll,
                self.pitch,
                self.topT,
                self.top_speed_interp,
                self.top_forward_run_interp,
                self.top_fine_motion_interp,
                self.top_backward_run_interp,
                self.top_immobility_interp,
                self.top_head_yaw_interp,
                self.top_body_yaw_interp,
                self.top_movement_yaw_interp
            ]),dtype=object).T
            unit_df.columns = cols
            unit_df.index = [ind]
            
            unit_data = pd.concat([unit_data, unit_df], axis=0)

        data_out = pd.concat([self.cells, unit_data], axis=1)

        # Add a prefix to all column names
        data_out = data_out.add_prefix('FmLt_')

        data_out['session'] = self.session_name

        data_out.to_hdf(os.path.join(self.recording_path, (self.recording_name+'_ephys_props.h5')), 'w')

    def save_as_dict(self):

        stim = 'FmLt'
        save_dict = {
            'CRF': self.contrast_tuning_bins,
            'CRF_err': self.contrast_tuning,
            'CRF_bins':self.contrast_tuning_err,
            'STA': self.sta,
            'STV': self.stv,
            'saccade_rightT': self.all_eyeR,
            'saccade_leftT': self.all_eyeL,
            'compensatory_rightT': self.compR,
            'compensatory_leftT': self.compL,
            'gazeshift_leftT': self.gazeL,
            'gazeshift_rightT': self.gazeR,
            'saccade_rightPSTH': self.rightsacc_avg,
            'saccade_leftPSTH': self.leftsacc_avg,
            'compensatory_rightPSTH': self.rightsacc_avg_comp,
            'compensatory_leftPSTH': self.leftsacc_avg_comp,
            'gazeshift_leftPSTH': self.leftsacc_avg_gaze_shift,
            'gazeshift_rightPSTH': self.rightsacc_avg_gaze_shift,
            'PupilRadius_tuning': self.pupilradius_tuning,
            'PupilRadius_bins': self.pupilradius_tuning_bins,
            'PupilRadius_err': self.pupilradius_tuning_err,
            'Theta_tuning': self.theta_tuning,
            'Theta_bins': self.theta_tuning_bins,
            'Theta_err': self.theta_tuning_err,
            'Phi_tuning': self.phi_tuning,
            'Phi_bins': self.phi_tuning_bins,
            'Phi_err': self.phi_tuning_err,
            'GyroZ_tuning': self.gyroz_tuning,
            'GyroZ_bins': self.gyroz_tuning_bins,
            'GyroZ_err': self.gyroz_tuning_err,
            'GyroX_tuning': self.gyrox_tuning,
            'GyroX_bins': self.gyrox_tuning_bins,
            'GyroX_err': self.gyrox_tuning_err,
            'GyroY_tuning': self.gyroy_tuning,
            'GyroY_bins': self.gyroy_tuning_bins,
            'GyroY_err': self.gyroy_tuning_err,
            'imuT': self.imuT.values,
            'gyroZ': self.gyro_z,
            'gyroX': self.gyro_x,
            'gyroY': self.gyro_y,
            'dHead': self.dHead,
            'eyeT': self.eyeT,
            'dEye': self.dEye_dps,
            'dEye_dpf': self.dEye,
            'theta': self.theta,
            'phi': self.phi,
            'dGaze': self.dGaze,
            'roll_tuning_bins': self.roll_tuning_bins,
            'roll_tuning': self.roll_tuning,
            'roll_tuning_err': self.roll_tuning_err,
            'pitch_tuning_bins': self.pitch_tuning_bins,
            'pitch_tuning': self.pitch_tuning,
            'pitch_tuning_err': self.pitch_tuning_err,
            'roll': self.roll,
            'pitch': self.pitch,
            'topT': self.topT,
            'topdown_speed': self.top_speed_interp,
            'is_running_forward': self.top_forward_run_interp,
            'is_fine_motion': self.top_fine_motion_interp,
            'is_running_backward': self.top_backward_run_interp,
            'is_stationary': self.top_immobility_interp,
            'top_head_yaw': self.top_head_yaw_interp,
            'top_body_yaw': self.top_body_yaw_interp,
            'top_movement_yaw': self.top_movement_yaw_interp,
            'session': self.session_name,
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

        fmEphys.write_h5(savepath, save_dict)

    def analyze(self):
        # delete the existing h5 file, so that a new one can be written
        if os.path.isfile(os.path.join(self.recording_path, (self.recording_name+'_ephys_props.h5'))):
            os.remove(os.path.join(self.recording_path, (self.recording_name+'_ephys_props.h5')))

        # self.overview_pdf = PdfPages(os.path.join(self.recording_path, (self.recording_name + '_overview_analysis_figures.pdf')))
        self.detail_pdf = PdfPages(os.path.join(self.recording_path, '{}_detailed_analysis_figures.pdf'.format(self.recording_name)))
        self.diagnostic_pdf = PdfPages(os.path.join(self.recording_path, '{}_diagnostic_analysis_figures.pdf'.format(self.recording_name)))

        print('starting ephys analysis for '+self.recording_name)
        self.base_ephys_analysis()

        print('making summary and overview figures')

        print('closing pdfs')
        self.detail_pdf.close()
        self.diagnostic_pdf.close()

        print('saving ephys file')
        self.save_as_df()

class FreelyMovingDark(fmEphys.Ephys):
    def __init__(self, cfg, recording_name, recording_path):
        fmEphys.Ephys.__init__(self, cfg, recording_name, recording_path)
        
        self.fm = True
        self.stim = 'dk'

    def save_as_df(self):
        unit_data = pd.DataFrame([])
        stim = 'FmDk'
        for unit_num, ind in enumerate(self.cells.index):
            cols = [
                'contrast_tuning_bins',
                'contrast_tuning',
                'contrast_tuning_err',
                'spike_triggered_average',
                'spike_triggered_variance',
                'saccade_rightT',
                'saccade_leftT',
                'compensatory_rightT',
                'compensatory_leftT',
                'gazeshift_leftT',
                'gazeshift_rightT',
                'saccade_rightPSTH',
                'saccade_leftPSTH',
                'compensatory_rightPSTH',
                'compensatory_leftPSTH',
                'gazeshift_leftPSTH',
                'gazeshift_rightPSTH',
                'pupilradius_tuning_bins',
                'pupilradius_tuning',
                'pupilradius_tuning_err',
                'theta_tuning_bins',
                'theta_tuning',
                'theta_tuning_err',
                'phi_tuning_bins',
                'phi_tuning',
                'phi_tuning_err',
                'gyroz_tuning_bins',
                'gyroz_tuning',
                'gyroz_tuning_err',
                'gyrox_tuning_bins',
                'gyrox_tuning',
                'gyrox_tuning_err',
                'gyroy_tuning_bins',
                'gyroy_tuning',
                'gyroy_tuning_err',
                'imuT',
                'gyro_x',
                'gyro_y',
                'gyro_z',
                'dHead',
                'eyeT',
                'dEye_dpf',
                'dEye',
                'theta',
                'phi',
                'dGaze',
                'roll_tuning_bins',
                'roll_tuning',
                'roll_tuning_err',
                'pitch_tuning_bins',
                'pitch_tuning',
                'pitch_tuning_err',
                'roll',
                'pitch'
            ]
            unit_df = pd.DataFrame(pd.Series([
                self.contrast_tuning_bins,
                self.contrast_tuning[unit_num],
                self.contrast_tuning_err[unit_num],
                self.sta[unit_num],
                self.stv[unit_num],
                self.all_eyeR,
                self.all_eyeL,
                self.compR,
                self.compL,
                self.gazeL,
                self.gazeR,
                self.rightsacc_avg[unit_num],
                self.leftsacc_avg[unit_num],
                self.rightsacc_avg_comp[unit_num],
                self.leftsacc_avg_comp[unit_num],
                self.leftsacc_avg_gaze_shift[unit_num],
                self.rightsacc_avg_gaze_shift[unit_num],
                self.pupilradius_tuning_bins,
                self.pupilradius_tuning[unit_num],
                self.pupilradius_tuning_err[unit_num],
                self.theta_tuning_bins,
                self.theta_tuning[unit_num],
                self.theta_tuning_err[unit_num],
                self.phi_tuning_bins,
                self.phi_tuning[unit_num],
                self.phi_tuning_err[unit_num],
                self.gyroz_tuning_bins,
                self.gyroz_tuning[unit_num],
                self.gyroz_tuning_err[unit_num],
                self.gyrox_tuning_bins,
                self.gyrox_tuning[unit_num],
                self.gyrox_tuning_err[unit_num],
                self.gyroy_tuning_bins,
                self.gyroy_tuning[unit_num],
                self.gyroy_tuning_err[unit_num],
                self.imuT.values,
                self.gyro_x,
                self.gyro_y,
                self.gyro_z,
                self.dHead,
                self.eyeT,
                self.dEye,
                self.dEye_dps,
                self.theta,
                self.phi,
                self.dGaze,
                self.roll_tuning_bins,
                self.roll_tuning[unit_num],
                self.roll_tuning_err[unit_num],
                self.pitch_tuning_bins,
                self.pitch_tuning[unit_num],
                self.pitch_tuning_err[unit_num],
                self.roll,
                self.pitch
            ]),dtype=object).T
            unit_df.columns = cols
            unit_df.index = [ind]

            unit_data = pd.concat([unit_data, unit_df], axis=0)

        data_out = pd.concat([self.cells, unit_data], axis=1)

        # Add a prefix to all column names
        data_out = data_out.add_prefix('FmDk_')

        data_out['session'] = self.session_name

        data_out.to_hdf(os.path.join(self.recording_path, (self.recording_name+'_ephys_props.h5')), 'w')

    def save_as_dict(self):

        stim = 'FmDk'
        save_dict = {
            'CRF': self.contrast_tuning_bins,
            'CRF_err': self.contrast_tuning,
            'CRF_bins':self.contrast_tuning_err,
            'STA': self.sta,
            'STV': self.stv,
            'saccade_rightT': self.all_eyeR,
            'saccade_leftT': self.all_eyeL,
            'compensatory_rightT': self.compR,
            'compensatory_leftT': self.compL,
            'gazeshift_leftT': self.gazeL,
            'gazeshift_rightT': self.gazeR,
            'saccade_rightPSTH': self.rightsacc_avg,
            'saccade_leftPSTH': self.leftsacc_avg,
            'compensatory_rightPSTH': self.rightsacc_avg_comp,
            'compensatory_leftPSTH': self.leftsacc_avg_comp,
            'gazeshift_leftPSTH': self.leftsacc_avg_gaze_shift,
            'gazeshift_rightPSTH': self.rightsacc_avg_gaze_shift,
            'PupilRadius_tuning': self.pupilradius_tuning,
            'PupilRadius_bins': self.pupilradius_tuning_bins,
            'PupilRadius_err': self.pupilradius_tuning_err,
            'Theta_tuning': self.theta_tuning,
            'Theta_bins': self.theta_tuning_bins,
            'Theta_err': self.theta_tuning_err,
            'Phi_tuning': self.phi_tuning,
            'Phi_bins': self.phi_tuning_bins,
            'Phi_err': self.phi_tuning_err,
            'GyroZ_tuning': self.gyroz_tuning,
            'GyroZ_bins': self.gyroz_tuning_bins,
            'GyroZ_err': self.gyroz_tuning_err,
            'GyroX_tuning': self.gyrox_tuning,
            'GyroX_bins': self.gyrox_tuning_bins,
            'GyroX_err': self.gyrox_tuning_err,
            'GyroY_tuning': self.gyroy_tuning,
            'GyroY_bins': self.gyroy_tuning_bins,
            'GyroY_err': self.gyroy_tuning_err,
            'imuT': self.imuT.values,
            'gyroZ': self.gyro_z,
            'gyroX': self.gyro_x,
            'gyroY': self.gyro_y,
            'dHead': self.dHead,
            'eyeT': self.eyeT,
            'dEye': self.dEye_dps,
            'dEye_dpf': self.dEye,
            'theta': self.theta,
            'phi': self.phi,
            'dGaze': self.dGaze,
            'roll_tuning_bins': self.roll_tuning_bins,
            'roll_tuning': self.roll_tuning,
            'roll_tuning_err': self.roll_tuning_err,
            'pitch_tuning_bins': self.pitch_tuning_bins,
            'pitch_tuning': self.pitch_tuning,
            'pitch_tuning_err': self.pitch_tuning_err,
            'roll': self.roll,
            'pitch': self.pitch,
            'session': self.session_name,
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

        fmEphys.write_h5(savepath, save_dict)

    def analyze(self):
        # delete the existing h5 file, so that a new one can be written
        if os.path.isfile(os.path.join(self.recording_path, (self.recording_name+'_ephys_props.h5'))):
            os.remove(os.path.join(self.recording_path, (self.recording_name+'_ephys_props.h5')))

        # self.overview_pdf = PdfPages(os.path.join(self.recording_path, (self.recording_name + '_overview_analysis_figures.pdf')))
        self.detail_pdf = PdfPages(os.path.join(self.recording_path, (self.recording_name + '_detailed_analysis_figures.pdf')))
        self.diagnostic_pdf = PdfPages(os.path.join(self.recording_path, (self.recording_name + '_diagnostic_analysis_figures.pdf')))

        print('starting ephys analysis for '+self.recording_name)
        self.base_ephys_analysis()

        print('closing pdfs')
        self.detail_pdf.close()
        self.diagnostic_pdf.close()

        print('saving files')
        self.save_as_df()
