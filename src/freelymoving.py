"""
FreelyMovingEphys/src/freelymoving.py
"""
import os, cv2
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from src.ephys import Ephys

class FreelyMovingLight(Ephys):
    def __init__(self, config, recording_name, recording_path):
        Ephys.__init__(self, config, recording_name, recording_path)

        self.fm = True
        self.stim = 'lt'

    def overview_fig(self):
        plt.figure(figsize=(5, int(np.ceil(self.n_cells/2))), dpi=50)

        for i, ind in enumerate(self.cells.index):

            # plot waveform
            plt.subplot(self.n_cells, 4, i*4+1)
            wv = self.cells.at[ind,'waveform']
            plt.plot(np.arange(len(wv))*1000/self.ephys_samprate, wv)
            plt.xlabel('msec'); plt.title(str(ind)+' '+self.cells.at[ind,'KSLabel']+' cont='+str(self.cells.at[ind,'ContamPct']))
            
            # plot contrast response function
            plt.subplot(self.n_cells, 4, i*4+2)
            plt.errorbar(self.crf_cent, self.crf_tuning[i], yerr=self.crf_err[i])
            plt.xlabel('contrast a.u.'); plt.ylabel('sp/sec')
            plt.ylim(0, np.nanmax(self.crf_tuning[i]*1.2))

            # plot sta
            plt.subplot(self.n_cells, 4, i*4+3)
            sta = self.sta[i,:,:]
            sta_range = np.nanmax(np.abs(sta))*1.2
            if sta_range < 0.25:
                sta_range = 0.25
            plt.imshow(sta, vmin=-sta_range, vmax=sta_range, cmap='seismic')
            
            # plot eye movements
            plt.subplot(self.n_cells, 4, i*4+4)
            plt.plot(self.trange_x, self.rightsacc_avg[i,:], color='tab:blue',label='right')
            plt.plot(self.trange_x, self.leftsacc_avg[i,:],color='red',label='left')
            maxval = np.max(np.maximum(self.rightsacc_avg[i,:], self.leftsacc_avg[i,:]))
            plt.vlines(0, 0, maxval*1.5, linestyles='dotted', colors='k')
            plt.ylim([0, maxval*1.2]); plt.ylabel('sp/sec'); plt.legend()
        
        plt.tight_layout()
        plt.tight_layout(); self.overview_pdf.savefig(); plt.close()

    def glm_save(self):
        unit_data = pd.DataFrame([])
        stim = 'FmLt'
        for unit_num, ind in enumerate(self.cells.index):
            cols = [stim+'_'+i for i in ['contrast_tuning_bins',
                                        'contrast_tuning',
                                        'contrast_tuning_err',
                                        'spike_triggered_average',
                                        'spike_triggered_variance',
                                        'rightsacc_avg',
                                        'leftsacc_avg',
                                        'rightsacc_avg_gaze_shift_dEye',
                                        'leftsacc_avg_gaze_shift_dEye',
                                        'rightsacc_avg_comp_dEye',
                                        'leftsacc_avg_comp_dEye',
                                        'rightsacc_avg_gaze_shift_dHead',
                                        'leftsacc_avg_gaze_shift_dHead',
                                        'rightsacc_avg_comp_dHead',
                                        'leftsacc_avg_comp_dHead',
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
                                        'dEye_dps',
                                        'theta',
                                        'phi',
                                        'dGaze',
                                        'roll_tuning_bins',
                                        'roll_tuning',
                                        'roll_tuning_err',
                                        'pitch_tuning_bins',
                                        'pitch_tuning',
                                        'pitch_tuning_err',
                                        'glm_rf',
                                        'glm_cc',
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
                                        'top_movement_yaw']]
            unit_df = pd.DataFrame(pd.Series([self.contrast_tuning_bins,
                                    self.contrast_tuning[unit_num],
                                    self.contrast_tuning_err[unit_num],
                                    self.sta[unit_num],
                                    self.stv[unit_num],
                                    self.rightsacc_avg[unit_num],
                                    self.leftsacc_avg[unit_num],
                                    self.rightsacc_avg_gaze_shift_dEye[unit_num],
                                    self.leftsacc_avg_gaze_shift_dEye[unit_num],
                                    self.rightsacc_avg_comp_dEye[unit_num],
                                    self.leftsacc_avg_comp_dEye[unit_num],
                                    self.rightsacc_avg_gaze_shift_dHead[unit_num],
                                    self.leftsacc_avg_gaze_shift_dHead[unit_num],
                                    self.rightsacc_avg_comp_dHead[unit_num],
                                    self.leftsacc_avg_comp_dHead[unit_num],
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
                                    self.imuT,
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
                                    self.glm_rf[unit_num],
                                    self.glm_cc[unit_num],
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
                                    self.top_movement_yaw_interp]),dtype=object).T
            unit_df.columns = cols
            unit_df.index = [ind]
            unit_df['session'] = self.session_name
            unit_data = pd.concat([unit_data, unit_df], axis=0)
        data_out = pd.concat([self.cells, unit_data], axis=1)
        data_out.to_hdf(os.path.join(self.recording_path, (self.recording_name+'_ephys_props.h5')), 'w')

    def save(self):
        unit_data = pd.DataFrame([])
        stim = 'FmLt'
        for unit_num, ind in enumerate(self.cells.index):
            cols = [stim+'_'+i for i in ['contrast_tuning_bins',
                                        'contrast_tuning',
                                        'contrast_tuning_err',
                                        'spike_triggered_average',
                                        'spike_triggered_variance',
                                        'rightsacc_avg',
                                        'leftsacc_avg',
                                        'rightsacc_avg_gaze_shift_dEye',
                                        'leftsacc_avg_gaze_shift_dEye',
                                        'rightsacc_avg_comp_dEye',
                                        'leftsacc_avg_comp_dEye',
                                        'rightsacc_avg_gaze_shift_dHead',
                                        'leftsacc_avg_gaze_shift_dHead',
                                        'rightsacc_avg_comp_dHead',
                                        'leftsacc_avg_comp_dHead',
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
                                        'dEye_dps',
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
                                        'top_movement_yaw']]
            unit_df = pd.DataFrame(pd.Series([self.contrast_tuning_bins,
                                    self.contrast_tuning[unit_num],
                                    self.contrast_tuning_err[unit_num],
                                    self.sta[unit_num],
                                    self.stv[unit_num],
                                    self.rightsacc_avg[unit_num],
                                    self.leftsacc_avg[unit_num],
                                    self.rightsacc_avg_gaze_shift_dEye[unit_num],
                                    self.leftsacc_avg_gaze_shift_dEye[unit_num],
                                    self.rightsacc_avg_comp_dEye[unit_num],
                                    self.leftsacc_avg_comp_dEye[unit_num],
                                    self.rightsacc_avg_gaze_shift_dHead[unit_num],
                                    self.leftsacc_avg_gaze_shift_dHead[unit_num],
                                    self.rightsacc_avg_comp_dHead[unit_num],
                                    self.leftsacc_avg_comp_dHead[unit_num],
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
                                    self.imuT,
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
                                    self.top_movement_yaw_interp]),dtype=object).T
            unit_df.columns = cols
            unit_df.index = [ind]
            unit_df['session'] = self.session_name
            unit_data = pd.concat([unit_data, unit_df], axis=0)
        data_out = pd.concat([self.cells, unit_data], axis=1)
        data_out.to_hdf(os.path.join(self.recording_path, (self.recording_name+'_ephys_props.h5')), 'w')

    def save_glm_model_inputs(self):
        """ Save an npz file out that has inputs needed for post-processing glm.
        Just do this to avoid duplicating videos, etc. for all units, when the stim is shared.
        """
        np.savez(file=os.path.join(self.recording_path, 'glm_model_inputs.h5'),
                 model_active=self.model_active,
                 model_t=self.model_t,
                 model_video=self.model_vid,
                 model_rough_correction_video=self.glm_model_vid,
                 model_nsp=self.model_nsp,
                 model_eye_use_thresh=self.model_eye_use_thresh,
                 model_active_thresh=self.model_active_thresh,
                 model_theta=self.model_theta,
                 model_phi=self.model_phi,
                 model_speed=self.top_speed_interp,
                 model_forward_run=self.top_forward_run_interp,
                 model_fine_motion=self.top_fine_motion_interp,
                 model_backward_run=self.top_backward_run_interp,
                 model_immobility=self.top_immobility_interp,
                 model_head_yaw=self.top_head_yaw_interp,
                 model_body_yaw=self.top_body_yaw_interp,
                 model_movement_yaw=self.top_movement_yaw_interp,
                 model_raw_gyro_z=self.model_raw_gyro_z,
                 model_use=self.model_use,
                 model_roll=self.model_roll,
                 model_pitch=self.model_pitch,
                 model_gyro_z=self.model_gyro_z
        )

    def analyze(self):
        # delete the existing h5 file, so that a new one can be written
        if os.path.isfile(os.path.join(self.recording_path, (self.recording_name+'_ephys_props.h5'))):
            os.remove(os.path.join(self.recording_path, (self.recording_name+'_ephys_props.h5')))

        # self.overview_pdf = PdfPages(os.path.join(self.recording_path, (self.recording_name + '_overview_analysis_figures.pdf')))
        self.detail_pdf = PdfPages(os.path.join(self.recording_path, (self.recording_name + '_detailed_analysis_figures.pdf')))
        self.diagnostic_pdf = PdfPages(os.path.join(self.recording_path, (self.recording_name + '_diagnostic_analysis_figures.pdf')))

        print('starting ephys analysis for '+self.recording_name)
        self.base_ephys_analysis()

        print('making summary and overview figures')
        # self.overview_fig()
        # self.summary_fig()

        print('closing pdfs')
        # self.overview_pdf.close();
        self.detail_pdf.close(); self.diagnostic_pdf.close()

        print('saving ephys file')
        if self.do_rough_glm_fit:
            self.glm_save()
        elif not self.do_rough_glm_fit:
            self.save()
        
        if self.do_glm_model_preprocessing:
            print('saving inputs to full glm model')
            self.save_glm_model_inputs()

class FreelyMovingDark(Ephys):
    def __init__(self, config, recording_name, recording_path):
        Ephys.__init__(self, config, recording_name, recording_path)
        
        self.fm = True
        self.stim = 'dk'

    def save(self):
        unit_data = pd.DataFrame([])
        stim = 'FmDk'
        for unit_num, ind in enumerate(self.cells.index):
            cols = [stim+'_'+i for i in ['contrast_tuning_bins',
                                        'contrast_tuning',
                                        'contrast_tuning_err',
                                        'spike_triggered_average',
                                        'spike_triggered_variance',
                                        'rightsacc_avg',
                                        'leftsacc_avg',
                                        'rightsacc_avg_gaze_shift_dEye',
                                        'leftsacc_avg_gaze_shift_dEye',
                                        'rightsacc_avg_comp_dEye',
                                        'leftsacc_avg_comp_dEye',
                                        'rightsacc_avg_gaze_shift_dHead',
                                        'leftsacc_avg_gaze_shift_dHead',
                                        'rightsacc_avg_comp_dHead',
                                        'leftsacc_avg_comp_dHead',
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
                                        'dEye_dps',
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
                                        'pitch']]
            unit_df = pd.DataFrame(pd.Series([self.contrast_tuning_bins,
                                    self.contrast_tuning[unit_num],
                                    self.contrast_tuning_err[unit_num],
                                    self.sta[unit_num],
                                    self.stv[unit_num],
                                    self.rightsacc_avg[unit_num],
                                    self.leftsacc_avg[unit_num],
                                    self.rightsacc_avg_gaze_shift_dEye[unit_num],
                                    self.leftsacc_avg_gaze_shift_dEye[unit_num],
                                    self.rightsacc_avg_comp_dEye[unit_num],
                                    self.leftsacc_avg_comp_dEye[unit_num],
                                    self.rightsacc_avg_gaze_shift_dHead[unit_num],
                                    self.leftsacc_avg_gaze_shift_dHead[unit_num],
                                    self.rightsacc_avg_comp_dHead[unit_num],
                                    self.leftsacc_avg_comp_dHead[unit_num],
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
                                    self.imuT,
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
                                    self.pitch]),dtype=object).T
            unit_df.columns = cols
            unit_df.index = [ind]
            unit_df['session'] = self.session_name
            unit_data = pd.concat([unit_data, unit_df], axis=0)
        data_out = pd.concat([self.cells, unit_data], axis=1)
        data_out.to_hdf(os.path.join(self.recording_path, (self.recording_name+'_ephys_props.h5')), 'w')

    def analyze(self):
        # delete the existing h5 file, so that a new one can be written
        if os.path.isfile(os.path.join(self.recording_path, (self.recording_name+'_ephys_props.h5'))):
            os.remove(os.path.join(self.recording_path, (self.recording_name+'_ephys_props.h5')))

        # self.overview_pdf = PdfPages(os.path.join(self.recording_path, (self.recording_name + '_overview_analysis_figures.pdf')))
        self.detail_pdf = PdfPages(os.path.join(self.recording_path, (self.recording_name + '_detailed_analysis_figures.pdf')))
        self.diagnostic_pdf = PdfPages(os.path.join(self.recording_path, (self.recording_name + '_diagnostic_analysis_figures.pdf')))

        print('starting ephys analysis for '+self.recording_name)
        self.base_ephys_analysis()

        print('making summary and overview figures')
        # self.overview_fig()
        # self.summary_fig()

        print('saving files')
        self.save()

        print('closing pdfs')
        # self.overview_pdf.close();
        self.detail_pdf.close(); self.diagnostic_pdf.close()