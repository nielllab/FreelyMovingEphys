from utils.ephys import Ephys

class FreelyMovingLight(Ephys):
    def __init__(self, config, recording_name, recording_path):
        super.__init__(self, config, recording_name, recording_path)

        self.fm = True
        self.stim = 'lt'

    def save(self):
        unit_data = pd.DataFrame([])
        stim = 'fm1'
        for unit_num, ind in enumerate(self.cells.index):
            cols = [stim+'_'+i for i in ['c_range',
                                        'crf_cent',
                                        'crf_tuning',
                                        'crf_err',
                                        'spike_triggered_average',
                                        'sta_shape',
                                        'spike_triggered_variance',
                                        'upsacc_avg',
                                        'downsacc_avg',
                                        'upsacc_avg_gaze_shift_dEye',
                                        'downsacc_avg_gaze_shift_dEye',
                                        'upsacc_avg_comp_dEye',
                                        'downsacc_avg_comp_dEye',
                                        'upsacc_avg_gaze_shift_dHead',
                                        'downsacc_avg_gaze_shift_dHead',
                                        'upsacc_avg_comp_dHead',
                                        'downsacc_avg_comp_dHead',
                                        'spike_rate_vs_pupil_radius_cent',
                                        'spike_rate_vs_pupil_radius_tuning',
                                        'spike_rate_vs_pupil_radius_err',
                                        'spike_rate_vs_theta_cent',
                                        'spike_rate_vs_theta_tuning',
                                        'spike_rate_vs_theta_err',
                                        'spike_rate_vs_gz_cent',
                                        'spike_rate_vs_gz_tuning',
                                        'spike_rate_vs_gz_err',
                                        'spike_rate_vs_gx_cent',
                                        'spike_rate_vs_gx_tuning',
                                        'spike_rate_vs_gx_err',
                                        'spike_rate_vs_gy_cent',
                                        'spike_rate_vs_gy_tuning',
                                        'spike_rate_vs_gy_err',
                                        'trange',
                                        'dHead',
                                        'dEye',
                                        'eyeT',
                                        'theta',
                                        'phi',
                                        'gz',
                                        'spike_rate_vs_roll_cent',
                                        'spike_rate_vs_roll_tuning',
                                        'spike_rate_vs_roll_err',
                                        'spike_rate_vs_pitch_cent',
                                        'spike_rate_vs_pitch_tuning',
                                        'spike_rate_vs_pitch_err',
                                        'glm_receptive_field',
                                        'glm_cc',
                                        'spike_rate_vs_phi_cent',
                                        'spike_rate_vs_phi_tuning',
                                        'spike_rate_vs_phi_err',
                                        'accT',
                                        'roll',
                                        'pitch']]
            unit_df = pd.DataFrame(pd.Series([self.contrast_range,
                                    self.crf_cent,
                                    self.crf_tuning[unit_num],
                                    self.crf_err[unit_num],
                                    np.ndarray.flatten(self.sta[unit_num]),
                                    np.shape(self.sta[unit_num]),
                                    np.ndarray.flatten(self.stv[unit_num]),
                                    self.upsacc_avg[unit_num],
                                    self.downsacc_avg[unit_num],
                                    self.upsacc_avg_gaze_shift_dEye[unit_num],
                                    self.downsacc_avg_gaze_shift_dEye[unit_num],
                                    self.upsacc_avg_comp_dEye[unit_num],
                                    self.downsacc_avg_comp_dEye[unit_num],
                                    self.upsacc_avg_gaze_shift_dHead[unit_num],
                                    self.downsacc_avg_gaze_shift_dHead[unit_num],
                                    self.upsacc_avg_comp_dHead[unit_num],
                                    self.downsacc_avg_comp_dHead[unit_num],
                                    self.spike_rate_vs_pupil_radius_cent,
                                    self.spike_rate_vs_pupil_radius_tuning[unit_num],
                                    self.spike_rate_vs_pupil_radius_err[unit_num],
                                    self.spike_rate_vs_theta_cent,
                                    self.spike_rate_vs_theta_tuning[unit_num],
                                    self.spike_rate_vs_theta_err[unit_num],
                                    self.spike_rate_vs_gz_cent,
                                    self.spike_rate_vs_gz_tuning[unit_num],
                                    self.spike_rate_vs_gz_err[unit_num],
                                    self.spike_rate_vs_gx_cent,
                                    self.spike_rate_vs_gx_tuning[unit_num],
                                    self.spike_rate_vs_gx_err[unit_num],
                                    self.spike_rate_vs_gy_cent,
                                    self.spike_rate_vs_gy_tuning[unit_num],
                                    self.spike_rate_vs_gy_err[unit_num],
                                    self.trange,
                                    self.dHead,
                                    self.dEye,
                                    self.eyeT,
                                    self.theta,
                                    self.phi,
                                    self.dGaze,
                                    self.spike_rate_vs_roll_cent,
                                    self.spike_rate_vs_roll_tuning[unit_num],
                                    self.spike_rate_vs_roll_err[unit_num],
                                    self.spike_rate_vs_pitch_cent,
                                    self.spike_rate_vs_pitch_tuning[unit_num],
                                    self.spike_rate_vs_pitch_err[unit_num],
                                    self.glm_rf[unit_num],
                                    self.glm_cc[unit_num],
                                    self.spike_rate_vs_phi_cent,
                                    self.spike_rate_vs_phi_tuning[unit_num],
                                    self.spike_rate_vs_phi_err[unit_num],
                                    self.imuT,
                                    self.roll,
                                    self.pitch]),dtype=object).T
            unit_df.columns = cols
            unit_df.index = [ind]
            unit_df['session'] = self.session_name
            unit_data = pd.concat([unit_data, unit_df], axis=0)

    def glm_save(self):
        """ Save a different h5 file out that has inputs needed for post-processing glm.
        Just do this to avoid duplicating videos, etc. for all units, when the stim is shared.
        """
        savedata = {
            'model_active': self.model_active,
            'model_t': self.model_t,
            'model_glm_vid': self.glm_model_vid,
            'model_nsp': self.model_nsp,
            'model_eye_use_thresh': self.model_eye_use_thresh,
            'model_active_thresh': self.model_active_thresh,
            'model_theta': self.model_theta,
            'model_phi': self.model_phi,
            'model_raw_gyro_z': self.model_raw_gyro_z,
            'model_use': self.model_use,
            'model_roll': self.model_roll,
            'model_pitch': self.model_pitch,
            'model_gyro_z': self.model_gyro_z
        }
        

    def process(self):
        


class FreelyMovingDark(Ephys):
    def __init__(self, config):
        FreelyMovingLight.__init__(self, config)

        self.fm = True
        self.stim = 'dk'

    def save(self):

    def process(self):