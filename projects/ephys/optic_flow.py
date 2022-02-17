import os, warnings, cv2
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm
from scipy.interpolate import interp1d

import src.utils.save as ioh5
from src.utils.path import find

class OpticFlow:
    def __init__(self, base_path):
        # input files
        date_animal = '_'.join(base_path.split('/')[-3:-1])
        self.model_data_file = os.path.join(base_path, date_animal+'_ModelData_dt025_rawWorldCam_2ds.h5')
        self.ephys_path = find('*fm1*_ephys_props.h5',base_path)[0]
        self.save_name = os.path.join(base_path,'fm1_optic_flow.npz')

    def calc_optic_flow(self, crop_pix=10, win=5, lag_list=[-2,-1,0,1,2]):
        """
        lag_list: sta lags
        win: number of frames for window of optic flow
        """
        print('reading '+self.model_data_file)
        data = ioh5.load(self.model_data_file)

        # shifted worldcam data
        wc_data = data['model_vis_sm_shift']

        # total number of frames in video
        # -1 since we're taking the difference between consecutive and -1 for first WC frame being nans
        fr = wc_data.shape[0] -2
        h = wc_data.shape[1] # image height
        w = wc_data.shape[2] # image width
        flow_data = np.zeros((fr,h,w,2)) # initialize optic flow array flow_data [frame#, x, y, u/v]

        print('calculating optic flow')
        # loop over all frames to get optic flow
        for f in tqdm(np.arange(1,fr,1)):
            flow = cv2.calcOpticalFlowFarneback(wc_data[f,:,:],wc_data[f+1,:,:], None, 0.5, 3, win, 3, 5, 1.2, 0)
            flow_data[f,:,:,0] = flow[:,:,0]
            flow_data[f,:,:,1] = flow[:,:,1]

        # get active times from gyro
        nan_idxs = []
        for key in data.keys():
            nan_idxs.append(np.where(np.isnan(data[key]))[0])
        good_idxs = np.ones(len(data['model_active']),dtype=bool)
        good_idxs[data['model_active'] < .5] = False
        good_idxs[np.unique(np.hstack(nan_idxs))] = False

        good_idxs[0] = False # first worldcam frame is nan
        good_idxs[1] = False # first flow frame is nan

        full_recording = np.ones(len(data['model_active']), dtype=bool)

        # if the ephys file has topdown movement states
        if self.ephys_path is not None:
            ephys_df = pd.read_hdf(self.ephys_path)

            # interpolate topdown state (which is in the timebase of eyeT) to
            # get it into 25ms bins (timebase of modelT)
            modelT = data['model_t']
            eyeT = ephys_df['FmLt_eyeT'].iloc[0]

            forward = interp1d(eyeT, list(ephys_df['FmLt_top_forward_run'].iloc[0].astype(bool)), bounds_error=False)(modelT)
            backward = interp1d(eyeT, list(ephys_df['FmLt_top_backward_run'].iloc[0].astype(bool)), bounds_error=False)(modelT)
            fine_motion = interp1d(eyeT, list(ephys_df['FmLt_top_fine_motion'].iloc[0].astype(bool)), bounds_error=False)(modelT)
            immobile = interp1d(eyeT, list(ephys_df['FmLt_top_immobility'].iloc[0].astype(bool)), bounds_error=False)(modelT)

            activity_state_bools = [full_recording, good_idxs, ~good_idxs, forward, backward, fine_motion, immobile]
            activity_state_names = ['full', 'active_gyro', 'inactive_gyro', 'running_forward', 'running_backward', 'fine_motion', 'immobile']
        # if we're just using the gyro
        elif self.ephys_path is None:
            activity_state_bools = [full_recording, good_idxs, ~good_idxs]
            activity_state_names = ['full','active_gyro','inactive_gyro']
            
        # pixels to crop on each side of the image
        flow_amp_dict = {}
        flow_vec_dict = {}
        for state_num in range(len(activity_state_names)):
            activity_state_bool = activity_state_bools[state_num].astype(bool)
            activity_state_name = activity_state_names[state_num]

            print(activity_state_name)
            
            # filter for current def of activity
            # also need to crop worldcam to get rid of shifter artifacts
            raw_nsp = data['model_nsp'].copy()
            state_nsp = raw_nsp[activity_state_bool]
            
            raw_flow_data = flow_data.copy()
            raw_flow_data = np.vstack((np.zeros((2,raw_flow_data.shape[1],raw_flow_data.shape[2],raw_flow_data.shape[3])),raw_flow_data))
            state_flow = raw_flow_data[activity_state_bool, crop_pix:-crop_pix, crop_pix:-crop_pix]

            num_cells = np.size(state_nsp,1)
            
            # optic flow amplitude
            flow_amp = np.sqrt(state_flow[:,:,:,0]**2 + state_flow[:,:,:,1]**2) # optic flow amp
            flow_amp_mn = np.mean(flow_amp, axis=0) # get mean
            flow_amp_std = np.std(flow_amp, axis=0) # get std
            flow_amp_mnsub = flow_amp.copy() # copy for z-scoring
            flow_amp_mnsub -= flow_amp_mn # subtract mean from flow data
            flow_amp_mnsub /= flow_amp_std # divide by standard deviation
            rolled_vid = np.hstack([np.roll(flow_amp_mnsub, nframes, axis=0) for nframes in lag_list]) # incorporate lags
            model_vid = rolled_vid.reshape(flow_amp_mnsub.shape[0],-1) # reshape for sta calculation
            sta = (model_vid.T @ state_nsp)/np.sum(state_nsp,0,keepdims=True) # get sta
            flow_amp_sta = sta.T.reshape((num_cells,len(lag_list)) + flow_amp_mnsub.shape[1:]) #reshape flow amp sta into [unit,lag,x,y]

            # optic flow vector
            flow_mn = np.mean(state_flow, axis=0) # calculate flow mean
            flow_std = np.std(state_flow, axis=0) # calculate flow std
            flow_data_mnsub = state_flow.copy() # copy flow data to be z-scored
            flow_data_mnsub -= flow_mn # subtract mean from flow data
            flow_data_mnsub /= flow_std # divide by standard deviation
            rolled_vid = np.hstack([np.roll(flow_data_mnsub, nframes, axis=0) for nframes in lag_list]) # incorporate lags
            model_vid = rolled_vid.reshape(flow_data_mnsub.shape[0],-1) # reshape flow data for sta
            sta = (model_vid.T @ state_nsp)/np.sum(state_nsp,0,keepdims=True) # calculate flow sta
            flow_vector_sta = sta.T.reshape((num_cells,len(lag_list)) + flow_data_mnsub.shape[1:]) # reshape flow sta into [unit,lag,x,y,U/V]
            
            flow_amp_dict[activity_state_name] = flow_amp_sta
            flow_vec_dict[activity_state_name] = flow_vector_sta
            
        print('saving')
        if self.ephys_path is not None:
            np.savez(file=self.save_name,
                    full_vec=flow_vec_dict['full'], full_amp=flow_amp_dict['full'],
                    active_gyro_vec=flow_vec_dict['active_gyro'], active_gyro_amp=flow_amp_dict['active_gyro'],
                    inactive_gyro_vec=flow_vec_dict['inactive_gyro'], inactive_gyro_amp=flow_amp_dict['inactive_gyro'],
                    running_forward_vec=flow_vec_dict['running_forward'], running_forward_amp=flow_amp_dict['running_forward'],
                    running_backward_vec=flow_vec_dict['running_backward'], running_backward_amp=flow_amp_dict['running_backward'],
                    fine_motion_vec=flow_vec_dict['fine_motion'], fine_motion_amp=flow_amp_dict['fine_motion'],
                    immobile_vec=flow_vec_dict['immobile'], immobile_amp=flow_amp_dict['immobile'])

        elif self.ephys_path is None:
            np.savez(file=self.save_name,
                    full_vec=flow_vec_dict['full'], full_amp=flow_amp_dict['full'],
                    active_gyro_vec=flow_vec_dict['active_gyro'], active_gyro_amp=flow_amp_dict['active_gyro'],
                    inactive_gyro_vec=flow_vec_dict['inactive_gyro'], inactive_gyro_amp=flow_amp_dict['inactive_gyro'])