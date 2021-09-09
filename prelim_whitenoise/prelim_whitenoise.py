"""
prelim_whitenoise.py
"""
from glob import glob
import os
from multiprocessing import freeze_support
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.backends.backend_pdf import PdfPages

from utils.ephys import plot_STA, plot_STV, plot_spike_rate_vs_var, plot_spike_raster
from scipy.interpolate import interp1d
from utils.video_correction import calibrate_new_world_vids
from utils.deinterlace import deinterlace_data
from utils.format_data import format_frames, h5_to_xr, safe_xr_merge
from utils.paths import find

def main(whitenoise_directory, probe):
    temp_config = {
        'animal_dir': whitenoise_directory,
        'deinterlace':{
            'flip_eye_during_deinter': True,
            'flip_world_during_deinter': True
        },
        'calibration': {
            'world_checker_npz': 'E:/freely_moving_ephys/camera_calibration_params/world_checkerboard_calib.npz'
        },
        'parameters':{
            'follow_strict_naming': True,
            'outputs_and_visualization':{
                'save_nc_vids': True,
                'dwnsmpl': 0.25
            },
            'ephys':{
                'ephys_sample_rate': 30000
            }
        }
    }
    # find world files
    world_vids = glob(os.path.join(whitenoise_directory, '*WORLD.avi'))
    world_times = glob(os.path.join(whitenoise_directory, '*WORLD_BonsaiTS.csv'))
    # deinterlace world video
    deinterlace_data(temp_config, world_vids, world_times)
    # apply calibration parameters to world video
    calibrate_new_world_vids(temp_config)
    # organize nomenclature
    trial_units = []; name_check = []; path_check = []
    for avi in find('*.avi', temp_config['animal_dir']):
        bad_list = ['plot','IR','rep11','betafpv','side_gaze'] # don't use trials that have these strings in their path
        if temp_config['parameters']['follow_strict_naming'] is True:
            if all(bad not in avi for bad in bad_list):
                split_name = avi.split('_')[:-1]
                trial = '_'.join(split_name)
                path_to_trial = os.path.join(os.path.split(trial)[0])
                trial_name = os.path.split(trial)[1]
        elif temp_config['parameters']['follow_strict_naming'] is False:
            if all(bad not in avi for bad in bad_list):
                trial_path_noext = os.path.splitext(avi)[0]
                path_to_trial, trial_name_long = os.path.split(trial_path_noext)
                trial_name = '_'.join(trial_name_long.split('_')[:3])
        try:
            if trial_name not in name_check:
                trial_units.append([path_to_trial, trial_name])
                path_check.append(path_to_trial); name_check.append(trial_name)
        except UnboundLocalError:
            pass
    # there should only be one item in trial_units in this case
    # iterate into that
    for trial_unit in trial_units:
        temp_config['trial_path'] = trial_unit[0]
        t_name = trial_unit[1]
        # find the timestamps and video for all camera inputs
        trial_cam_csv = find(('*BonsaiTS*.csv'), temp_config['trial_path'])
        trial_cam_avi = find(('*.avi'), temp_config['trial_path'])
        trial_cam_csv = [x for x in trial_cam_csv if x != []]
        trial_cam_avi = [x for x in trial_cam_avi if x != []]
        # filter the list of files for the current trial to get the world view of this side
        world_csv = [i for i in trial_cam_csv if 'WORLD' in i and 'formatted' in i][0]
        world_avi = [i for i in trial_cam_avi if 'WORLD' in i and 'calib' in i][0]
        # make an xarray of timestamps without dlc points, since there aren't any for world camera
        worlddlc = h5_to_xr(pt_path=None, time_path=world_csv, view=('WORLD'), config=temp_config)
        worlddlc.name = 'WORLD_times'
        # make xarray of video frames
        if temp_config['parameters']['outputs_and_visualization']['save_nc_vids'] is True:
            xr_world_frames = format_frames(world_avi, temp_config); xr_world_frames.name = 'WORLD_video'
        # merge but make sure they're not off in lenght by one value, which happens occasionally
        print('saving nc file of world view...')
        if temp_config['parameters']['outputs_and_visualization']['save_nc_vids'] is True:
            trial_world_data = safe_xr_merge([worlddlc, xr_world_frames])
            trial_world_data.to_netcdf(os.path.join(temp_config['trial_path'], str(t_name+'_world.nc')), engine='netcdf4', encoding={'WORLD_video':{"zlib": True, "complevel": 4}})
        elif temp_config['parameters']['outputs_and_visualization']['save_nc_vids'] is False:
            worlddlc.to_netcdf(os.path.join(temp_config['trial_path'], str(t_name+'_world.nc')))
        # now start minimal ephys analysis
        print('generating ephys plots')
        pdf = PdfPages(os.path.join(whitenoise_directory, (t_name + '_prelim_wn_figures.pdf')))
        ephys_file_path = glob(os.path.join(whitenoise_directory, '*_ephys_merge.json'))[0]
        world_file_path = glob(os.path.join(whitenoise_directory, '*_world.nc'))[0]
        world_data = xr.open_dataset(world_file_path)
        world_vid_raw = np.uint8(world_data['WORLD_video'])
        # ephys data
        if '16' in probe:
            ch_count = 16
        elif '64' in probe:
            ch_count = 64
        elif '128' in probe:
            ch_count = 128
        ephys_data = pd.read_json(ephys_file_path)
        ephysT0 = ephys_data.iloc[0,12]
        worldT = world_data.timestamps - ephysT0
        ephys_data['spikeTraw'] = ephys_data['spikeT'].copy()
        # sort ephys units by channel
        ephys_data = ephys_data.sort_values(by='ch', axis=0, ascending=True)
        ephys_data = ephys_data.reset_index()
        ephys_data = ephys_data.drop('index', axis=1)
        # correct offset between ephys and other data inputs
        offset0 = 0.1
        drift_rate = -0.1/1000
        for i in ephys_data.index:
            ephys_data.at[i,'spikeT'] = np.array(ephys_data.at[i,'spikeTraw']) - (offset0 + np.array(ephys_data.at[i,'spikeTraw']) *drift_rate)
        # get cells labeled as good
        goodcells = ephys_data.loc[ephys_data['group']=='good']
        # occasional problem with worldcam timestamps
        if worldT[0]<-600:
            worldT = worldT + 8*60*60
        # resize worldcam to make more manageable
        world_vid = world_vid_raw.copy()
        # img correction applied to worldcam
        cam_gamma = 2
        world_norm = (world_vid/255)**cam_gamma
        std_im = np.std(world_norm,axis=0)
        std_im[std_im<10/255] = 10/255
        img_norm = (world_norm-np.mean(world_norm,axis=0))/std_im
        img_norm = img_norm * (std_im>20/255)
        contrast = np.empty(worldT.size)
        for i in range(worldT.size):
            contrast[i] = np.std(img_norm[i,:,:])
        newc = interp1d(worldT,contrast,fill_value="extrapolate")
        # bin ephys spike times as spike rate / s
        dt = 0.025
        t = np.arange(0, np.max(worldT),dt)
        ephys_data['rate'] = np.nan
        ephys_data['rate'] = ephys_data['rate'].astype(object)
        for i,ind in enumerate(ephys_data.index):
            ephys_data.at[ind,'rate'], bins = np.histogram(ephys_data.at[ind,'spikeT'],t)
        ephys_data['rate']= ephys_data['rate']/dt
        goodcells = ephys_data.loc[ephys_data['group']=='good']
        n_units = len(goodcells)
        contrast_interp = newc(t[0:-1])
        # worldcam interp and set floor to values
        img_norm[img_norm<-2] = -2
        movInterp = interp1d(worldT,img_norm,axis=0, bounds_error=False) # added extrapolate for cases where x_new is below interpolation range
        # raster
        raster_fig = plot_spike_raster(goodcells)
        pdf.savefig()
        plt.close()
        print('making diagnostic plots')
        # plot contrast over entire video
        plt.figure()
        plt.plot(worldT[0:12000],contrast[0:12000])
        plt.xlabel('time')
        plt.ylabel('contrast')
        pdf.savefig()
        plt.close()
        # plot contrast over ~2min
        plt.figure()
        plt.plot(t[0:600],contrast_interp[0:600])
        plt.xlabel('secs'); plt.ylabel('contrast')
        pdf.savefig()
        plt.close()
        # worldcam timing diff
        plt.figure()
        plt.plot(np.diff(worldT)); plt.xlabel('frame'); plt.ylabel('deltaT'); plt.title('world cam')
        pdf.savefig()
        plt.close()
        print('getting contrast response function')
        crange = np.arange(0,1.2,0.1)
        crf_cent, crf_tuning, crf_err, crf_fig = plot_spike_rate_vs_var(contrast, crange, goodcells, worldT, t, 'contrast')
        pdf.savefig()
        plt.close()
        print('getting spike-triggered average')
        _, STA_singlelag_fig = plot_STA(goodcells, img_norm, worldT, movInterp, ch_count, lag=2, show_title=True)
        pdf.savefig()
        plt.close()
        print('getting spike-triggered average with range in lags')
        _, STA_multilag_fig = plot_STA(goodcells, img_norm, worldT, movInterp, ch_count, lag=np.arange(-2,8,2), show_title=False)
        pdf.savefig()
        plt.close()
        print('getting spike-triggered variance')
        _, STV_fig = plot_STV(goodcells, movInterp, img_norm, worldT)
        pdf.savefig()
        plt.close()
        print('closing pdf')
        pdf.close()
        print('done')