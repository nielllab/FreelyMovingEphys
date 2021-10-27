class Worldcam(Camera):
    def __init__(self):
        super().__init__()
        
    def process(self):
        self.pack_position_data()
        if not self.config['preycapture_analysis']['cricket_worldcam']:
            self.xrpts.name = 'WORLD_times'
        elif self.config['preycapture_analysis']['cricket_worldcam']:
            self.xpts.name = 'WORLD_pts'
        if self.config['parameters']['outputs_and_visualization']['save_nc_vids']
            self.pack_video_frames()

    def save(self):
        if self.onfig['parameters']['outputs_and_visualization']['save_nc_vids']:
            trial_world_data = self.safe_merge([self.xrpts, self.xrframes])
        elif not self.onfig['parameters']['outputs_and_visualization']['save_nc_vids']:
            worlddlc.to_netcdf(os.path.join(config['recording_path'], str(recording_name+'_world.nc')))




"""
track_world.py
"""
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib
import cv2
from tqdm import tqdm

from utils.paths import find
from utils.format_data import h5_to_xr
from utils.aux_funcs import smooth_convolve
# from utils.dlc import runDLCbatch

def track_LED(config):
    # DLC tracking
    dlc_config_eye = config['ir_spot_in_space']['LED_eye_view_config']
    dlc_config_world = config['ir_spot_in_space']['LED_world_view_config']
    led_dir = os.path.join(config['animal_dir'], config['ir_spot_in_space']['ir_spot_in_space_dir_name'])
    led_dir_avi = find('*IR*.avi', led_dir)
    led_dir_csv = find('*IR*BonsaiTSformatted.csv', led_dir)
    if led_dir_avi == []:
        led_dir_avi = find('*IR*.avi', config['animal_dir'])
        led_dir_csv = find('*IR*BonsaiTSformatted.csv', config['animal_dir'])
        led_dir_h5 = find('*IR*.h5', config['animal_dir'])
    # get the trial name
    t_name = os.path.split('_'.join(led_dir_avi[0].split('_')[:-1]))[1]
    # find the correct eye anbd world video and time files
    eye_csv = [i for i in led_dir_csv if 'REYE' in i and 'formatted' in i][0]
    eye_avi = [i for i in led_dir_avi if 'REYE' in i and 'deinter' in i][0]
    world_csv = [i for i in led_dir_csv if 'WORLD' in i and 'formatted' in i][0]
    world_avi = [i for i in led_dir_avi if 'WORLD' in i and 'calib' in i][0]
    # generate .h5 files
    # runDLCbatch(world_avi, dlc_config_world, {'pose_estimation':{'crop_for_dlc':False, 'filter_dlc_predictions':False}})
    # runDLCbatch(eye_avi, dlc_config_eye, {'pose_estimation':{'crop_for_dlc':False, 'filter_dlc_predictions':False}})
    # then, get the h5 files for this trial that were just written to file
    led_dir_h5 = find('*IR*.h5', led_dir)
    if led_dir_h5 == []:
        led_dir_h5 = find('*IR*.h5',config['animal_dir'])
    world_h5 = [i for i in led_dir_h5 if 'WORLD' in i and 'calib' in i][0]
    eye_h5 = [i for i in led_dir_h5 if 'REYE' in i and 'deinter' in i][0]
    # format everything into an xarray
    eyexr = h5_to_xr(eye_h5, eye_csv, 'REYE', config=config)
    worldxr = h5_to_xr(world_h5, world_csv, 'WORLD', config=config) # format in xarray
    # save out the paramters in nc files
    eyexr.to_netcdf(os.path.join(led_dir, str('led_eye_positions.nc')))
    worldxr.to_netcdf(os.path.join(led_dir, str('led_world_positions.nc')))
    # then make some plots in a pdf
    if config['parameters']['outputs_and_visualization']['save_figs'] is True:
        pdf = matplotlib.backends.backend_pdf.PdfPages(os.path.join(led_dir, (t_name + 'LED_tracking.pdf')))
        
        eye_x = eyexr.sel(point_loc='light_x')
        plt.figure()
        plt.plot(eye_x); plt.title('light x position in eye')
        plt.ylabel('eye x'); plt.xlabel('frame')
        pdf.savefig()
        plt.close()
        eye_y = eyexr.sel(point_loc='light_y')
        plt.figure()
        plt.plot(eye_y); plt.title('light y position in eye')
        plt.ylabel('eye y'); plt.xlabel('frame')
        pdf.savefig()
        plt.close()
        world_x = worldxr.sel(point_loc='light_x')
        plt.figure()
        plt.plot(world_x); plt.title('light x position in worldcam')
        plt.ylabel('world x'); plt.xlabel('frame')
        pdf.savefig()
        plt.close()
        world_y = worldxr.sel(point_loc='light_y')
        plt.figure()
        plt.plot(world_y); plt.title('light y position in worldcam')
        plt.ylabel('world y'); plt.xlabel('frame')
        pdf.savefig()
        plt.close()

        # threshold out frames with low likelihood
        # seems to work well for thresh=0.99
        eye_x[eyexr.sel(point_loc='light_likelihood')<config['ir_spot_in_space']['lik_thresh_strict']] = np.nan
        eye_y[eyexr.sel(point_loc='light_likelihood')<config['ir_spot_in_space']['lik_thresh_strict']] = np.nan
        world_x[worldxr.sel(point_loc='light_likelihood')<config['ir_spot_in_space']['lik_thresh_strict']] = np.nan
        world_y[worldxr.sel(point_loc='light_likelihood')<config['ir_spot_in_space']['lik_thresh_strict']] = np.nan
        # eliminate frames in which there is very little movementin the worldcam (movements should be large!)
        orig_world_x = world_x.copy(); orig_world_y = world_y.copy()
        world_x = world_x[:-1]; world_y = world_y[:-1]
        world_x[np.logical_and(np.diff(orig_world_x)<1,np.diff(orig_world_x)>-1)] = np.nan
        world_y[np.logical_and(np.diff(orig_world_y)<1,np.diff(orig_world_y)>-1)] = np.nan

        plt.figure()
        plt.plot(eye_x); plt.title('light x position in eye (thresh applied)')
        plt.ylabel('eye x'); plt.xlabel('frame')
        pdf.savefig()
        plt.close()
        plt.figure()
        plt.plot(eye_y); plt.title('light y position in eye (thresh applied)')
        plt.ylabel('eye y'); plt.xlabel('frame')
        pdf.savefig()
        plt.close()
        plt.figure()
        plt.plot(world_x); plt.title('light x position in worldcam (thresh applied)')
        plt.ylabel('world x'); plt.xlabel('frame')
        pdf.savefig()
        plt.close()
        plt.figure()
        plt.plot(world_y); plt.title('light y position in worldcam (thresh applied)')
        plt.ylabel('world y'); plt.xlabel('frame')
        pdf.savefig()
        plt.close()

        # apply a smoothing convolution
        eye_x = smooth_convolve(eye_x); eye_y = smooth_convolve(eye_y)
        world_x = smooth_convolve(world_x); world_y = smooth_convolve(world_y)
        
        plt.figure()
        plt.plot(eye_x); plt.title('light x position in eye (conv applied)')
        plt.ylabel('eye x'); plt.xlabel('frame')
        pdf.savefig()
        plt.close()
        plt.figure()
        plt.plot(eye_y); plt.title('light y position in eye (conv applied)')
        plt.ylabel('eye y'); plt.xlabel('frame')
        pdf.savefig()
        plt.close()
        plt.figure()
        plt.plot(world_x); plt.title('light x position in worldcam (conv applied)')
        plt.ylabel('world x'); plt.xlabel('frame')
        pdf.savefig()
        plt.close()
        plt.figure()
        plt.plot(world_y); plt.title('light y position in worldcam (conv applied)')
        plt.ylabel('world y'); plt.xlabel('frame')
        pdf.savefig()
        plt.close()

        # plot eye vs world for x and y
        if len(eye_x) > len(world_x):
            diff_in_len = len(world_x) - len(eye_x)
        elif len(eye_x) < len(world_x):
            diff_in_len = len(eye_x) - len(world_x)
        print(diff_in_len)
        plt.subplots(1,2)
        plt.subplot(121)
        plt.plot(eye_x[:diff_in_len],world_x, '.')
        plt.ylabel('world x'); plt.xlabel('eye x')
        plt.title('x in eye vs world')
        plt.subplot(122)
        plt.plot(eye_y[:diff_in_len], world_y, '.')
        plt.ylabel('world y'); plt.xlabel('eye y')
        plt.title('y in eye vs world')
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        plt.subplots(1,2)
        plt.subplot(121)
        plt.plot(world_x,world_y,'.')
        plt.ylabel('world y'); plt.xlabel('world x')
        plt.title('world x vs y')
        plt.subplot(122)
        plt.plot(eye_x, eye_y,'.')
        plt.ylabel('eye y'); plt.xlabel('eye x')
        plt.title('eye x vs y')
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        plt.subplots(1,2)
        plt.subplot(121)
        plt.plot(eye_x[:diff_in_len],world_y,'.')
        plt.ylabel('world y'); plt.xlabel('eye x')
        plt.title('eye x vs world y')
        plt.subplot(122)
        plt.plot(eye_y[:diff_in_len], world_x,'.')
        plt.ylabel('world x'); plt.xlabel('eye y')
        plt.title('eye y vs world x')
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        pdf.close()
    
        np.savez(os.path.join(led_dir, (t_name + 'LED_positions.npz')), eye_x=eye_x, eye_y=eye_y, world_x=world_x, world_y=world_y)

    if config['parameters']['outputs_and_visualization']['save_avi_vids'] is True:
        plot_IR_track(world_avi, worldxr, eye_avi, eyexr, t_name, config)
    
    print('done preprocessing IR LED calibration videos')

def plot_IR_track(world_vid, world_dlc, eye_vid, eye_dlc, trial_name, config):
    
    print('plotting avi of IR LED tracking')

    led_dir = os.path.join(config['animal_dir'], config['ir_spot_in_space']['ir_spot_in_space_dir_name'])
    savepath = os.path.join(led_dir, (trial_name + '_IR_LED_tracking.avi'))
    
    world_vid_read = cv2.VideoCapture(world_vid)
    w_width = int(world_vid_read.get(cv2.CAP_PROP_FRAME_WIDTH))
    w_height = int(world_vid_read.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    eye_vid_read = cv2.VideoCapture(eye_vid)
    e_width = int(eye_vid_read.get(cv2.CAP_PROP_FRAME_WIDTH))
    e_height = int(eye_vid_read.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out_vid = cv2.VideoWriter(savepath, fourcc, 60.0, (e_width*2, e_height))

    if config['parameters']['outputs_and_visualization']['num_save_frames'] > int(world_vid_read.get(cv2.CAP_PROP_FRAME_COUNT)):
        num_save_frames = int(world_vid_read.get(cv2.CAP_PROP_FRAME_COUNT))
    else:
        num_save_frames = config['parameters']['outputs_and_visualization']['num_save_frames']

    for step in tqdm(range(0,num_save_frames)):
        w_ret, w_frame = world_vid_read.read()
        e_ret, e_frame = eye_vid_read.read()

        for ret in [w_ret, e_ret]:
            if not ret:
                break
        
        try:
            e_pt = eye_dlc.sel(frame=step)
            eye_pt_cent = (int(e_pt.sel(point_loc='light_x').values), int(e_pt.sel(point_loc='light_y').values))
            if e_pt.sel(point_loc='light_likelihood').values < config['ir_spot_in_space']['lik_thresh_strict']: # bad points in red
                e_frame = cv2.circle(e_frame, eye_pt_cent, 8, (0,0,255), 1)
            elif e_pt.sel(point_loc='light_likelihood').values >= config['ir_spot_in_space']['lik_thresh_strict']: # good points in green
                e_frame = cv2.circle(e_frame, eye_pt_cent, 8, (0,255,0), 1)
        except ValueError:
            pass
                
        try:
            w_pt = world_dlc.sel(frame=step)
            world_pt_cent = (int(w_pt.sel(point_loc='light_x').values), int(w_pt.sel(point_loc='light_y').values))
            if w_pt.sel(point_loc='light_likelihood').values < config['ir_spot_in_space']['lik_thresh_strict']: # bad points in red
                w_frame = cv2.circle(w_frame, world_pt_cent, 8, (0,0,255), 1)
            elif w_pt.sel(point_loc='light_likelihood').values >= config['ir_spot_in_space']['lik_thresh_strict']: # good points in green
                w_frame = cv2.circle(w_frame, world_pt_cent, 8, (0,255,0), 1)
        except ValueError:
            pass
                
        plotted = np.concatenate([e_frame, w_frame], axis=1)

        out_vid.write(plotted)

    out_vid.release()