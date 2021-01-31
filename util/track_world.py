"""
track_world.py

tracking world camera and finding pupil rotation

Jan. 10, 2021
"""

# package imports
import xarray as xr
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import FuncFormatter
import cv2
from scipy import signal
from scipy.optimize import curve_fit
import scipy.stats as st
import time
import subprocess as sp
import multiprocessing
import sys
import warnings
from scipy import ndimage
import time
from tqdm import tqdm
import matplotlib as mpl
from astropy.convolution import convolve
from scipy.interpolate import interp1d

# module imports
from util.time import open_time
from util.paths import find
from util.aux_funcs import nanxcorr
from util.dlc import run_DLC_on_LED
from util.format_data import h5_to_xr

def smooth_tracking(y, box_pts=3):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def track_LED(config):
    # DLC tracking
    calib = config['calibration']
    dlc_config_eye = calib['eye_LED_config']
    dlc_config_world = calib['world_LED_config']
    led_dir = os.path.join(config['data_path'], config['LED_dir_name'])
    led_dir_avi = find('*IR*.avi', led_dir)
    led_dir_csv = find('*IR*BonsaiTSformatted.csv', led_dir)
    if led_dir_avi == []:
        led_dir_avi = find('*IR*.avi', config['data_path'])
        led_dir_csv = find('*IR*BonsaiTSformatted.csv', config['data_path'])
        led_dir_h5 = find('*IR*.h5', config['data_path'])
    # get the trial name
    t_name = os.path.split('_'.join(led_dir_avi[0].split('_')[:-1]))[1]
    # find the correct eye anbd world video and time files
    eye_csv = [i for i in led_dir_csv if 'REYE' in i and 'formatted' in i][0]
    eye_avi = [i for i in led_dir_avi if 'REYE' in i and 'deinter' in i][0]
    world_csv = [i for i in led_dir_csv if 'WORLD' in i and 'formatted' in i][0]
    world_avi = [i for i in led_dir_avi if 'WORLD' in i and 'calib' in i][0]
    # generate .h5 files
    run_DLC_on_LED(dlc_config_world, world_avi)
    run_DLC_on_LED(dlc_config_eye, eye_avi)
    # then, get the h5 files for this trial that were just written to file
    led_dir_h5 = find('*IR*.h5', led_dir)
    if led_dir_h5 == []:
        led_dir_h5 = find('*IR*.h5',config['data_path'])
    world_h5 = [i for i in led_dir_h5 if 'WORLD' in i and 'calib' in i][0]
    eye_h5 = [i for i in led_dir_h5 if 'REYE' in i and 'deinter' in i][0]
    # format everything into an xarray
    eyexr = h5_to_xr(eye_h5, eye_csv, 'REYE', config=config)
    worldxr = h5_to_xr(world_h5, world_csv, 'WORLD', config=config) # format in xarray
    # save out the paramters in nc files
    eyexr.to_netcdf(os.path.join(config['data_path'], str('led_eye_positions.nc')))
    worldxr.to_netcdf(os.path.join(config['data_path'], str('led_world_positions.nc')))
    # then make some plots in a pdf
    if config['save_figs'] is True:
        pdf = matplotlib.backends.backend_pdf.PdfPages(os.path.join(config['data_path'], (t_name + 'LED_tracking.pdf')))
        
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
        eye_x[eyexr.sel(point_loc='light_likelihood')<config['lik_thresh_strict']] = np.nan
        eye_y[eyexr.sel(point_loc='light_likelihood')<config['lik_thresh_strict']] = np.nan
        world_x[worldxr.sel(point_loc='light_likelihood')<config['lik_thresh_strict']] = np.nan
        world_y[worldxr.sel(point_loc='light_likelihood')<config['lik_thresh_strict']] = np.nan
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
        eye_x = smooth_tracking(eye_x); eye_y = smooth_tracking(eye_y)
        world_x = smooth_tracking(world_x); world_y = smooth_tracking(world_y)

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
        diff_in_len = len(world_x) - len(eye_x)
        plt.subplots(1,2)
        plt.subplot(121)
        plt.plot(eye_x,world_x[:-diff_in_len], '.')
        plt.ylabel('world x'); plt.xlabel('eye x')
        plt.title('x in eye vs world')
        plt.subplot(122)
        plt.plot(eye_y, world_y[:-diff_in_len], '.')
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
        plt.plot(eye_x,world_y[:-diff_in_len],'.')
        plt.ylabel('world y'); plt.xlabel('eye x')
        plt.title('eye x vs world y')
        plt.subplot(122)
        plt.plot(eye_y, world_x[:-diff_in_len],'.')
        plt.ylabel('world x'); plt.xlabel('eye y')
        plt.title('eye y vs world x')
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        pdf.close()
    
    if config['save_avi_vids'] is True:
        plot_IR_track(world_avi, worldxr, eye_avi, eyexr, t_name, config)
    
    print('done preprocessing IR LED calibration videos')

def adjust_world(world_vid, world_dlc, eye_vid, eye_dlc, trial_name, config):
    
    print('plotting avi of IR LED tracking')

    savepath = os.path.join(config['data_path'], (trial_name + '_IR_LED_tracking.avi'))
    
    world_vid_read = cv2.VideoCapture(world_vid)
    w_width = int(world_vid_read.get(cv2.CAP_PROP_FRAME_WIDTH))
    w_height = int(world_vid_read.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    eye_vid_read = cv2.VideoCapture(eye_vid)
    e_width = int(eye_vid_read.get(cv2.CAP_PROP_FRAME_WIDTH))
    e_height = int(eye_vid_read.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out_vid = cv2.VideoWriter(savepath, fourcc, 60.0, (e_width*2, e_height))

    if config['num_save_frames'] > int(world_vid_read.get(cv2.CAP_PROP_FRAME_COUNT)):
        num_save_frames = int(world_vid_read.get(cv2.CAP_PROP_FRAME_COUNT))
    else:
        num_save_frames = config['num_save_frames']

    for step in tqdm(range(0,num_save_frames)):
        w_ret, w_frame = world_vid_read.read()
        e_ret, e_frame = eye_vid_read.read()

        for ret in [w_ret, e_ret]:
            if not ret:
                break
        
        try:
            e_pt = eye_dlc.sel(frame=step)
            eye_pt_cent = (int(e_pt.sel(point_loc='light_x').values), int(e_pt.sel(point_loc='light_y').values))
            if e_pt.sel(point_loc='light_likelihood').values < config['lik_thresh_strict']: # bad points in red
                e_frame = cv2.circle(e_frame, eye_pt_cent, 8, (0,0,255), 1)
            elif e_pt.sel(point_loc='light_likelihood').values >= config['lik_thresh_strict']: # good points in green
                e_frame = cv2.circle(e_frame, eye_pt_cent, 8, (0,255,0), 1)
        except ValueError:
            pass
                
        try:
            w_pt = world_dlc.sel(frame=step)
            world_pt_cent = (int(w_pt.sel(point_loc='light_x').values), int(w_pt.sel(point_loc='light_y').values))
            if w_pt.sel(point_loc='light_likelihood').values < config['lik_thresh_strict']: # bad points in red
                w_frame = cv2.circle(w_frame, world_pt_cent, 8, (0,0,255), 1)
            elif w_pt.sel(point_loc='light_likelihood').values >= config['lik_thresh_strict']: # good points in green
                w_frame = cv2.circle(w_frame, world_pt_cent, 8, (0,255,0), 1)
        except ValueError:
            pass
                
        plotted = np.concatenate([e_frame, w_frame], axis=1)

        out_vid.write(plotted)

    out_vid.release()

# basic world shifting without pupil rotation
def worldcam_correction(worldvid, eyeT, th, phi, worldT, config):

    overview_pdf = PdfPages(os.path.join(file_dict['save'], (file_dict['name'] + '_overview_analysis_figures.pdf')))

    print('getting worldcam correction')
    number_of_iterations = 5000
    termination_eps = 1e-4
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)
    warp_mode = cv2.MOTION_TRANSLATION
    max_frames = 60*300
    cc = np.zeros(max_frames); xshift = np.zeros(max_frames); yshift = np.zeros(max_frames);
    for i in tqdm(range(max_frames)):
        warp_matrix = np.eye(2, 3, dtype=np.float32)
        (cc[i], warp_matrix) = cv2.findTransformECC (world_vid[i,:,:],world_vid[i+1,:,:],warp_matrix, warp_mode, criteria, inputMask = None, gaussFiltSize = 1)
        xshift[i] = warp_matrix[0,2]; yshift[i] = warp_matrix[1,2]

    th_interp = interp1d(eyeT,th,bounds_error = False)
    phi_interp = interp1d(eyeT, phi, bounds_error = False)
    dth = np.diff(th_interp(worldT))
    dphi = np.diff(phi_interp(worldT))
    plt.figure(figsize = (12,8))
    plt.subplot(2,2,1)
    plt.plot(dth[0:max_frames],xshift[0:max_frames],'.');plt.plot([-5, 5], [5, -5],'r'); plt.xlim(-8,8); plt.ylim(-6,6); plt.xlabel('dtheta'); plt.ylabel('xshift')
    plt.subplot(2,2,2)
    plt.plot(dth[0:max_frames],yshift[0:max_frames],'.');plt.plot([-5, 5], [5, -5],'r'); plt.xlim(-8,8); plt.ylim(-6,6); plt.xlabel('dtheta'); plt.ylabel('yshift')
    plt.subplot(2,2,3)
    plt.plot(dphi[0:max_frames],xshift[0:max_frames],'.');plt.plot([-5, 5], [5, -5],'r'); plt.xlim(-8,8); plt.ylim(-6,6); plt.xlabel('dphi'); plt.ylabel('xshift')
    plt.subplot(2,2,4)
    plt.plot(dphi[0:max_frames],yshift[0:max_frames],'.');plt.plot([-5, 5], [5, -5],'r'); plt.xlim(-8,8); plt.ylim(-6,6); plt.xlabel('dphi'); plt.ylabel('yshift')
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(worldT[0:max_frames],cc); plt.ylabel('cc')
    if file_dict['imu'] is not None:
        plt.subplot(3,1,2)
        plt.plot(worldT[0:max_frames],xshift, label = 'image x shift');
        plt.plot(accT,-(gz-2.9)*7.5, label = 'gyro')
        #plt.plot(worldT[0:max_frames],yshift, label = 'y');
        #plt.plot(eyeT[0:-1],-dEye,label = 'eye dtheta')
        plt.xlim(0,2); plt.ylim(-2,2)
        plt.xlabel('secs'); plt.ylabel('deg')
        plt.legend()
    plt.subplot(3,1,3)
    plt.plot(worldT[0:max_frames],xshift, label = 'image x shift');
    #plt.plot(accT,-(gz-2.9)*7.5, label = 'gyro')
    #plt.plot(worldT[0:max_frames],yshift, label = 'y');
    plt.plot(worldT[0:-1],-dphi,'r',label = 'eye dtheta', alpha = 1)
    plt.xlim(0,2); plt.ylim(-0.5,0.5)
    plt.xlabel('secs'); plt.ylabel('deg')
    plt.legend()
    plt.tight_layout()
    diagnostic_pdf.savefig()
    plt.close()

    xmodel = LinearRegression()
    ymodel = LinearRegression()
    eyeData = np.zeros((max_frames,2))
    eyeData[:,0] = dth[0:max_frames];
    eyeData[:,1] = dphi[0:max_frames];
    xshiftdata = xshift[0:max_frames];
    yshiftdata = yshift[0:max_frames];
    usedata = ~np.isnan(eyeData[:,0]) & ~np.isnan(eyeData[:,1])  & (np.abs(eyeData[:,0])<2) & (np.abs(eyeData[:,1])<2)
    xmodel.fit(eyeData[usedata,:],xshiftdata[usedata])

    #offset0 = xmodel.intercept
    xmap = xmodel.coef_;
    print(xmap)

    ymodel.fit(eyeData[usedata,:],yshiftdata[usedata])
    ymap = ymodel.coef_;

    # eye correction movie
    print('getting eye correction movie')
    tr = [15,20]
    fig = plt.figure(figsize = (8,16))
    gs = fig.add_gridspec(10,1)
    axEye = fig.add_subplot(gs[0,0])
    axWorld = fig.add_subplot(gs[0:3,:])
    axWorldFix = fig.add_subplot(gs[3:6,:])

    axTheta = fig.add_subplot(gs[6,:])
    axPhi = fig.add_subplot(gs[7,:])
    axOmega = fig.add_subplot(gs[8,:])
    axGyro = fig.add_subplot(gs[9,:])

    th = np.array((eye_params.sel(ellipse_params = 'theta')-np.nanmean(eye_params.sel(ellipse_params = 'theta')))*180/3.14159)
    phi = np.array((eye_params.sel(ellipse_params = 'phi')-np.nanmean(eye_params.sel(ellipse_params = 'phi')))*180/3.14159)

    axTheta.plot(eyeT,th)
    axTheta.set_xlim(tr[0],tr[1]); 
    axTheta.set_ylabel('theta - deg'); axTheta.set_ylim(-30,30)

    axPhi.plot(eyeT,phi)
    axPhi.set_xlim(tr[0],tr[1]); 
    axPhi.set_ylabel('phi - deg'); axPhi.set_ylim(-30,30)

    #axOmega.plot(eyeT,omega)
    axOmega.set_xlim(tr[0],tr[1]); 
    axOmega.set_ylabel('omega - deg'); axOmega.set_ylim(-20,20)

    if free_move & has_imu:
        axGyro.plot(accT,gz)
        axGyro.set_xlim(tr[0],tr[1]); 
        axGyro.set_ylabel('gyro - deg'); axGyro.set_ylim(1,4)

    thInterp =interp1d(eyeT,th)
    phiInterp =interp1d(eyeT,phi)
    pix_per_deg = 1.6

    vidfile = os.path.join(file_dict['save'], (file_dict['name']+'_unit'+str(this_unit)+'_corrected.mp4'))
    # now animate
    writer = FFMpegWriter(fps=30)
    with writer.saving(fig, vidfile, 100):
    #    for t in np.arange(tr[0],tr[1],1/30):
        for t in tqdm(worldT[(worldT>tr[0]) & (worldT<tr[1])]):        
            # show eye and world frames
            axEye.cla(); axEye.axis('off'); 
            axEye.imshow(eyeInterp(t),'gray',vmin=0,vmax=255,aspect = "equal")
            #axEye.set_xlim(0,160); axEye.set_ylim(0,120)
            
            world = worldInterp(t)
            axWorld.cla(); axWorld.axis('off'); 
            axWorld.imshow(world,'gray',vmin=0,vmax=255,aspect = "equal")
            
            worldFix= np.roll(world,(-np.int8(thInterp(t)*ymap[0] + phiInterp(t)*ymap[1]),-np.int8(thInterp(t)*xmap[0] + phiInterp(t)*xmap[1])),axis = (0,1))
            axWorldFix.imshow(worldFix,'gray',vmin=0, vmax = 255, aspect = 'equal')
            
            #plot line for time, then remove
            ln1 = axTheta.vlines(t,-0.5,30,'b')
            ln2 = axPhi.vlines(t,-0.5,30,'b')
            writer.grab_frame()
            ln1.remove()
            ln2.remove()
        

    max_frames = 60*60
    thInterp =interp1d(eyeT,th, bounds_error = False, fill_value = 0)
    phiInterp =interp1d(eyeT,phi, bounds_error = False, fill_value = 0)

    world_fix = np.zeros((max_frames, np.size(world_vid,1), np.size(world_vid,2)),'uint8')
    for f in tqdm(range(max_frames)):
        t = worldT[f]
        thInt = thInterp(t)
        if np.isnan(thInt):
            thInt =0
        phiInt = phiInterp(t) 
        if np.isnan(phiInt):
            phiInt =0
                
        world_fix[f,:,:]= imshift(world_vid[f,:,:],(-(thInt*ymap[0] + phiInt*ymap[1]),-(thInt*xmap[0] + phiInt*xmap[1])))

    number_of_iterations = 5000
    termination_eps = 1e-4
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)
    warp_mode = cv2.MOTION_TRANSLATION
    cc_fix = np.zeros(max_frames); xshift_fix = np.zeros(max_frames); yshift_fix = np.zeros(max_frames);
    for i in tqdm(range(max_frames-1)):
        try:
            warp_matrix = np.eye(2, 3, dtype=np.float32)
            (cc_fix[i], warp_matrix) = cv2.findTransformECC (world_fix[i,:,:],world_fix[i+1,:,:],warp_matrix, warp_mode, criteria, inputMask = None, gaussFiltSize = 1)
            xshift_fix[i] = warp_matrix[0,2]; yshift_fix[i] = warp_matrix[1,2]
        except:
            xshift_fix[i] = np.nan; yshift_fix[i] = np.nan # very rarely, a frame will raise cv2 error when iterations do not converge for transform

    if free_move:
        plt.figure()
        plt.subplot(2,1,1)
        plt.plot(xshift,label = 'x pre alignment')
        plt.plot(xshift_fix,label = 'x post alignement')
        plt.ylim(-5,5); plt.xlim(1000,1500)

        plt.subplot(2,1,2)
        plt.plot(yshift,label = 'y pre alignment')
        plt.plot(yshift_fix, label = 'y post alignement')
        plt.ylim(-5,5); plt.xlim(1000,1500)

        diagnostic_pdf.savefig()
        plt.close()

    max_frame = 60*60
    flow = np.zeros((max_frame, np.size(world_vid,1), np.size(world_vid,2),2))
    flow_fix = np.zeros((max_frame, np.size(world_vid,1), np.size(world_vid,2),2))
    x,y = np.meshgrid(np.arange(0, np.size(world_vid,2)), np.arange(0,np.size(world_vid,1)))
    vidfile = os.path.join(file_dict['save'], (file_dict['name']+'_flowfix.mp4'))

    print('plotting video of optical flow')
    fig, axs = plt.subplots(1,2,figsize = (16,8))
    # now animate
    writer = FFMpegWriter(fps=30)
    nx = 5
    with writer.saving(fig, vidfile, 100):
        for f in tqdm(range(max_frame-1)):

            flow[f,:,:,:] = cv2.calcOpticalFlowFarneback(world_vid[f,:,:],world_vid[f+1,:,:], None, 0.5, 3, 15, 3, 5, 1.2, 0)
            axs[0].cla()
            axs[0].imshow(world_vid[f,:,:],vmin = 0, vmax = 255)
            u = flow[f,:,:,0]; v = flow[f,:,:,1]
            axs[0].quiver(x[::nx,::nx],y[::nx,::nx],u[::nx,::nx],-v[::nx,::nx], scale = 100 )
            
            flow_fix[f,:,:,:] = cv2.calcOpticalFlowFarneback(world_fix[f,:,:],world_fix[f+1,:,:], None, 0.5, 3, 15, 3, 5, 1.2, 0)
            axs[1].cla()
            axs[1].imshow(world_fix[f,:,:],vmin = 0, vmax = 255)
            u = flow_fix[f,:,:,0]; v = flow[f,:,:,1]
            axs[1].quiver(x[::nx,::nx],y[::nx,::nx],u[::nx,::nx],-v[::nx,::nx], scale = 100 )
            
            writer.grab_frame()