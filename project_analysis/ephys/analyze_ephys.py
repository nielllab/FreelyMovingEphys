"""
analyze_ephys.py

make ephys figures
called by analysis jupyter notebook

Jan. 20, 2021
"""
# package imports
from netCDF4 import Dataset
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import pickle
import time
import subprocess
from matplotlib.animation import FFMpegWriter
import matplotlib as mpl 
import wavio
mpl.rcParams['animation.ffmpeg_path'] = r'C:\Program Files\ffmpeg\bin\ffmpeg.exe'
from scipy.interpolate import interp1d
from numpy import nan
from matplotlib.backends.backend_pdf import PdfPages
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
from util.aux_funcs import nanxcorr
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
from scipy.ndimage import shift as imshift
from scipy import signal
from sklearn.cluster import KMeans
# module imports
from project_analysis.ephys.ephys_figures import *

def find_files(rec_path, rec_name, free_move, cell, stim_type):
    print('find ephys files')

    # get the files names in the provided path
    eye_file = os.path.join(rec_path, rec_name + '_Reye.nc')
    world_file = os.path.join(rec_path, rec_name + '_world.nc')
    ephys_file = os.path.join(rec_path, rec_name + '_ephys_merge.json')
    imu_file = os.path.join(rec_path, rec_name + '_imu.nc')
    speed_file = os.path.join(rec_path, rec_name + '_speed.nc')

    if stim_type == 'gratings':
        stim_type = 'grat'
    elif stim_type == 'white_noise':
        pass
    elif stim_type == 'sparse_noise':
        pass
    else:
        stim_type = None

    if free_move is True:
        dict_out = {'cell':cell,'eye':eye_file,'world':world_file,'ephys':ephys_file,'speed':None,'imu':imu_file,'save':rec_path,'name':rec_name,'stim_type':stim_type}
    elif free_move is False:
        dict_out = {'cell':cell,'eye':eye_file,'world':world_file,'ephys':ephys_file,'speed':speed_file,'imu':None,'save':rec_path,'name':rec_name,'stim_type':stim_type}

    return dict_out

def run_ephys_analysis(file_dict):

    if file_dict['speed'] is None:
        free_move = True; has_imu = True; has_mouse = True
    else:
        free_move = False; has_imu = False; has_mouse = True

    print('opening pdfs')
    # three pdf outputs will be saved
    overview_pdf = PdfPages(os.path.join(file_dict['save'], (file_dict['name'] + '_overview_analysis_figures.pdf')))
    detail_pdf = PdfPages(os.path.join(file_dict['save'], (file_dict['name'] + '_detailed_analysis_figures.pdf')))
    diagnostic_pdf = PdfPages(os.path.join(file_dict['save'], (file_dict['name'] + '_diagnostic_analysis_figures.pdf')))

    print('opening data')
    # load worldcam
    world_data = xr.open_dataset(file_dict['world'])
    world_vid_raw = np.uint8(world_data['WORLD_video'])

    # resize worldcam to make more manageable
    sz = world_vid_raw.shape
    downsamp = 0.5
    world_vid = np.zeros((sz[0],np.int(sz[1]*downsamp),np.int(sz[2]*downsamp)), dtype = 'uint8')
    for f in range(sz[0]):
        world_vid[f,:,:] = cv2.resize(world_vid_raw[f,:,:],(np.int(sz[2]*downsamp),np.int(sz[1]*downsamp)))
    worldT = world_data.timestamps.copy()

    # plot worldcam timing
    worldcam_fig = plot_cam_time(worldT, 'world')
    diagnostic_pdf.savefig()
    plt.close()

    # plot mean world image
    plt.figure()
    plt.imshow(np.mean(world_vid,axis=0)); plt.title('mean world image')
    diagnostic_pdf.savefig()
    plt.close()

    # load IMU data
    if file_dict['imu'] is not None:
        imu_data = xr.open_dataset(file_dict['imu'])
        accT = imu_data.timestamps
        acc_chans = imu_data.IMU_data
        gx = np.array(acc_chans.sel(channel='gyro_x'))
        gy = np.array(acc_chans.sel(channel='gyro_y'))
        gz = np.array(acc_chans.sel(channel='gyro_z'))

    # load optical mouse data
    if file_dict['speed'] is not None:
        speed_data = xr.open_dataset(file_dict['speed'])
        spdVals = speed_data.BALL_data
        try:
            spd = spdVals.sel(move_params = 'cm_per_sec')
            spd_tstamps = spdVals.sel(move_params = 'timestamps')
        except:
            spd = spdVals.sel(frame = 'cm_per_sec')
            spd_tstamps = spdVals.sel(frame = 'timestamps')


    # read ephys data
    ephys_data = pd.read_json(file_dict['ephys'])
    ephys_data['spikeTraw'] = ephys_data['spikeT']

    # select good cells from phy2
    goodcells = ephys_data.loc[ephys_data['group']=='good']
    goodcells.shape
    units = goodcells.index.values

    # get number of good units
    n_units = len(goodcells)

    # spike rasters
    spikeraster_fig = plot_spike_rasters(goodcells)
    detail_pdf.savefig()
    plt.close()

    # load eye data
    eye_data = xr.open_dataset(file_dict['eye'])
    eye_vid = np.uint8(eye_data['REYE_video'])
    eyeT = eye_data.timestamps.copy()

    # plot eye timestamps
    reyecam_fig = plot_cam_time(worldT, 'reye')
    diagnostic_pdf.savefig()
    plt.close()

    # plot eye postion across recording
    eye_params = eye_data['REYE_ellipse_params']
    eyepos_fig = plot_eye_pos(eye_params)
    detail_pdf.savefig()
    plt.close()

    if file_dict['speed'] is not None:
        # plot optical mouse speeds
        optical_mouse_sp_fig = plot_optmouse_spd(spd_tstamps, spd)
        detail_pdf.savefig()
        plt.close()

    # adjust eye/world/top times relative to ephys
    ephysT0 = ephys_data.iloc[0,12]
    eyeT = eye_data.timestamps  - ephysT0
    if eyeT[0]<-600:
        eyeT = eyeT + 8*60*60 # 8hr offset for some data
    worldT = world_data.timestamps - ephysT0
    if worldT[0]<-600:
        worldT = worldT + 8*60*60
    if free_move is True and has_imu is True:
        accTraw = imu_data.timestamps-ephysT0
    if free_move is False and has_mouse is True:
        speedT = spd_tstamps-ephysT0

    # check that deinterlacing worked correctly
    # plot theta and theta switch
    # want theta switch to be jagged, theta to be smooth
    theta_switch_fig, th_switch = plot_param_switch_check(eye_params)
    diagnostic_pdf.savefig()
    plt.close()

    # plot eye variables
    eye_param_fig = plot_eye_params(eye_params, eyeT)
    detail_pdf.savefig()
    plt.close()

    # calculate eye veloctiy
    dEye = np.diff(np.rad2deg(eye_params.sel(ellipse_params='theta')))

    print('checking accelerometer / eye temporal alignment')
    # check accelerometer / eye temporal alignment
    if file_dict['imu'] is not None:
        lag_range = np.arange(-0.2,0.2,0.002)
        cc = np.zeros(np.shape(lag_range))
        t1 = np.arange(5,1600,20)
        t2 = t1 + 60
        offset = np.zeros(np.shape(t1))
        ccmax = np.zeros(np.shape(t1))
        acc_interp = interp1d(accTraw, (gz-3)*7.5)
        for tstart in range(len(t1)):
            for l in range(len(lag_range)):
                c, lag= nanxcorr(dEye[t1[tstart]*60 : t2[tstart]*60] + 0.5/60, acc_interp(eyeT[t1[tstart]*60:t2[tstart]*60]+lag_range[l]),1)
                cc[l] = c[1]
            offset[tstart] = lag_range[np.argmax(cc)]    
            ccmax[tstart] = np.max(cc)
        offset[ccmax<0.1] = nan
        plt.subplot(1,2,1)
        plt.plot(eyeT[t1*60],offset)
        plt.xlabel('secs'); plt.ylabel('offset - secs')
        plt.subplot(1,2,2)
        plt.plot(eyeT[t1*60],ccmax)
        plt.xlabel('secs'); plt.ylabel('max cc')
        diagnostic_pdf.savefig()
        plt.close()

    print('fitting regression to timing drift')
    # fit regression to timing drift
    if file_dict['imu'] is not None:
        model = LinearRegression()
        dataT = np.array(eyeT[t1*60 + 30])
        model.fit(dataT[offset>0].reshape(-1,1),offset[offset>0])
        offset0 = model.intercept_
        drift_rate = model.coef_
        plt.plot(dataT,offset,'.')
        plt.plot(dataT, offset0 + dataT*drift_rate)
        plt.xlabel('secs'); plt.ylabel('offset - secs')
        print(offset0)
        print(drift_rate)
    elif file_dict['speed'] is not None:
        offset0 = 0.1
        drift_rate = 0.1/1000

    if file_dict['imu'] is not None:
        accT = accTraw - (offset0 + accTraw*drift_rate)

    for i in range(len(ephys_data)):
        ephys_data['spikeT'].iloc[i] = np.array(ephys_data['spikeTraw'].iloc[i]) - (offset0 + np.array(ephys_data['spikeTraw'].iloc[i]) *drift_rate)

    print('finding contrast of normalized worldcam')
    # normalize world movie and calculate contrast
    cam_gamma = 2
    world_norm = (world_vid/255)**cam_gamma
    std_im = np.std(world_norm,axis=0)
    std_im[std_im<10/255] = 10/255
    img_norm = (world_norm-np.mean(world_norm,axis=0))/std_im

    contrast = np.empty(worldT.size)
    for i in range(worldT.size):
        contrast[i] = np.std(img_norm[i,:,:])
    plt.plot(worldT[0:6000],contrast[0:6000])
    plt.xlabel('time')
    plt.ylabel('contrast')
    diagnostic_pdf.savefig()
    plt.close()

    fig = plt.figure()
    plt.imshow(std_im)
    plt.colorbar(); plt.title('std img')
    diagnostic_pdf.savefig()
    plt.close()

    # set up interpolators for eye and world videos
    eyeInterp = interp1d(eyeT,eye_vid,axis=0)
    worldInterp = interp1d(worldT,world_vid,axis=0)

    # make movie and sound
    print('making video figure')
    this_unit = file_dict['cell']

    if file_dict['imu'] is not None:
        vidfile = make_movie(file_dict, eyeT, worldT, eye_vid, world_vid, contrast, eye_params, dEye, goodcells, units, this_unit, eyeInterp, worldInterp, accT=accT, gz=gz)
    elif file_dict['speed'] is not None:
        vidfile = make_movie(file_dict, eyeT, worldT, eye_vid, world_vid, contrast, eye_params, dEye, goodcells, units, this_unit, eyeInterp, worldInterp, speedT=speedT, spd=spd)

    print('making audio figure')
    audfile = make_sound(file_dict, ephys_data, units, this_unit)
    
    # merge video and audio
    merge_mp4_name = os.path.join(file_dict['save'], (file_dict['name']+'_unit'+str(this_unit)+'_merge.mp4'))

    print('merging movie with sound')
    subprocess.call(['ffmpeg', '-i', vidfile, '-i', audfile, '-c:v', 'copy', '-c:a', 'aac', merge_mp4_name]) 

    if free_move is True and file_dict['imu'] is not None:
        plt.figure()
        plt.plot(eyeT[0:-1],np.diff(th_switch),label = 'dTheta')
        plt.plot(accT-0.1,(gz-3)*10, label = 'gyro')
        plt.xlim(30,40); plt.ylim(-12,12); plt.legend(); plt.xlabel('secs')
        diagnostic_pdf.savefig()
        plt.close()

    th = np.array((eye_params.sel(ellipse_params = 'theta')-np.nanmean(eye_params.sel(ellipse_params = 'theta')))*180/3.14159)
    phi = np.array((eye_params.sel(ellipse_params = 'phi')-np.nanmean(eye_params.sel(ellipse_params = 'phi')))*180/3.14159)
    
    if free_move:
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

    else:
        xmap = [-0.080764229, -0.075781153]
        ymap =[-0.076365844,  0.083263225]

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
        warp_matrix = np.eye(2, 3, dtype=np.float32)
        (cc_fix[i], warp_matrix) = cv2.findTransformECC (world_fix[i,:,:],world_fix[i+1,:,:],warp_matrix, warp_mode, criteria, inputMask = None, gaussFiltSize = 1)
        xshift_fix[i] = warp_matrix[0,2]; yshift_fix[i] = warp_matrix[1,2]

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

    dEye = np.diff(th)
    if free_move and file_dict['imu'] is not None:
        gInterp = interp1d(accT,(gz-np.nanmean(gz))*7.5 , bounds_error = False)
        plt.figure(figsize = (8,4))
        plt.subplot(1,2,1)
        plt.plot(eyeT[0:-1],dEye, label = 'dEye')
        plt.plot(eyeT, gInterp(eyeT), label = 'dHead')
        #plt.plot(accT-0.11,(gz-np.nanmean(gz))*7 )
        #plt.plot(eyeT[0:-1],gInterp(eyeT[0:-1])-dEye, label = 'dgaze')
        plt.xlim(37,39); plt.ylim(-10,10); plt.legend(); plt.ylabel('deg'); plt.xlabel('secs')
        plt.subplot(1,2,2)
        plt.plot(eyeT[0:-1],np.nancumsum(gInterp(eyeT[0:-1])), label = 'head')
        plt.plot(eyeT[0:-1],np.nancumsum(gInterp(eyeT[0:-1])-dEye),label ='gaze')
        plt.plot(eyeT[1:],th[0:-1],label ='eye')
        plt.xlim(35,40); plt.ylim(-30,30); plt.legend(); plt.ylabel('deg'); plt.xlabel('secs')
        plt.tight_layout()
        diagnostic_pdf.savefig()
        plt.close()
    
    plt.subplot(1,2,1)
    plt.imshow(std_im)
    plt.title('std dev of image')
    plt.subplot(1,2,2)
    plt.imshow(np.mean(world_vid,axis=0),vmin=0,vmax=255)
    plt.title('mean of image')
    diagnostic_pdf.savefig()
    plt.close()

    # set up timebase for subsequent analysis
    dt = 0.025
    t = np.arange(0, np.max(worldT),dt)

    # interpolate and plot contrast
    newc = interp1d(worldT,contrast)
    contrast_interp = newc(t[0:-1])
    contrast_interp.shape
    plt.plot(t[0:600],contrast_interp[0:600])
    plt.xlabel('secs'); plt.ylabel('world contrast')
    diagnostic_pdf.savefig()
    plt.close()

    print('calculating firing rate')
    # calculate firing rate at new timebase
    ephys_data['rate'] = nan
    ephys_data['rate'] = ephys_data['rate'].astype(object)
    for i,ind in enumerate(ephys_data.index):
        ephys_data.at[ind,'rate'],bins = np.histogram(ephys_data.at[ind,'spikeT'],t)
    ephys_data['rate']= ephys_data['rate']/dt
    goodcells = ephys_data.loc[ephys_data['group']=='good']

    print('calculating contrast reponse functions')
    # calculate contrast - response functions
    # mean firing rate in timebins correponding to contrast ranges
    resp = np.empty((n_units,12))
    crange = np.arange(0,1.2,0.1)
    for i,ind in enumerate(goodcells.index):
        for c,cont in enumerate(crange):
            resp[i,c] = np.mean(goodcells.at[ind,'rate'][(contrast_interp>cont) & (contrast_interp<(cont+0.1))])
    plt.plot(crange,np.transpose(resp))
    #plt.ylim(0,10)
    plt.xlabel('contrast')
    plt.ylabel('sp/sec')
    plt.title('mean firing rate in timebins correponding to contrast ranges')
    detail_pdf.savefig()
    plt.close()

    print('plotting individual contrast response functions')
    # plot individual contrast response functions in subplots
    fig = plt.figure(figsize = (6,np.ceil(n_units/2)))
    for i, ind in enumerate(goodcells.index):
        plt.subplot(np.ceil(n_units/4),4,i+1)
        plt.plot(crange[2:-1],resp[i,2:-1])
    # plt.ylim([0 , max(resp[i,1:-3])*1.2])
        plt.xlabel('contrast a.u.'); plt.ylabel('sp/sec'); plt.ylim([0,np.nanmax(resp[i,2:-1])])
    plt.tight_layout()
    plt.title('individual contrast reponse')
    detail_pdf.savefig()
    plt.close()

    # create interpolator for movie data so we can evaluate at same timebins are firing rate
    img_norm[img_norm<-2] = -2
    movInterp = interp1d(worldT,img_norm,axis=0)

    print('getting spike-triggered average for lag=0.125')
    # calculate spike-triggered average
    spike_corr = 1 + 0.125/1200  # correction factor for ephys timing drift

    staAll = np.zeros((n_units,np.shape(img_norm)[1],np.shape(img_norm)[2]))
    lag = 0.125
    plt.figure(figsize = (12,np.ceil(n_units/2)))
    for c, ind in enumerate(goodcells.index):
        r = goodcells.at[ind,'rate']
        sta = 0; nsp = 0
        sp = goodcells.at[ind,'spikeT'].copy()
        if c==1:
            ensemble = np.zeros((len(sp),np.shape(img_norm)[1],np.shape(img_norm)[2]))
        for s in sp:
            if (s-lag >5) & ((s-lag)*spike_corr <np.max(worldT)):
                nsp = nsp+1
                im = movInterp((s-lag)*spike_corr)
                if c==1:
                    ensemble[nsp-1,:,:] = im
                sta = sta+im
        plt.subplot(np.ceil(n_units/4),4,c+1)
        sta = sta/nsp
        #sta[abs(sta)<0.1]=0
        plt.imshow((sta-np.mean(sta) ),vmin=-0.3,vmax=0.3,cmap = 'jet')
        staAll[c,:,:] = sta
    # plt.title('spike triggered average (lag=0.125)')
    plt.tight_layout()
    detail_pdf.savefig()
    plt.close()


    print('getting grating flow')
    nf = np.size(img_norm,0)-1
    u_mn = np.zeros((nf,1)); v_mn = np.zeros((nf,1))
    sx_mn = np.zeros((nf,1)) ; sy_mn = np.zeros((nf,1))
    flow_norm = np.zeros((nf,np.size(img_norm,1),np.size(img_norm,2),2 ))
    vidfile = os.path.join(file_dict['save'], (file_dict['name']+'_grating_flow'))

    fig, ax = plt.subplots(1,1,figsize = (16,8))
    # now animate
    #writer = FFMpegWriter(fps=30)
    #with writer.saving(fig, vidfile, 100):
    for f in tqdm(range(nf)):
        frm = np.uint8(32*(img_norm[f,:,:]+4))
        frm2 = np.uint8(32*(img_norm[f+1,:,:]+4))
        flow_norm[f,:,:,:] = cv2.calcOpticalFlowFarneback(frm,frm2, None, 0.5, 3, 30, 3, 7, 1.5, 0)
        #ax.cla()
        #ax.imshow(frm,vmin = 0, vmax = 255)
        u = flow_norm[f,:,:,0]; v = -flow_norm[f,:,:,1]  # negative to fix sign for y axis in images
        sx = cv2.Sobel(frm,cv2.CV_64F,1,0,ksize=7)
        sy = -cv2.Sobel(frm,cv2.CV_64F,0,1,ksize=7)# negative to fix sign for y axis in images
        sx[std_im<0.05]=0; sy[std_im<0.05]=0; # get rid of values outside of monitor
        sy[sx<0] = -sy[sx<0]  #make vectors point in positive x direction (so opposite sides of grating don't cancel)
        sx[sx<0] = -sx[sx<0]
        #ax.quiver(x[::nx,::nx],y[::nx,::nx],sx[::nx,::nx],sy[::nx,::nx], scale = 100000 )
        u_mn[f]= np.mean(u); v_mn[f]= np.mean(v); sx_mn[f] = np.mean(sx); sy_mn[f] = np.mean(sy)
        #plt.title(str(np.round(np.arctan2(sy_mn[f],sx_mn[f])*180/np.pi))
        #writer.grab_frame()

    stimOn = contrast>0.5
    stimOn = signal.medfilt(stimOn,11)

    stim_start = np.array(worldT[np.where(np.diff(stimOn)>0)])
    stim_end = np.array(worldT[np.where(np.diff(stimOn)<0)])
    stim_end = stim_end[stim_end>stim_start[0]]
    stim_start = stim_start[stim_start<stim_end[-1]]
    grating_th = np.zeros(len(stim_start))
    grating_mag = np.zeros(len(stim_start))
    grating_dir = np.zeros(len(stim_start))
    for i in range(len(stim_start)):
        stim_u = np.median(u_mn[np.where((worldT>stim_start[i] + 0.025) & (worldT<stim_end[i]-0.025))])
        stim_v = np.median(v_mn[np.where((worldT>stim_start[i] + 0.025) & (worldT<stim_end[i]-0.025))])
        stim_sx = np.median(sx_mn[np.where((worldT>stim_start[i] + 0.025) & (worldT<stim_end[i]-0.025))])
        stim_sy = np.median(sy_mn[np.where((worldT>stim_start[i] + 0.025) & (worldT<stim_end[i]-0.025))])
        grating_th[i] = np.arctan2(stim_sy,stim_sx)
        grating_mag[i] = np.sqrt(stim_sx**2 + stim_sy**2)
        grating_dir[i] = np.sign(stim_u*stim_sx + stim_v*stim_sy) # dot product of gratient and flow gives direction
    #grating_th = np.round(grating_th *10)/10

    grating_ori = grating_th.copy()
    grating_ori[grating_dir<0] = grating_ori[grating_dir<0] + np.pi
    grating_ori = grating_ori - np.min(grating_ori)
    np.unique(grating_ori)
    plt.figure(figsize = (8,8))

    ori_cat = np.floor((grating_ori+np.pi/8)/(np.pi/4))


    km = KMeans(n_clusters=3).fit(np.reshape(grating_mag,(-1,1)))
    sf_cat = km.labels_
    order = np.argsort(np.reshape(km.cluster_centers_, 3))
    sf_catnew = sf_cat.copy()
    for i in range(3):
        sf_catnew[sf_cat == order[i]]=i
    sf_cat = sf_catnew.copy()
    plt.scatter(grating_mag,grating_ori,c=ori_cat)
    detail_pdf.savefig()
    plt.plot()

    print('plotting grading orientation and tuning curves')
    edge_win = 0.025
    grating_rate = np.zeros((len(goodcells),len(stim_start)))
    spont_rate = np.zeros((len(goodcells),len(stim_start)))
    ori_tuning = np.zeros((len(goodcells),8,3))
    drift_spont = np.zeros(len(goodcells))
    plt.figure(figsize = (12,n_units*2))

    for c, ind in enumerate(goodcells.index):
        sp = goodcells.at[ind,'spikeT'].copy()
        for i in range(len(stim_start)):
            grating_rate[c,i] = np.sum((sp> stim_start[i]+edge_win) & (sp<stim_end[i])) / (stim_end[i] - stim_start[i]- edge_win)
        for i in range(len(stim_start)-1):
            spont_rate[c,i] = np.sum((sp> stim_end[i]+edge_win) & (sp<stim_start[i+1])) / (stim_start[i+1] - stim_end[i]- edge_win)  
        for ori in range(8):
            for sf in range(3):
                ori_tuning[c,ori,sf] = np.mean(grating_rate[c,(ori_cat==ori) & (sf_cat ==sf)])
        drift_spont[c] = np.mean(spont_rate[c,:])
        plt.subplot(n_units,2,2*c+1)
        plt.scatter(grating_ori,grating_rate[c,:],c= sf_cat)
        plt.plot(3*np.ones(len(spont_rate[c,:])),spont_rate[c,:],'r.')
        plt.subplot(n_units,2,2*c+2)
        plt.plot(ori_tuning[c,:,0],label = 'low sf'); plt.plot(ori_tuning[c,:,1],label = 'mid sf');plt.plot(ori_tuning[c,:,2],label = 'hi sf')
        plt.plot([0,7],[drift_spont[c],drift_spont[c]],'r:', label = 'spont')
        plt.legend()
        plt.ylim(0,np.nanmax(ori_tuning[c,:,:]*1.2))

    print('getting spike-triggered average with range in lags')
    # calculate spike-triggered average
    spike_corr = 1 + 0.125/1200
    sta = 0
    lag = 0.075
    lagRange = np.arange(0,0.25,0.05)
    plt.figure(figsize = (12,2*n_units))
    for c, ind in enumerate(goodcells.index):
        sp = goodcells.at[ind,'spikeT'].copy()
        for  lagInd, lag in enumerate(lagRange):
            sta = 0; nsp = 0
            for s in sp:
                if (s-lag >5) & ((s-lag)*spike_corr <np.max(worldT)):
                    nsp = nsp+1
                    sta = sta+movInterp((s-lag)*spike_corr)
            plt.subplot(n_units,6,(c*6)+lagInd + 1)
            sta = sta/nsp
        #sta[abs(sta)<0.1]=0
            plt.imshow(sta ,vmin=-0.35,vmax=0.35,cmap = 'jet')
            plt.title(str(c) + ' ' + str(np.round(lag*1000)) + 'msec')
    plt.tight_layout()
    detail_pdf.savefig()
    plt.close()

    print('getting spike-triggered variance')
    # calculate spike-triggered variance
    sta = 0
    lag = 0.125
    plt.figure(figsize = (12,np.ceil(n_units/2)))
    for c, ind in enumerate(goodcells.index):
        r = goodcells.at[ind,'rate']
        sta = 0
        for i in range(5,t.size-10):
            sta = sta+r[i]*(movInterp(t[i]-lag))**2
        plt.subplot(np.ceil(n_units/4),4,c+1)
        sta = sta/np.sum(r)
        plt.imshow(sta - np.mean(img_norm**2,axis=0),vmin=-1,vmax=1)
    plt.tight_layout()
    detail_pdf.savefig()
    plt.close()

    print('getting rasters around saccades')
    # calculate saccade-locked psth
    # spike_corr = 1 + 0.125/1200  # correction factor for ephys timing drift
    dEye= np.diff(th_switch)

    fig = plt.figure(figsize = (12,np.ceil(n_units/2)))
    trange = np.arange(-1,1.1,0.1)
    sthresh = 8
    upsacc = eyeT[np.append(dEye,0)>sthresh]/spike_corr
    upsacc = upsacc[upsacc>5]
    upsacc = upsacc[upsacc<np.max(t)-5]
    downsacc= eyeT[np.append(dEye,0)<-sthresh]/spike_corr
    downsacc = downsacc[downsacc>5]
    downsacc = downsacc[downsacc<np.max(t)-5]
    upsacc_avg = np.zeros((units.size,trange.size))
    downsacc_avg = np.zeros((units.size,trange.size))
    for i, ind in enumerate(goodcells.index):
        rateInterp = interp1d(t[0:-1],goodcells.at[ind,'rate'])
        for s in upsacc:
            upsacc_avg[i,:] = upsacc_avg[i,:]+ rateInterp(np.array(s)+trange)/upsacc.size
        for s in downsacc:
            downsacc_avg[i,:]= downsacc_avg[i,:]+ rateInterp(np.array(s)+trange)/downsacc.size
        plt.subplot(np.ceil(n_units/4),4,i+1)
        plt.plot(trange,upsacc_avg[i,:])
        plt.plot(trange,downsacc_avg[i,:],'r')
        plt.vlines(0,0,np.max(upsacc_avg[i,:]*0.2),'r')
        plt.ylim([0, np.max(upsacc_avg[i,:])*1.8])
        plt.ylabel('sp/sec')
    plt.tight_layout()
    detail_pdf.savefig()
    plt.close()

    # rasters around positive saccades
    fig = plt.figure(figsize = (12,n_units)) 
    for i, ind in enumerate(goodcells.index): 
        sp = np.array(goodcells.at[units[i],'spikeT']) *spike_corr
        plt.subplot(np.ceil(n_units/4),4,i+1) 
        n = 0 
        for s in upsacc: 
            n= n+1 
            sd = np.abs(sp-np.array(s))<10 
            sacc_sp = sp[sd] 
            plt.vlines(sacc_sp-np.array(s),n-0.25,n+0.25) 
        plt.xlim(-1,1) #; plt.ylim(0,50)
    detail_pdf.savefig()
    plt.close()

    #rasters around negative saccades
    fig = plt.figure(figsize = (12,n_units))
    for i, ind in enumerate(goodcells.index):
        sp = np.array(goodcells.at[units[i],'spikeT'])
        plt.subplot(np.ceil(n_units/4),4,i+1)
        n = 0
        for s in downsacc:
            n= n+1
            sd = np.abs(sp-np.array(s))<10
            sacc_sp = sp[sd]
            plt.vlines(sacc_sp-np.array(s),n-0.25,n+0.25)
        plt.xlim(-1,1)
    detail_pdf.savefig()
    plt.close()

    # normalize and plot eye radius
    eyeR = eye_params.sel(ellipse_params = 'longaxis').copy()
    Rnorm = (eyeR - np.mean(eyeR))/np.std(eyeR)
    plt.plot(eyeT,Rnorm)
    plt.xlim([0,60])
    plt.xlabel('secs')
    plt.ylabel('normalized pupil R')
    diagnostic_pdf.savefig()
    plt.close()

    print('plotting rate vs pupil radius')
    # plot rate vs pupil
    n_units = len(goodcells)
    R_range = np.arange(-4,4,0.5)
    useEyeT = eyeT[(eyeT<t[-2]) & (eyeT>t[0])].copy()
    useR = Rnorm[(eyeT<t[-2]) & (eyeT>t[0])].copy()
    R_scatter = np.zeros((n_units,len(useR)))
    R_tuning = np.zeros((n_units,len(R_range)-1))
    R_tuning_err =R_tuning.copy()
    for i, ind in enumerate(goodcells.index):
        rateInterp = interp1d(t[0:-1],goodcells.at[ind,'rate'])
        R_scatter[i,:] = rateInterp(useEyeT)
        for j in range(len(R_range)-1):
            usePts =(useR>R_range[j]) & (useR<R_range[j+1])
            R_tuning[i,j] = np.mean(R_scatter[i,usePts])
            R_tuning_err[i,j] = np.std(R_scatter[i,usePts])/np.sqrt(np.count_nonzero(usePts))
    fig = plt.figure(figsize = (12,np.ceil(n_units/2)))
    for i in range(n_units):
        plt.subplot(np.ceil(n_units/4),4,i+1)
        plt.errorbar(R_range[:-1],R_tuning[i,:],yerr=R_tuning_err[i,:])
        plt.ylim(0,np.nanmax(R_tuning[i,2:-2]*1.2))
        plt.xlim([-2, 2])
        plt.xlabel('normalized pupil R'); plt. ylabel('sp/sec'); plt.title(i)
    plt.tight_layout()
    detail_pdf.savefig()
    plt.close()

    # normalize eye position
    eyeTheta = eye_params.sel(ellipse_params = 'theta').copy()
    thetaNorm = (eyeTheta - np.mean(eyeTheta))/np.std(eyeTheta)
    plt.plot(eyeT[0:3600],thetaNorm[0:3600])
    plt.xlabel('secs'); plt.ylabel('normalized eye theta')
    diagnostic_pdf.savefig()
    plt.close()

    print('plotting rate vs theta')
    # plot rate vs theta
    n_units = len(goodcells)
    th_range = np.arange(-2,3,0.5)
    useEyeT = eyeT[(eyeT<t[-2]) & (eyeT>t[0])].copy()
    useTh = thetaNorm[(eyeT<t[-2]) & (eyeT>t[0])].copy()
    th_scatter = np.zeros((n_units,len(useR)))
    th_tuning = np.zeros((n_units,len(th_range)-1))
    th_tuning_err =th_tuning.copy()
    for i, ind in enumerate(goodcells.index):
        rateInterp = interp1d(t[0:-1],goodcells.at[ind,'rate'])
        th_scatter[i,:] = rateInterp(useEyeT)
        for j in range(len(th_range)-1):
            usePts =(useTh>th_range[j]) & (useTh<th_range[j+1])
            th_tuning[i,j] = np.mean(th_scatter[i,usePts])
            th_tuning_err[i,j] = np.std(th_scatter[i,usePts])/np.sqrt(np.count_nonzero(usePts))
    fig = plt.figure(figsize = (3*np.ceil(n_units/2),6))
    for i in range(n_units):
        plt.subplot(2,np.ceil(n_units/2),i+1)
        plt.errorbar(th_range[:-1],th_tuning[i,:],yerr=th_tuning_err[i,:])
        plt.ylim(0,np.nanmax(th_tuning[i,:]*1.2))
        plt.xlim([-2, 2])
        plt.xlabel('normalized pupil theta'); plt. ylabel('sp/sec'); plt.title(i)
    plt.tight_layout()
    detail_pdf.savefig()
    plt.close()

    print('generating summary plot')
    # generate summary plot
    samprate = 30000  # ephys sample rate
    plt.figure(figsize = (12,np.ceil(n_units)*2))
    for i, ind in enumerate(goodcells.index): 
        # plot waveform
        plt.subplot(n_units,4,i*4 + 1)
        wv = goodcells.at[ind,'waveform']
        plt.plot(np.arange(len(wv))*1000/samprate,goodcells.at[ind,'waveform'])
        plt.xlabel('msec'); plt.title(str(i) + ' ' + goodcells.at[ind,'KSLabel']  +  ' cont='+ str(goodcells.at[ind,'ContamPct']))
        
        # plot CRF
        plt.subplot(n_units,4,i*4 + 2)
        plt.plot(crange[2:-1],resp[i,2:-1])
        plt.xlabel('contrast a.u.'); plt.ylabel('sp/sec'); plt.ylim([0,np.nanmax(resp[i,2:-1])])
                                    
        #plot STA or tuning curve
        plt.subplot(n_units,4,i*4 + 3)
        if file_dict['stim_type'] == 'grat':
            plt.plot(np.arange(8)*45, ori_tuning[i,:,0],label = 'low sf'); plt.plot(np.arange(8)*45,ori_tuning[i,:,1],label = 'mid sf');plt.plot(np.arange(8)*45,ori_tuning[i,:,2],label = 'hi sf')
            plt.plot([0,315],[drift_spont[i],drift_spont[i]],'r:', label = 'spont')
        # plt.legend()
            plt.ylim(0,np.nanmax(ori_tuning[i,:,:]*1.2)); plt.xlabel('orientation (deg)')
        else:
            sta = staAll[i,:,:]
            staRange = np.max(np.abs(sta))*1.2
            if staRange<0.25:
                staRange=0.25
            plt.imshow(staAll[i,:,:],vmin = -staRange, vmax= staRange, cmap = 'jet')
                      
        # plot eye movements
        plt.subplot(n_units,4,i*4 + 4)
        plt.plot(trange,upsacc_avg[i,:])
        plt.plot(trange,downsacc_avg[i,:],'r')
        plt.vlines(0,0,np.max(upsacc_avg[i,:]*0.2),'r')
        plt.ylim([0, np.max(upsacc_avg[i,:])*1.8])
        plt.ylabel('sp/sec')
    plt.tight_layout()
    overview_pdf.savefig()
    plt.close()

    hist_dt = 1
    hist_t = np.arange(0, np.max(worldT),hist_dt)
    plt.figure(figsize = (12,n_units*2))
    plt.subplot(n_units+3,1,1)
    if has_imu:
        plt.plot(accT,gz)
        plt.xlim(0, np.max(worldT)); plt.ylabel('gz'); plt.title('gyro')
    elif has_mouse:
        plt.plot(speedT,spd)
        plt.xlim(0, np.max(worldT)); plt.ylabel('cm/sec'); plt.title('mouse speed')  

    plt.subplot(n_units+3,1,2)
    plt.plot(eyeT,eye_params.sel(ellipse_params = 'longaxis'))
    plt.xlim(0, np.max(worldT)); plt.ylabel('rad (pix)'); plt.title('pupil diameter')

    plt.subplot(n_units+3,1,3)
    plt.plot(worldT,contrast)
    plt.xlim(0, np.max(worldT)); plt.ylabel('contrast a.u.'); plt.title('contrast')

    for i,ind in enumerate(goodcells.index):
        rate,bins = np.histogram(ephys_data.at[ind,'spikeT'],hist_t)
        plt.subplot(n_units+3,1,i+4)
        plt.plot(bins[0:-1],rate)
        plt.xlabel('secs')
        plt.ylabel('sp/sec'); plt.xlim(bins[0],bins[-1]); plt.title('unit ' + str(i))
    plt.tight_layout()
    detail_pdf.savefig()
    plt.close()

    overview_pdf.close(); detail_pdf.close(); diagnostic_pdf.close()

    print('organizing data and saving as xarray')

    split_base_name = file_dict['name'].split('_')

    date = split_base_name[0]; mouse = split_base_name[1]; exp = split_base_name[2]; rig = split_base_name[3]
    try:
        stim = '_'.join(split_base_name[4:])
    except:
        stim = split_base_name[4:]
    
    unit_names = [(file_dict['name']+'_unit'+str(i)) for i in range(1,n_units+1)]
    ephys_params_names = ['contrast_range','orientation_tuning','contrast_response','STA','waveform','trange','upsacc_avg','downsacc_avg']
    for unit_num in range(n_units):
        unit = unit_num+1
        unit_xr = xr.DataArray([crange,ori_tuning[unit_num],resp[unit_num],staAll[unit_num],goodcells.at[unit_num,'waveform'],trange,upsacc_avg[unit_num],downsacc_avg[unit_num]], dims=['ephys_params'], coords=[('ephys_params', ephys_params_names)])
        unit_xr.attrs['date'] = date; unit_xr.attrs['mouse'] = mouse; unit_xr.attrs['exp'] = exp; unit_xr.attrs['rig'] = rig; unit_xr.attrs['stim'] = stim; unit_xr.attrs['unit_id'] = unit_names[0]; unit_xr.attrs['unit'] = unit
        if unit_num == 0:
            all_units_xr = unit_xr
        else:
            all_units_xr = xr.merge([all_units_xr, unit_xr])

    all_units_xr.to_netcdf(os.path.join(file_dict['save'], (file_dict['name']+'_ephys_props.nc')))

    print('analysis complete; pdfs closed and xarray saved to file')
