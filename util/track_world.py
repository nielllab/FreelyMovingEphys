"""
track_world.py

Utilities for tracking world camera and finding pupil rotation

Last modified September 14, 2020
"""

# package imports
import xarray as xr
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
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

# module imports
from util.read_data import open_time, find, nanxcorr

# time formatting so that timedelta can be plotted
def format_func(x, pos):
    hours = int(x//3600)
    minutes = int((x%3600)//60)
    seconds = int(x%60)
    return "{:d}:{:02d}:{:02d}".format(hours, minutes, seconds)
formatter = FuncFormatter(format_func)

# basic world shifting
def adjust_world(data_path, file_name, eyeext, topext, worldext, eye_ds, savepath):
    # get eye data out of dataset
    eye_pts = eye_data.raw_pt_values
    eye_ell_params = eye_data.ellipse_param_values

    # find the needed files from path and trial key
    top1vidpath = os.path.join(data_path, file_name) + '_' + topext + '.avi'
    eyevidpath = os.path.join(data_path, file_name) + '_' + eyeext + '.avi'
    worldvidpath = os.path.join(data_path, file_name) + '_' + worldext + '.avi'
    top1timepath = os.path.join(data_path, file_name) + '_' + topext +'_BonsaiTS.csv'
    eyetimepath = os.path.join(data_path, file_name) + '_' + eyeext +'_BonsaiTS.csv'
    worldtimepath = os.path.join(data_path, file_name) + '_' + worldext +'_BonsaiTS.csv'

    # create save directory if it does not already exist
    fig_dir = savepath + '/' + file_name + '/'
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    # open time files
    eyeTS = open_time(eyetimepath)
    worldTS = open_time(worldtimepath)
    topTS = open_time(top1timepath)

    # interpolate ellipse parameters to worldcam timestamps
    eye_ell_interp_params = eye_ell_params.interp_like(xr.DataArray(worldTS), method=interp_method)

    # the very first timestamp
    start_time = min(eyeTS[0], worldTS[0], topTS[0])

    eye_theta = eye_ell_interp_params.sel(ellipse_params='theta')
    eye_phi = eye_ell_interp_params.sel(ellipse_params='phi')
    eye_longaxis= eye_ell_interp_params.sel(ellipse_params='longaxis')
    eye_shortaxis = eye_ell_interp_params.sel(ellipse_params='shortaxis')

    eye_raw_theta = eye_ell_params.sel(ellipse_params='theta')
    eye_raw_phi = eye_ell_params.sel(ellipse_params='phi')
    eye_raw_longaxis= eye_ell_params.sel(ellipse_params='longaxis')
    eye_raw_shortaxis = eye_ell_params.sel(ellipse_params='shortaxis')

    eyeTSminusstart = [(t-start_time).seconds for t in eyeTS]
    worldTSminusstart = [(t-start_time).seconds for t in worldTS]

    # saftey check
    plt.subplots(2, 1, figsize=(15, 15))
    plt.subplot(211)
    plt.title('raw/interpolated theta for ' + eyeext + ' side')
    plt.plot(eyeTSminusstart, eye_raw_theta.values, 'r--', label='raw theta')
    plt.plot(worldTSminusstart[:-1], eye_theta.values, 'b-', label='interp theta')
    plt.subplot(212)
    plt.title('raw/interpolated phi for ' + eyeext + ' side')
    plt.plot(eyeTSminusstart, eye_raw_phi.values, 'r--', label='raw phi')
    plt.plot(worldTSminusstart[:-1], eye_phi.values, 'b-', label='interp phi')
    plt.savefig(fig_dir + eyeext + 'rawinterp_phitheta.png', dpi=300)
    plt.close()

    worldvid = cv2.VideoCapture(worldvidpath)
    topvid = cv2.VideoCapture(top1vidpath)
    eyevid = cv2.VideoCapture(eyevidpath)

    # setup the file to save out of this
    savepath = os.path.join(fig_dir, str(file_name + '_worldshift_' + eyeext + '.avi'))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out_vid = cv2.VideoWriter(savepath, fourcc, 20.0, (int(eyevid.get(cv2.CAP_PROP_FRAME_WIDTH))*2, int(eyevid.get(cv2.CAP_PROP_FRAME_HEIGHT))*2))

    set_size = (int(eyevid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(eyevid.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    while(1):
        # read the frame for this pass through while loop
        wrld_ret, wrld_frame = worldvid.read()
        eye_ret, eye_frame = eyevid.read()
        top_ret, top_frame = topvid.read()

        if not wrld_ret:
            break
        if not eye_ret:
            break
        if not top_ret:
            break

        # create empty frame to shift the world in
        wrld_shift = np.zeros(set_size)

        # limit range to shift over
        phi_max = np.where(eye_phi > 15, -15, 15)
        theta_max = np.where(eye_theta > 20, -20, 20)

        # insert world frame into world shift with offset
        if np.isnan(eye_theta) is False and np.isnan(eye_phi) is False:
            wrld_shift[(range(61,180) - np.round(phi_max * pix_deg)), (range(81,240) - np.round(theta_max * pix_deg))] = wrld_frame

        # resize the frames before plotting
        wrld_frame_resz = cv2.resize(wrld_frame, set_size)
        wrld_shift_resz = cv2.resize(np.uint8(wrld_shift), set_size)
        eye_frame_resz = cv2.resize(eye_frame, set_size)
        top_frame_resz = cv2.resize(top_frame, set_size)

        # concat frames together into a 2x2 grid
        a = np.concatenate((cv2.cvtColor(eye_frame_resz, cv2.COLOR_BGR2GRAY), cv2.cvtColor(top_frame_resz, cv2.COLOR_BGR2GRAY)), axis=1)
        b = np.concatenate((cv2.cvtColor(wrld_frame_resz, cv2.COLOR_BGR2GRAY), wrld_shift_resz), axis=1)
        all_vids = np.concatenate((a, b), axis=0)

        out_vid.write(all_vids)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    out_vid.release()
    cv2.destroyAllWindows()

# sigmoid function
def curve_func(xval, a, b, c, d):
    return a+(b-a)/(1+10**((c-xval)*d))

# multiprocessing-ready fit to sigmoid function
def sigm_fit_mp(d):
    try:
        popt, pcov = curve_fit(curve_func, xdata=range(1,len(d)+1), ydata=d, p0=[100,200,10,0.5], bounds=([50, 100, 5, .05],[150, 250, 20, 5]), method='trf')
        ci = np.sqrt(np.diagonal(pcov))
    except RuntimeError:
        popt = np.nan*np.zeros(4);
        ci = np.nan*np.zeros(4);
    return (popt, ci)

# function to get into find_pupil_rotation (this will be eliminated soon)
def pupil_rotation_wrapper(eye_params, config, trial_name, side_letter):
    eyevidpath = find((trial_name + '*' + side_letter + 'EYE.avi'), config['data_path'])[0]
    toptimepath = find(('*' + trial_name + '*' + 'TOP_BonsaiTS.csv'), config['data_path'])[0]
    eyetimepath = find(('*' + trial_name + '*' + side_letter + 'EYE_BonsaiTS.csv'), config['data_path'])[0]
    worldtimepath = find(('*' + trial_name + '*' + side_letter + 'WORLD_BonsaiTS.csv'), config['data_path'])[0]

    return find_pupil_rotation(eyevidpath, toptimepath, eyetimepath, worldtimepath, trial_name, 'REYE', eye_params, config['save_path'], config['world_interp_method'], config['range_radius'], config)

# find pupil edge and align over time to calculate cyclotorsion
def find_pupil_rotation(eyevidpath, toptimepath, eyetimepath, worldtimepath, trial_name, eyeext, eye_ell_params, save_path, world_interp_method, ranger, config):

    print('found ' + str(multiprocessing.cpu_count()) + ' as cpu count for multiprocessing')

    # set up range of degrees in radians
    rad_range = np.deg2rad(np.arange(360))

    # open time files
    eyeTS = open_time(eyetimepath, np.size(eye_ell_params, axis=0))
    worldTS = open_time(worldtimepath, np.size(eye_ell_params, axis=0))
    topTS = open_time(toptimepath)

    # interpolate ellipse parameters to worldcam timestamps
    eye_ell_interp_params = eye_ell_params.interp_like(xr.DataArray(worldTS), method=world_interp_method)

    # the very first timestamp
    start_time = min(eyeTS[0], worldTS[0], topTS[0])

    # get the ellipse parameters for this trial from the time-interpolated xarray
    eye_theta = eye_ell_interp_params.sel(ellipse_params='theta')
    eye_phi = eye_ell_interp_params.sel(ellipse_params='phi')
    eye_longaxis= eye_ell_interp_params.sel(ellipse_params='longaxis')
    eye_shortaxis = eye_ell_interp_params.sel(ellipse_params='shortaxis')
    eye_centX = eye_ell_interp_params.sel(ellipse_params='centX')
    eye_centY = eye_ell_interp_params.sel(ellipse_params='centY')

    # also get the the ellipse parameters that haven't been interpolated over
    eye_raw_theta = eye_ell_params.sel(ellipse_params='theta')
    eye_raw_phi = eye_ell_params.sel(ellipse_params='phi')
    eye_raw_longaxis= eye_ell_params.sel(ellipse_params='longaxis')
    eye_raw_shortaxis = eye_ell_params.sel(ellipse_params='shortaxis')

    eyeTSminusstart = [(t-start_time).seconds for t in eyeTS]
    worldTSminusstart = [(t-start_time).seconds for t in worldTS]

    # set up for the read-in video
    eyevid = cv2.VideoCapture(eyevidpath)
    totalF = int(eyevid.get(cv2.CAP_PROP_FRAME_COUNT)) # this can be changed to a small number of frames for testing
    set_size = (int(eyevid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(eyevid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    
    # set up for the multiprocessing that'll be used during sigmoid fit function
    n_proc = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=n_proc)
    n=0

    print('getting cross-section of pupil at each angle and fitting to sigmoid (slow)')
    for step in tqdm(np.arange(totalF)):
        # frame reading and black and white conversion
        eye_ret, eye_frame = eyevid.read()
        if not eye_ret:
            break
        eye_frame = cv2.cvtColor(eye_frame, cv2.COLOR_BGR2GRAY)

        # get ellisepe parameters for this time
        current_time = eyevid.get(cv2.CAP_PROP_POS_FRAMES)
        current_theta = eye_theta.sel(frame=current_time).values
        current_phi = eye_phi.sel(frame=current_time).values
        current_longaxis = eye_longaxis.sel(frame=current_time).values
        current_shortaxis = eye_shortaxis.sel(frame=current_time).values
        current_centX = eye_centX.sel(frame=current_time).values
        current_centY = eye_centY.sel(frame=current_time).values

        # some configuration 
        meanr = 0.5 * (current_longaxis + current_shortaxis) # mean radius
        r = range(int(meanr - ranger), int(meanr + ranger)) # range of values over mean radius (meanr)
        pupil_edge = np.zeros([totalF, 360, len(r)]) # empty array that the calculated edge of the pupil will be put into

        rad_range = np.deg2rad(np.arange(360))
        # get cross-section of pupil at each angle 1-360 and fit to sigmoid
        for i in range(0, len(r)):
            pupil_edge[step,:,i] = eye_frame[((current_centY + r[i]*(np.sin(rad_range))).astype(int),(current_centX + r[i]*(np.cos(rad_range))).astype(int))]
        d = pupil_edge[step,:,:]
        param_mp = [pool.apply_async(sigm_fit_mp, args=(d[n],)) for n in range(360)]
        params_output = [result.get() for result in param_mp]
        params = []; ci = []
        for vals in params_output:
            params.append(vals[0])
            ci.append(vals[1])
        params = np.stack(params); ci = np.stack(ci)

        fit_thresh = 1

        # extract radius variable from parameters
        rfit = params[:,2] - 1

        # if confidence interval in estimate is > fit_thresh pix, set to to NaN
        # then, remove if luminance goes the wrong way (e.g. from reflectance)
        for deg_th in range(0,360):
            rfit[deg_th] = np.where(ci[deg_th,2] > fit_thresh, np.nan, rfit[deg_th])
            rfit[deg_th] = np.where((ci[deg_th,1] - ci[deg_th,0]) < 0, np.nan, rfit[deg_th])

        # interpolate because convolution will create large NaN holes
        # is interpolation a good idea here? either way, the way this is done can be improved
        interp_x = [item for sublist in np.argwhere(np.isnan(rfit)) for item in sublist]
        interp_xp = [item for sublist in np.argwhere(~np.isnan(rfit)) for item in sublist]
        interp_fp = rfit[~np.isnan(rfit)]
        rfit_interp_vals = np.interp(interp_x, interp_xp, interp_fp)
        # replace values in rfit_interp if they were NaN with the values found in interpolation
        rfit_interp = rfit; j=0
        for i in range(0,len(rfit_interp)):
            if np.isnan(rfit_interp[i]):
                rfit_interp[i] = rfit_interp_vals[j]
                j = j + 1

        # median filter
        rfit_interp = signal.medfilt(rfit_interp,3)

        # subtract baseline because our points aren't perfectly centered on ellipse
        filtsize = 30
        rfit_conv = rfit - np.convolve(rfit_interp, np.ones(filtsize)/filtsize, mode='same')
        # edges have artifact from conv, so set to NaNs
        # could fix this by padding data with wraparound at 0 and 360deg before conv
        # the astropy package can do this with the convolution.convolve package
        # TO DO: test and impliment wraparound convolution with astropy function convolve
        rfit_conv[range(0,int(filtsize/2+1))] = np.nan
        rfit_conv[range((len(rfit_conv)-int(filtsize/2-1)),len(rfit_conv))] = np.nan

        # save out pupil edge data into one xarray for all frames
        if step == 0:
            rfit_conv_xr = xr.DataArray(rfit_conv)
            rfit_conv_xr['frame'] = eyevid.get(cv2.CAP_PROP_POS_FRAMES)
            rfit_conv_xr = xr.DataArray.rename(rfit_conv_xr, {'dim_0':'deg'})
        if step > 0:
            rfit_conv_temp = xr.DataArray(rfit_conv)
            rfit_conv_temp['frame'] = eyevid.get(cv2.CAP_PROP_POS_FRAMES)
            rfit_conv_temp = xr.DataArray.rename(rfit_conv_temp, {'dim_0':'deg'})
            rfit_conv_xr = xr.concat([rfit_conv_xr, rfit_conv_temp], dim='frame', fill_value=np.nan)

    # plot rfit for all trials and highlight mean
    if config['save_figs'] is True:
        plt.figure()
        plt.plot(rfit_conv_xr.T, alpha=0.3)
        plt.plot(np.mean(rfit_conv_xr.T, 1), 'b--')
        plt.title('convolved rfit for all trials, mean in blue')
        plt.ylim([-3,3])
        plt.savefig(os.path.join(config['save_path'], (trial_name + '_rfit_conv_alltrials.png')), dpi=300)
        plt.close()

    # correlation across timepoints
    timepoint_corr_rfit = pd.DataFrame(rfit_conv_xr.values).T.corr()

    # plot the correlation matrix of rfit over all timepoints
    if config['save_figs'] is True: 
        plt.figure()
        fig, ax = plt.subplots()
        im = ax.imshow(timepoint_corr_rfit)
        ax.set_title('correlation of radius fit across timepoints')
        ax.set_xticks(np.arange(len(timepoint_corr_rfit)))
        ax.set_yticks(np.arange(len(timepoint_corr_rfit)))
        ax.set_xticklabels(range(1,len(timepoint_corr_rfit)+1))
        ax.set_yticklabels(range(1,len(timepoint_corr_rfit)+1))
        fig.colorbar(im, ax=ax)
        plt.savefig(os.path.join(config['save_path'], (trial_name + '_rfit_alltime_correlations.png')), dpi=300)
        plt.close()

    n = np.size(rfit_conv_xr.values, 0)
    pupil_update = rfit_conv_xr.values
    total_shift = np.zeros(n); peak = np.zeros(n)
    c = total_shift
    template = np.nanmean(rfit_conv_xr.values, 0)

    # calculate mean as template
    template_rfitconv_cc, template_rfit_cc_lags = nanxcorr(rfit_conv_xr[7].values, template, 30)
    if config['save_figs'] is True:
        plt.figure()
        plt.plot(template)
        plt.title('mean as template')
        plt.savefig(os.path.join(config['save_path'], (trial_name + '_mean_template.png')), dpi=300)
        plt.close()

        plt.plot(template_rfitconv_cc)
        plt.title('rfit_conv template cross correlation')
        plt.savefig(os.path.join(config['save_path'], (trial_name + '_rfit_template_cc.png')), dpi=300)
        plt.close()

    # xcorr of two random timepoints
    if config['save_figs'] is True:
        t0 = np.random.random_integers(0,totalF-1); t1 = np.random.random_integers(0,totalF-1)
        rfit2times_cc, rfit2times_lags = nanxcorr(rfit_conv_xr.isel(frame=t0).values, rfit_conv_xr.isel(frame=t1).values, 10)
        plt.figure()
        plt.plot(rfit2times_cc, 'b-')
        plt.title('nanxcorr of frames ' + str(t0) + ' and ' + str(t1))
        plt.savefig(os.path.join(config['save_path'], (trial_name + '_xcorr_of_two_times.png')), dpi=300)
        plt.close()

    # iterative fit to alignment
    # start with mean as template
    # on each iteration, shift individual frames to max xcorr with template
    # then recalculate mean template
    print('doing iterative fit on frames to find alignment for each frame')
    for rep in tqdm(range(0,12)):
        # calculate and plot template
        if config['save_vids'] is True:
            plt.figure()
            plt.title('rep='+str(rep)+' in iterative fit')
            plt.plot(template)
            plt.savefig(os.path.join(config['save_path'], (trial_name + '_rep'+str(rep)+'_pupil_rot_template.png')), dpi=300)
            plt.close()

        eyevid = cv2.VideoCapture(eyevidpath)

        # for each frame, get correlation, and shift
        for frame_num in range(0,n):
            xc, lags = nanxcorr(template, pupil_update[frame_num,:], 10)
            c[frame_num] = np.amax(xc)
            peaklag = np.argmax(xc)
            peak[frame_num] = lags[peaklag]
            total_shift[frame_num] = total_shift[frame_num] + peak[frame_num]
            pupil_update[frame_num,:] = np.roll(pupil_update[frame_num,:], int(peak[frame_num]))

        # histogram of correlations
        if config['save_vids'] is True:
            plt.figure()
            plt.hist(c)
            plt.savefig(os.path.join(config['save_path'], (trial_name + '_rep'+str(rep)+'_correlation_hist.png')), dpi=300)
            plt.close()

    win = 3
    shift_nan = -total_shift
    shift_nan[c < 0.2] = np.nan # started at [c < 0.4], is it alright to change this? many values go to NaN otherwise
    shift_smooth = convolve(shift_nan, np.ones(win)/win, boundary='wrap') # convolve using astopy.convolution.convolve, which should work like nanconv by interpolating over nans as appropriate
    shift_smooth = shift_smooth - np.nanmedian(shift_smooth)
    shift_nan = shift_nan - np.nanmedian(shift_nan)

    if config['save_figs'] is True:
        plt.figure()
        plt.plot(shift_nan)
        plt.savefig(os.path.join(config['save_path'], (trial_name + '_shift_nan.png')), dpi=300)
        plt.close()

        plt.figure()
        plt.plot(shift_smooth)
        plt.savefig(os.path.join(config['save_path'], (trial_name + '_shift_smooth.png')), dpi=300)
        plt.close

    if config['save_vids'] is True:
        vidsavepath = os.path.join(config['save_path'], str(trial_name + '_pupil_rotation_rep' + str(rep) + '_' + eyeext + '.avi'))
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        vidout = cv2.VideoWriter(vidsavepath, fourcc, 60.0, (int(eyevid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(eyevid.get(cv2.CAP_PROP_FRAME_HEIGHT))))

        print('plotting pupil rotation on eye video')
        for step in tqdm(np.arange(totalF)):
            eye_ret, eye_frame = eyevid.read()

            if not eye_ret:
                break

            eye_frame = cv2.cvtColor(eye_frame, cv2.COLOR_BGR2GRAY)

            # get ellisepe parameters for this time
            current_time = int(eyevid.get(cv2.CAP_PROP_POS_FRAMES))
            current_theta = eye_theta.sel(frame=current_time).values
            current_phi = eye_phi.sel(frame=current_time).values
            current_longaxis = eye_longaxis.sel(frame=current_time).values
            current_shortaxis = eye_shortaxis.sel(frame=current_time).values
            current_centX = eye_centX.sel(frame=current_time).values
            current_centY = eye_centY.sel(frame=current_time).values

            # plot the ellipse edge
            rmin = 0.5 * (current_longaxis + current_shortaxis) - ranger
            for deg_th in range(0,360):
                try:
                    rad_th = rad_range[deg_th]
                    eye_frame = cv2.circle(eye_frame, (int(round(current_centX+(rmin+rfit[deg_th])*np.cos(rad_th))),int(round(current_centY+(rmin+rfit[deg_th])*np.sin(rad_th)))), 1, (0,0,0), thickness=-1)
                    # eye_frame = cv2.circle(eye_frame, (int(round(current_centX+(rmin+rfit[deg_th])*np.rad2deg(np.cos(deg_th)))),int(round(current_centY+(rmin+rfit[deg_th])*np.rad2deg(np.sin(deg_th))))), 1, (0,0,0), thickness=-1)
                except ValueError:
                    pass

            # plot the rotation of the eye as a vertical line made up of many circles
            d = range(-20,20)
            for d1 in d:
                try:
                    eye_frame = cv2.circle(eye_frame, (int(round(current_centX + d1 * (np.cos(np.deg2rad(shift_smooth[current_time]+90))))),int(round(current_centY + d1 * (np.sin(np.deg2rad(shift_smooth[current_time]+90)))))),1,(0,0,0),thickness=-1)
                except (ValueError, IndexError):
                    pass

            # plot the center of the eye on the frame as a larger dot than the others
            try:
                eye_frame = cv2.circle(eye_frame, (int(current_centX),int(current_centY)),3,(0,0,0),thickness=-1)
            except (ValueError, IndexError):
                pass

            # if user wants a few frames printed out (e.g. if only a few frames are being analyzed, this will save out every frame as a seperate .png figure)
            # plt.figure()
            # plt.imshow(eye_frame)
            # plt.savefig(os.path.join(config['save_path'], (trial_name + 'frame' + str(int(eyevid.get(cv2.CAP_PROP_POS_FRAMES))) + '_imshow.png')), dpi=300)
            # plt.close

            vidout.write(eye_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            vidout.release()
            cv2.destroyAllWindows()

    shift = xr.DataArray(shift_smooth, coords=[('frame',range(0,np.size(shift_smooth,0)))])
    # rfit_conv_xr = xr.DataArray.rename(rfit_conv_xr, dim_0='deg')

    return rfit_conv_xr, shift
