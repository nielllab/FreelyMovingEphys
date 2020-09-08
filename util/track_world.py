"""
track_world.py

World tracking utilities

Last modified September 07, 2020
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

# module imports
from util.read_data import open_time, find

# get the mean confidence interval
def find_ci(a, confidence=0.95):
    return ((np.std(a) / np.sqrt(np.size(a, 0))) * (scipy.stats.t.ppf((1 - confidence) / 2, (np.size(a, 0) - 1))))

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
    eye_pts = xr.Dataset.to_array(eye_ds).sel(variable='raw_pt_values')
    eye_ell_params = xr.Dataset.to_array(eye_ds).sel(variable='ellipse_param_values')

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

# calculates xcorr ignoring NaNs without altering timing
# adapted from /niell-lab-analysis/freely moving/nanxcorr.m
def nanxcorr(x, y, maxlag=25, normalization='full'):
    if normalization == 'zero':
        x = x - np.nanmean(x)
        y = y - np.nanmean(y)
        normalization = 'full'
    lags = range(-maxlag, maxlag)
    cc = []
    for i in range(0,len(lags)):
        yshift = np.roll(y, lags[i])
        use = ~pd.isnull(x + yshift)
        cc_i = np.correlate(x[use], yshift[use], mode=normalization)
        cc.append(cc_i)

    return cc, lags

# sigmoid function
def curve_func(xval, a, b, c, d):
    return a+(b-a)/(1+10**((c-xval)*d))

# fit to sigmoid function
def sigm_fit(d):
    return curve_fit(curve_func, xdata=range(1,len(d)+1), ydata=d, p0=[100,200,10,0.5], maxfev=1000000, bounds=([50,100, 5, .05],[150, 250, 20, 5]), method='dogbox')

# multiprocessing-ready fit to sigmoid function
def sigm_fit_mp(d):
    try:
        popt, pcov = curve_fit(curve_func, xdata=range(1,len(d)+1), ydata=d, p0=[100,200,10,0.5], maxfev=1000000, bounds=([50,100, 5, .05],[150, 250, 20, 5]), method='dogbox')
        ci = np.sqrt(np.diagonal(pcov))
    except RuntimeError:
        popt = np.nan; ci = np.nan
    return (popt, ci)

def pupil_rotation_wrapper(eye_params, config, trial_name, side_letter):
    eyevidpath = find((trial_name + '*' + side_letter + 'EYE.avi'), config['data_path'])[0]
    toptimepath = find(('*' + trial_name + '*' + 'TOP_BonsaiTS.csv'), config['data_path'])[0]
    eyetimepath = find(('*' + trial_name + '*' + side_letter + 'EYE_BonsaiTS.csv'), config['data_path'])[0]
    worldtimepath = find(('*' + trial_name + '*' + side_letter + 'WORLD_BonsaiTS.csv'), config['data_path'])[0]

    return find_pupil_rotation(eyevidpath, toptimepath, eyetimepath, worldtimepath, trial_name, 'REYE', eye_params, config['save_path'], config['world_interp_method'], config['range_radius'], config)

# find pupil edge and align over time to calculate cyclotorsion
def find_pupil_rotation(eyevidpath, toptimepath, eyetimepath, worldtimepath, trial_name, eyeext, eye_ell_params, save_path, world_interp_method, ranger, config):
    starting_time = time.time()

    print('found ' + str(multiprocessing.cpu_count()) + ' as cpu count for multiprocessing')

    rad_range = [np.deg2rad(i) for i in range(0,360)]

    # open time files
    eyeTS = open_time(eyetimepath, np.size(eye_ell_params, axis=0))
    worldTS = open_time(worldtimepath, np.size(eye_ell_params, axis=0))
    topTS = open_time(toptimepath)

    # interpolate ellipse parameters to worldcam timestamps
    eye_ell_interp_params = eye_ell_params.interp_like(xr.DataArray(worldTS), method=world_interp_method)

    # the very first timestamp
    start_time = min(eyeTS[0], worldTS[0], topTS[0])

    eye_theta = eye_ell_interp_params.sel(ellipse_params='theta')
    eye_phi = eye_ell_interp_params.sel(ellipse_params='phi')
    eye_longaxis= eye_ell_interp_params.sel(ellipse_params='longaxis')
    eye_shortaxis = eye_ell_interp_params.sel(ellipse_params='shortaxis')
    eye_centX = eye_ell_interp_params.sel(ellipse_params='centX')
    eye_centY = eye_ell_interp_params.sel(ellipse_params='centY')

    eye_raw_theta = eye_ell_params.sel(ellipse_params='theta')
    eye_raw_phi = eye_ell_params.sel(ellipse_params='phi')
    eye_raw_longaxis= eye_ell_params.sel(ellipse_params='longaxis')
    eye_raw_shortaxis = eye_ell_params.sel(ellipse_params='shortaxis')

    eyeTSminusstart = [(t-start_time).seconds for t in eyeTS]
    worldTSminusstart = [(t-start_time).seconds for t in worldTS]

    eyevid = cv2.VideoCapture(eyevidpath)

    # setup the file to save out of this
    vidsavepath = os.path.join(config['save_path'], str(trial_name + '_pupil_edge_detection_' + eyeext + '.avi'))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    vidout = cv2.VideoWriter(vidsavepath, fourcc, 60.0, (int(eyevid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(eyevid.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    set_size = (int(eyevid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(eyevid.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    while(1):
        sys.stdout.write('reading frame {} for sigmoid fit of pupil edge, now at {}% progress\r'.format(str(int(eyevid.get(cv2.CAP_PROP_POS_FRAMES))), str(int(eyevid.get(cv2.CAP_PROP_POS_FRAMES)/int(eyevid.get(cv2.CAP_PROP_FRAME_COUNT))))))
        sys.stdout.flush()

        # read the frame for this pass through while loop
        eye_ret, eye_frame = eyevid.read()

        if not eye_ret:
            break

        # debug with a few frames
        # if eyevid.get(cv2.CAP_PROP_POS_FRAMES) > 6:
        #     break

        eye_frame = cv2.cvtColor(eye_frame, cv2.COLOR_BGR2GRAY)

        # get ellisepe parameters for this time
        current_time = eyevid.get(cv2.CAP_PROP_POS_FRAMES)
        current_theta = eye_theta.sel(frame=current_time).values
        current_phi = eye_phi.sel(frame=current_time).values
        current_longaxis = eye_longaxis.sel(frame=current_time).values
        current_shortaxis = eye_shortaxis.sel(frame=current_time).values
        current_centX = eye_centX.sel(frame=current_time).values
        current_centY = eye_centY.sel(frame=current_time).values

        # get cross-section of pupil at each angle 1-360 and fit to sigmoid
        ci = []; params = []
        n_proc = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(processes=n_proc)
        start_mp = time.time()
        param_mp = []
        for deg_th in range(0,360):
            rad_th = rad_range[deg_th]
            meanr = 0.5 * (current_longaxis + current_shortaxis)
            r = range(int(meanr - ranger), int(meanr + ranger))
            # go out along radius and get pixel values
            pupil_edge = np.zeros([360, len(r)])
            for i in range(0, len(r)):
                pupil_edge[deg_th,i] = eye_frame[int(current_centY+r[i]*(np.sin(rad_th))), int(current_centX+r[i]*(np.cos(rad_th)))]
            # fit sigmoind to pupil edge at this theta
            d = pupil_edge[deg_th,:]
            param_mp.append(pool.apply_async(sigm_fit_mp, args=(d,)))
        params_output = [result.get() for result in param_mp]

        params = []; ci = []
        for deg_th in range(0,360):
            params.append(params_output[deg_th][0])
            ci.append(params_output[deg_th][1])
        params = np.stack(params); ci = np.stack(ci)

        fit_thresh = 1

        # extract radius variable from parameters
        rfit = params[:,2] - 1

        # if confidence interval in estimate is > fit_thresh pix, set to to NaN
        # then, remove if luminance goes the wrong way (e.g. from reflectance)
        for deg_th in range(0,360):
            rfit[deg_th] = np.where(ci[deg_th,2] > fit_thresh, np.nan, rfit[deg_th])
            rfit[deg_th] = np.where(ci[deg_th,2] < 0, np.nan, rfit[deg_th])

        # # interpolate because convolution will create large NaN holes
        interp_x = [item for sublist in np.argwhere(np.isnan(rfit)) for item in sublist]
        interp_xp = [item for sublist in np.argwhere(~np.isnan(rfit)) for item in sublist]
        interp_fp = rfit[~np.isnan(rfit)]
        rfit_interp_vals = np.interp(interp_x, interp_xp, interp_fp)
        # replace values in rfit_interp if they were np.nan with the values found in interpolation
        rfit_interp = rfit; j=0
        for i in range(0,len(rfit_interp)):
            if np.isnan(rfit_interp[i]):
                rfit_interp[i] = rfit_interp_vals[j]
                j = j + 1

        # median filter
        rfit_interp = signal.medfilt(rfit_interp,3)

        # subtract baseline (boxcar average using conv)
        # because our points aren't perfectly centered on ellipse
        filtsize = 30
        rfit_conv = rfit - np.convolve(rfit_interp, np.ones(filtsize)/filtsize, mode='same')
        # edges have artifact from conv, so set to NaNs
        # could fix this by padding data with wraparound at 0 and 360deg before conv
#         rfit_conv[range(0,int(filtsize/2+1))] = np.nan
#         rfit_conv[range((len(rfit_conv)-int(filtsize/2-1)))] = np.nan

        # save out video with detected edge of pupil outlined in green
        rmin = 0.5 * (current_longaxis + current_shortaxis) - ranger
        for deg_th in range(0,360):
            rad_th = rad_range[deg_th]
            try:
                eye_frame = cv2.circle(eye_frame, (int(round(current_centX+(rmin+rfit[deg_th])*(np.cos(rad_th)))),int(round(current_centY+(rmin+rfit[deg_th])*(np.sin(rad_th))))), 1, (0,0,0), thickness=-1)
            except ValueError:
                pass

        # save out data
        if eyevid.get(cv2.CAP_PROP_POS_FRAMES) == 1:
            rfit_conv_xr = xr.DataArray(rfit_conv)
            rfit_conv_xr['frame'] = eyevid.get(cv2.CAP_PROP_POS_FRAMES)
#             rfit_conv_xr = xr.DataArray.rename(rfit_conv_xr, {'dim_1':'deg'})
        if eyevid.get(cv2.CAP_PROP_POS_FRAMES) > 1:
            rfit_conv_temp = xr.DataArray(rfit_conv)
            rfit_conv_temp['frame'] = eyevid.get(cv2.CAP_PROP_POS_FRAMES)
#             rfit_conv_temp = xr.DataArray.rename(rfit_conv_temp, {'dim_1':'deg'})
            rfit_conv_xr = xr.concat([rfit_conv_xr, rfit_conv_temp], dim='frame', fill_value=np.nan)

        vidout.write(eye_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        vidout.release()
        cv2.destroyAllWindows()

    # correlation across timepoints
    timepoint_corr_rfit = np.corrcoef([rfit_conv_xr.isel(frame=1), rfit_conv_xr.isel(frame=2)]) # best way to make this apply to the entire dimension 'frame'? return to this later

    plt.figure()
    fig, ax = plt.subplots()
    im = ax.imshow(timepoint_corr_rfit)
    ax.set_title('correlation of radius fit across timepoints')
    ax.set_xticks(np.arange(len(timepoint_corr_rfit)))
    ax.set_yticks(np.arange(len(timepoint_corr_rfit)))
    ax.set_xticklabels(range(1,len(timepoint_corr_rfit)+1))
    ax.set_yticklabels(range(1,len(timepoint_corr_rfit)+1))
    plt.savefig(os.path.join(config['save_path'], (trial_name + '_corr_radius_fit.png')), dpi=300)
    plt.close()

    # calculate mean as template
    template = np.nanmean(rfit_conv_xr.values, 0)
    plt.figure()
    plt.plot(template)
    plt.title('mean as template')
    plt.savefig(os.path.join(config['save_path'], (trial_name + '_mean_template.png')), dpi=300)
    plt.close()

    plt.xcorr(rfit_conv_xr.isel(frame=1).values,template)
    plt.title('mean as template')
    plt.title('rfit_conv template cross correlation')
    plt.savefig(os.path.join(config['save_path'], (trial_name + '_rfit_template_cc.png')), dpi=300)
    plt.close()

    # xcorr of two random timepoints
    t1 = 1; t2 = 2;
    plt.figure()
    plt.xcorr(rfit_conv_xr.sel(frame=t1).values, rfit_conv_xr.sel(frame=t2).values)
    plt.title('xcorr of time ' + str(t1) + ' and ' + str(t2))
    plt.savefig(os.path.join(config['save_path'], (trial_name + '_xcorr_of_two_times.png')), dpi=300)
    plt.close()

    # iterative fit to alignment
    # start with mean as template
    # on each iteration, shift individual frames to max xcorr with template
    # then recalculate mean template
    n = np.size(rfit_conv_xr.values, 0)
    pupil_update = rfit_conv_xr.values
    total_shift = np.zeros(n); peak = np.zeros(n)
    c = total_shift;
    for rep in range(0,12):
        sys.stdout.write('starting rep {} for iterative fit of pupil rotation\r'.format(str(int(rep))))
        sys.stdout.flush()
        # calculate and plot template
        template = np.nanmean(rfit_conv_xr.values, 0)
        plt.figure()
        plt.title('rep='+str(rep)+' in iterative fit')
        plt.plot(template)
        plt.savefig(os.path.join(config['save_path'], (trial_name + '_rep'+str(rep)+'_pupil_rot_template.png')), dpi=300)
        plt.close()

        eyevid = cv2.VideoCapture(eyevidpath)

        for frame_num in range(1,n):
            # for each frame, get correlation and shift
            xc = np.correlate(template, pupil_update[frame_num,:], 'same'); lags = range(-int(0.5*len(template)),int(0.5*len(template)))
            c[frame_num] = np.amax(xc); peaklag = np.argmax(xc)
            peak[frame_num] = lags[peaklag]
            total_shift[frame_num] = total_shift[frame_num] + peak[frame_num]
            pupil_update[frame_num,:] = np.roll(pupil_update[frame_num,:], int(peak[frame_num]))

        # histogram of correlations
        plt.figure()
        plt.hist(c)
        plt.savefig(os.path.join(config['save_path'], (trial_name + '_rep'+str(rep)+'_correlation_hist.png')), dpi=300)
        plt.close()

    win = 3
    shift_nan = -total_shift
    shift_nan[c < 0.4] = np.nan

    # interp_x = [item for sublist in np.argwhere(np.isnan(shift_nan)) for item in sublist]
    # interp_xp = [item for sublist in np.argwhere(~np.isnan(shift_nan)) for item in sublist]
    # interp_fp = shift_nan[~np.isnan(shift_nan)]
    # shift_interp = np.interp(interp_x, interp_xp, interp_fp)

    shift_smooth = np.convolve(shift_nan, np.ones(win)/win, mode='same')
    shift_smooth = shift_smooth - np.nanmedian(shift_smooth)
    shift_nan = shift_nan - np.nanmedian(shift_nan)

    plt.figure()
    plt.plot(shift_nan)
    plt.savefig(os.path.join(config['save_path'], (trial_name + '_shift_nan.png')), dpi=300)
    plt.close()

    plt.figure()
    plt.plot(shift_smooth)
    plt.savefig(os.path.join(config['save_path'], (trial_name + '_shift_smooth.png')), dpi=300)
    plt.close

    # interpolate ellipse parameters to worldcam timestamps
    eye_ell_interp_params = eye_ell_params.interp_like(xr.DataArray(worldTS), method=world_interp_method)

    vidsavepath = os.path.join(config['save_path'], str(trial_name + '_pupil_rotation_rep' + str(rep) + '_' + eyeext + '.avi'))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    vidout = cv2.VideoWriter(vidsavepath, fourcc, 60.0, (int(eyevid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(eyevid.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    while(1):
        eye_ret, eye_frame = eyevid.read()

        if not eye_ret:
            break

        # debug with a few frames
        # if eyevid.get(cv2.CAP_PROP_POS_FRAMES) > 6:
        #     break

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
                eye_frame = cv2.circle(eye_frame, (int(round(current_centX+(rmin+rfit[deg_th])*np.rad2deg(np.cos(rad_th)))),int(round(current_centY+(rmin+rfit[deg_th])*np.rad2deg(np.sin(rad_th))))), 1, (0,0,0), thickness=-1)
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

        # plot the actual ellipse from track_eye
        try:
            eye_frame = cv2.ellipse(eye_frame, (int(current_centX),int(current_centY)), (int(current_longaxis),int(current_shortaxis)), int(np.rad2deg(current_phi)), 0, 360, (0,0,0), 2)
        except (ValueError, IndexError):
            pass


        plt.figure()
        plt.imshow(eye_frame)
        plt.savefig(os.path.join(config['save_path'], (trial_name + 'frame' + str(int(eyevid.get(cv2.CAP_PROP_POS_FRAMES))) + '_imshow.png')), dpi=300)
        plt.close

        vidout.write(eye_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        vidout.release()
        cv2.destroyAllWindows()

    print('total time for finding pupil rotation: ' + str((time.time() - starting_time)/60) + ' min')
    print('time per frame: ' + str((time.time() - starting_time)/int(eyevid.get(cv2.CAP_PROP_FRAME_COUNT))) + ' sec') # CHANGE TO: /int(eyevid.get(cv2.CAP_PROP_FRAME_COUNT)

    shift = xr.DataArray(shift_smooth, coords=[('frame',range(0,np.size(shift_smooth,0)))])
    rfit_conv_xr = xr.DataArray.rename(rfit_conv_xr, dim_0='deg')

    return rfit_conv_xr, shift
