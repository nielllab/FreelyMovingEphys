"""
track_world.py

tracking world camera and finding pupil rotation

Nov. 09, 2020
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

# module imports
from util.read_data import open_time, find, nanxcorr

# basic world shifting without pupil rotation
def adjust_world(data_path, file_name, eyeext, topext, worldext, eye_ds, savepath):
    # get eye data out of dataset
    eye_pts = eye_data.raw_pt_values
    eye_ell_params = eye_data.ellipse_param_values

    # find the needed files from path and trial key
    top1vidpath = os.path.join(data_path, file_name) + '_' + topext + '.avi'
    eyevidpath = os.path.join(data_path, file_name) + '_' + eyeext + '.avi'
    worldvidpath = os.path.join(data_path, file_name) + '_' + worldext + '.avi'
    top1timepath = os.path.join(data_path, file_name) + '_' + topext +'_BonsaiTSformatted.csv'
    eyetimepath = os.path.join(data_path, file_name) + '_' + eyeext +'_BonsaiTSformatted.csv'
    worldtimepath = os.path.join(data_path, file_name) + '_' + worldext +'_BonsaiTSformatted.csv'

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
        popt, pcov = curve_fit(curve_func, xdata=range(1,len(d)+1), ydata=d, p0=[100,200,10,0.5], bounds=([50, 100, 5, .05],[150, 250, 20, 5]), method='trf', xtol=10**-5)
        ci = np.sqrt(np.diagonal(pcov))
    except RuntimeError:
        popt = np.nan*np.zeros(4)
        ci = np.nan*np.zeros(4)
    return (popt, ci)

# function to get into find_pupil_rotation (this will be eliminated once the pupil rotation is working well)
def pupil_rotation_wrapper(eye_params, config, trial_name, side_letter):
    eyevidpath = find((trial_name + '*' + side_letter + 'EYEdeinter.avi'), config['data_path'])[0]
    eyetimepath = find(('*' + trial_name + '*' + side_letter + 'EYE_BonsaiTSformatted.csv'), config['data_path'])[0]

    return find_pupil_rotation(eyevidpath, eyetimepath, trial_name, 'REYE', eye_params, config['trial_path'], config['world_interp_method'], config['range_radius'], config)

# find pupil edge and align over time to calculate cyclotorsion
def find_pupil_rotation(eyevidpath, eyetimepath, trial_name, eyeext, eye_ell_params, save_path, world_interp_method, ranger, config):

    print('found ' + str(multiprocessing.cpu_count()) + ' as cpu count for multiprocessing')

    if config['save_figs'] is True:
        pdf = matplotlib.backends.backend_pdf.PdfPages(os.path.join(config['trial_path'], (trial_name + '_' + eyeext + 'EYE_pupil_rotation_figs.pdf')))

    # set up range of degrees in radians
    rad_range = np.deg2rad(np.arange(360))

    # open time files
    eyeTS = open_time(eyetimepath, np.size(eye_ell_params, axis=0))
    # worldTS = open_time(worldtimepath, np.size(eye_ell_params, axis=0))

    # interpolate ellipse parameters to worldcam timestamps
    # eye_ell_interp_params = eye_ell_params.interp_like(xr.DataArray(worldTS), method=world_interp_method)
    eye_ell_interp_params = eye_ell_params

    # get the ellipse parameters for this trial from the time-interpolated xarray
    eye_theta = eye_ell_interp_params.sel(ellipse_params='theta')
    eye_phi = eye_ell_interp_params.sel(ellipse_params='phi')
    eye_longaxis= eye_ell_interp_params.sel(ellipse_params='longaxis')
    eye_shortaxis = eye_ell_interp_params.sel(ellipse_params='shortaxis')
    eye_centX = eye_ell_interp_params.sel(ellipse_params='X0')
    eye_centY = eye_ell_interp_params.sel(ellipse_params='Y0')

    # set up for the read-in video
    eyevid = cv2.VideoCapture(eyevidpath)
    totalF = int(eyevid.get(cv2.CAP_PROP_FRAME_COUNT)) # this can be changed to a small number of frames for testing
    set_size = (int(eyevid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(eyevid.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    # set up for the multiprocessing that'll be used during sigmoid fit function
    n_proc = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=n_proc)
    n=0

    key_error_count = 0

    print('getting cross-section of pupil at each angle and fitting to sigmoid (slow)')
    for step in tqdm(np.arange(totalF)):
        try:
            # frame reading and black and white conversion
            eye_ret, eye_frame = eyevid.read()

            if not eye_ret:
                break

            eye_frame = cv2.cvtColor(eye_frame, cv2.COLOR_BGR2GRAY)

            # get ellipse parameters for this time
            current_theta = eye_theta.sel(frame=step).values
            current_phi = eye_phi.sel(frame=step).values
            current_longaxis = eye_longaxis.sel(frame=step).values
            current_shortaxis = eye_shortaxis.sel(frame=step).values
            current_centX = eye_centX.sel(frame=step).values
            current_centY = eye_centY.sel(frame=step).values

            # some configuration
            meanr = 0.5 * (current_longaxis + current_shortaxis) # mean radius
            r = range(int(meanr - ranger), int(meanr + ranger)) # range of values over mean radius (meanr)
            pupil_edge = np.zeros([totalF, 360, len(r)]) # empty array that the calculated edge of the pupil will be put into

            rad_range = np.deg2rad(np.arange(360))
            # get cross-section of pupil at each angle 1-360 and fit to sigmoid
            for i in range(0, len(r)):
                pupil_edge[step,:,i] = eye_frame[((current_centY + r[i]*(np.sin(rad_range))).astype(int),(current_centX + r[i]*(np.cos(rad_range))).astype(int))]
            d = pupil_edge[step,:,:]

            # apply sigmoid fit with multiprocessing
            param_mp = [pool.apply_async(sigm_fit_mp, args=(d[n,:],)) for n in range(360)]
            params_output = [result.get() for result in param_mp]

            # apply signoid fit without multiprocessing
            # params_output = []
            # for n in range(360):
            #     params_output.append(sigm_fit_mp(d[n,:]))

            # unpack outputs of sigmoid fit
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
                rfit[deg_th] = np.where((params[deg_th,1] - params[deg_th,0]) < 0, np.nan, rfit[deg_th])

            try:
                # median filter
                rfit_interp = signal.medfilt(rfit,3)

                # subtract baseline because our points aren't perfectly centered on ellipse
                filtsize = 31
                rfit_conv = rfit - convolve(rfit_interp, np.ones(filtsize)/filtsize, boundary='wrap')
                # edges have artifact from conv, so set to NaNs
                #   no edge artifacts anymore -- astropy convolve wraps around
                # rfit_conv[range(0,int(filtsize/2+1))] = np.nan
                # rfit_conv[range((len(rfit_conv)-int(filtsize/2-1)),len(rfit_conv))] = np.nan

            except ValueError: # in case every value in rfit is NaN
                rfit_conv = np.empty(np.shape(rfit_conv)) # make an rfit_conv with the shape of the last one
        except (KeyError, ValueError) as e:
            key_error_count = key_error_count + 1
            rfit_conv = np.empty(360)

        # save out pupil edge data into one xarray for all frames
        if step == 0:
            rfit_conv_xr = xr.DataArray(rfit_conv)
            rfit_conv_xr['frame'] = step
            rfit_conv_xr = xr.DataArray.rename(rfit_conv_xr, {'dim_0':'deg'})

            rfit_xr = xr.DataArray(rfit)
            rfit_xr['frame'] = step
            rfit_xr = xr.DataArray.rename(rfit_xr, {'dim_0':'deg'})
        if step > 0:
            rfit_conv_temp = xr.DataArray(rfit_conv)
            rfit_conv_temp['frame'] = step
            rfit_conv_temp = xr.DataArray.rename(rfit_conv_temp, {'dim_0':'deg'})
            rfit_conv_xr = xr.concat([rfit_conv_xr, rfit_conv_temp], dim='frame', fill_value=np.nan)

            rfit_temp = xr.DataArray(rfit)
            rfit_temp['frame'] = step
            rfit_temp = xr.DataArray.rename(rfit_temp, {'dim_0':'deg'})
            rfit_xr = xr.concat([rfit_xr, rfit_temp], dim='frame', fill_value=np.nan)

    # threshold out any frames with large or small rfit_conv distributions
    for frame in range(0,np.size(rfit_conv_xr,0)):
        if np.min(rfit_conv_xr[frame,:]) < -10 or np.max(rfit_conv_xr[frame,:]) > 10:
            rfit_conv_xr[frame,:] = np.nan

    # plot rfit for all trials and highlight mean
    if config['save_figs'] is True:
        plt.figure()
        plt.plot(rfit_conv_xr.T, alpha=0.3)
        plt.plot(np.mean(rfit_conv_xr.T, 1), 'b--')
        plt.title('convolved rfit for all trials, mean in blue')
        plt.ylim([-3,3])
        pdf.savefig()
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
        pdf.savefig()
        plt.close()

    n = np.size(rfit_conv_xr.values, 0)
    pupil_update = rfit_conv_xr.values
    total_shift = np.zeros(n); peak = np.zeros(n)
    c = total_shift
    template = np.nanmean(rfit_conv_xr.values, 0)

    # calculate mean as template
    try:
        template_rfitconv_cc, template_rfit_cc_lags = nanxcorr(rfit_conv_xr[7].values, template, 30)
        template_nanxcorr = True
    except ZeroDivisionError:
        template_nanxcorr = False

    if config['save_figs'] is True:
        plt.figure()
        plt.plot(template)
        plt.title('mean as template')
        pdf.savefig()
        plt.close()

        if template_nanxcorr is True:
            plt.plot(template_rfitconv_cc)
            plt.title('rfit_conv template cross correlation')
            pdf.savefig()
            plt.close()

    # xcorr of two random timepoints
    if config['save_figs'] is True:
        try:
            t0 = np.random.random_integers(0,totalF-1); t1 = np.random.random_integers(0,totalF-1)
            rfit2times_cc, rfit2times_lags = nanxcorr(rfit_conv_xr.isel(frame=t0).values, rfit_conv_xr.isel(frame=t1).values, 10)
            rand_frames = True
        except ZeroDivisionError:
            rand_frames = False
        if rand_frames is True:
            plt.figure()
            plt.plot(rfit2times_cc, 'b-')
            plt.title('nanxcorr of frames ' + str(t0) + ' and ' + str(t1))
            pdf.savefig()
            plt.close()

    # iterative fit to alignment
    # start with mean as template
    # on each iteration, shift individual frames to max xcorr with template
    # then recalculate mean template
    print('doing iterative fit on frames to find alignment for each frame')
    for rep in tqdm(range(0,12)):

        # for each frame, get correlation, and shift
        for frame_num in range(0,n):
            try:
                xc, lags = nanxcorr(template, pupil_update[frame_num,:], 10)
                c[frame_num] = np.amax(xc) # value of max
                peaklag = np.argmax(xc) # position of max
                peak[frame_num] = lags[peaklag]
                total_shift[frame_num] = total_shift[frame_num] + peak[frame_num]
                pupil_update[frame_num,:] = np.roll(pupil_update[frame_num,:], int(peak[frame_num]))
            except ZeroDivisionError:
                total_shift[frame_num] = np.nan
                pupil_update[frame_num,:] = np.nan

        if config['save_figs'] is True:
            # plot template with pupil_update for each iteration of fit
            plt.figure()
            plt.title('pupil_update of rep='+str(rep)+' in iterative fit')
            plt.plot(template, 'k--', alpha=0.8)
            plt.plot(pupil_update.T, alpha=0.2)
            pdf.savefig()
            plt.close()

            # histogram of correlations
            plt.figure()
            plt.title('correlations of rep='+str(rep)+' in iterative fit')
            plt.hist(c, bins=300)
            pdf.savefig()
            plt.close()

    # total_shift[np.mean(rfit_conv,1) > 25] = np.nan

    win = 3
    shift_nan = -total_shift
    shift_nan[c < 0.2] = np.nan # started at [c < 0.4], is it alright to change this? many values go to NaN otherwise
    shift_nan[shift_nan > 0.6] = np.nan; shift_nan[shift_nan < -0.6] = np.nan # get rid of very large shifts
    shift_smooth = np.convolve(shift_nan, np.ones(win)/win, mode='same')
    shift_smooth = shift_smooth - np.nanmedian(shift_smooth)
    shift_nan = shift_nan - np.nanmedian(shift_nan)

    if config['save_figs'] is True:
        plt.figure()
        plt.plot(shift_nan)
        plt.title('shift nan')
        pdf.savefig()
        plt.close()

        plt.figure()
        plt.plot(shift_smooth)
        plt.title('shift smooth')
        pdf.savefig()
        plt.close()

        plt.figure()
        plt.plot(rfit_xr.T)
        plt.plot(np.nanmean(rfit_xr.T,1), 'b--')
        plt.title('rfit')
        pdf.savefig()
        plt.close()

        plt.figure()
        plt.plot(rfit_conv_xr.T)
        plt.plot(np.nanmean(rfit_conv_xr.T,1), 'b--')
        plt.title('rfit conv')
        pdf.savefig()
        plt.close()

        # rfit, rfit_conv, plot astropy and numpy conv

    shift_smooth1 = xr.DataArray(shift_smooth, dims=['frame'])

    if config['save_vids'] is True:
        eyevid = cv2.VideoCapture(eyevidpath)
        vidsavepath = os.path.join(config['trial_path'], str(trial_name + '_pupil_rotation_rep' + str(rep) + '_' + eyeext + '.avi'))
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        vidout = cv2.VideoWriter(vidsavepath, fourcc, 60.0, (int(eyevid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(eyevid.get(cv2.CAP_PROP_FRAME_HEIGHT))))
        
        print('plotting pupil rotation on eye video')
        for step in tqdm(np.arange(totalF)):
            eye_ret, eye_frame = eyevid.read()

            if not eye_ret:
                break

            # get ellisepe parameters for this time
            current_theta = eye_theta.sel(frame=step).values
            current_phi = eye_phi.sel(frame=step).values
            current_longaxis = eye_longaxis.sel(frame=step).values
            current_shortaxis = eye_shortaxis.sel(frame=step).values
            current_centX = eye_centX.sel(frame=step).values
            current_centY = eye_centY.sel(frame=step).values

            # plot the ellipse edge
            rmin = 0.5 * (current_longaxis + current_shortaxis) - ranger
            for deg_th in range(0,360):
                rad_th = rad_range[deg_th]
                edge_x = np.round(current_centX+(rmin+rfit_xr.isel(frame=step,deg=deg_th).values)*np.cos(rad_th))
                edge_y = np.round(current_centY+(rmin+rfit_xr.isel(frame=step,deg=deg_th).values)*np.sin(rad_th))
                if pd.isnull(edge_x) is False and pd.isnull(edge_y) is False:
                    eye_frame = cv2.circle(eye_frame, (int(edge_x),int(edge_y)), 1, (235,52,155), thickness=-1)

            # plot the rotation of the eye as a vertical line made up of many circles
            for d in np.linspace(-0.5,0.5,100):
                rot_x = np.round(current_centX + d*(np.rad2deg(np.cos(shift_smooth1.isel(frame=step).values))))
                rot_y = np.round(current_centY + d*(np.rad2deg(np.sin(shift_smooth1.isel(frame=step).values))))
                if pd.isnull(rot_x) is False and pd.isnull(rot_y) is False:
                    eye_frame = cv2.circle(eye_frame, (int(rot_x),int(rot_y)),1,(255,255,255),thickness=-1)

            # plot the center of the eye on the frame as a larger dot than the others
            if pd.isnull(current_centX) is False and pd.isnull(current_centY) is False:
                eye_frame = cv2.circle(eye_frame, (int(current_centX),int(current_centY)),3,(0,255,0),thickness=-1)

            vidout.write(eye_frame)

        vidout.release()

    # temporary: save pupil rotation values to csv in case of error during xarray formatting
    shift_smooth_pd = pd.DataFrame(shift_smooth)
    # shift_smooth_pd.to_csv(os.path.join(config['trial_path'], str(trial_name + '_shift_smooth.csv')), index=False)
    shift = xr.DataArray(shift_smooth_pd, dims=['frame','shift'])
    print('key/value error count during sigmoid fit: ' + str(key_error_count))

    pdf.close()

    return rfit_xr, rfit_conv_xr, shift
