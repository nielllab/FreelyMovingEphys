"""
FreelyMovingEphys world tracking utilities
track_world.py

Last modified August 17, 2020
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
import multiprocessing as mp
import sys

# module imports
from util.read_data import open_h5, open_time, read_paths, read1path

# nonlinear regression parameter confidence intervals
# adapted from https://gist.github.com/danieljfarrell/81809e6f53c07a18cd12
def nlparci(fvec, jac):
    # residual sum of squares
    rss = np.sum(fvec**2)
    # number of data points and parameters
    n, p = jac.shape
    # the statistical degrees of freedom
    nmp = n - p
    # mean residual error
    ssq = rss / nmp
    # the Jacobian
    J = np.matrix(jac)
    # covariance matrix
    c = np.linalg.inv(J.T*J)
    # variance-covariance matrix.
    pcov = c * ssq
    # Diagonal terms provide error estimate based on uncorrelated parameters.
    # The sqrt convert from variance to std. dev. units.
    err = np.sqrt(np.diag(np.abs(pcov))) * 1.96  # std. dev. x 1.96 -> 95% conf
    # Here err is the full 95% area under the normal distribution curve. This
    # means that the plus-minus error is the half of this value
    return err

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

def curve_func(xval, a, b, c, d):
    return (a+b-a)/(1+10**((c-xval)*d))

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
    eyeTS = open_time(eyetimepath, np.size(eye_pts, axis=0))
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

# find pupil edge and align over time to calculate cyclotorsion
# all inputs must be deinterlaced
def find_pupil_rotation(data_path, file_name, eyeext, topext, worldext, eye_ds, save_path, world_interp_method, ranger):
    start_time = time.time()

    # get eye data out of dataset
    print('managing files')
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
    fig_dir = save_path + '/' + file_name + '/'
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    # open time files
    eyeTS = open_time(eyetimepath, np.size(eye_pts, axis=0))
    worldTS = open_time(worldtimepath)
    topTS = open_time(top1timepath)

    print('interpolating and selecting parameters')
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

    print('opening video')
    eyevid = cv2.VideoCapture(eyevidpath)

    # setup the file to save out of this
    vidsavepath = os.path.join(fig_dir, str(file_name + '_worldshift_' + eyeext + '.avi'))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    vidout = cv2.VideoWriter(vidsavepath, fourcc, 20.0, (int(eyevid.get(cv2.CAP_PROP_FRAME_WIDTH))*2, int(eyevid.get(cv2.CAP_PROP_FRAME_HEIGHT))*2))

    set_size = (int(eyevid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(eyevid.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    while(1):
        # read the frame for this pass through while loop
        eye_ret, eye_frame = eyevid.read()

        if not eye_ret:
            break

        if eyevid.get(cv2.CAP_PROP_POS_FRAMES) > 5:
            break

        cur = str(int(eyevid.get(cv2.CAP_PROP_POS_FRAMES)))
        tot = str(int(eyevid.get(cv2.CAP_PROP_FRAME_COUNT)))
        progress = str(int((eyevid.get(cv2.CAP_PROP_POS_FRAMES) / eyevid.get(cv2.CAP_PROP_FRAME_COUNT))*100))
        sys.stdout.write('working on pupil in frame {} of {}; now at {}% progress\r'.format(cur, tot, progress))
        sys.stdout.flush()

        eye_frame = cv2.cvtColor(eye_frame, cv2.COLOR_BGR2GRAY)

        # get ellisepe parameters for this time
        current_time = eyevid.get(cv2.CAP_PROP_POS_FRAMES)
        current_theta = eye_theta.sel(frame=current_time).values[0]
        current_phi = eye_phi.sel(frame=current_time).values[0]
        current_longaxis = eye_longaxis.sel(frame=current_time).values[0]
        current_shortaxis = eye_shortaxis.sel(frame=current_time).values[0]
        current_centX = eye_centX.sel(frame=current_time).values[0]
        current_centY = eye_centY.sel(frame=current_time).values[0]

        # get cross-section of pupil at each angle 1-360 and fit to sigmoid
        ci = []; params = []
        for th in range(0, 360):
            meanr = 0.5 * (current_longaxis + current_shortaxis)
            r = range(int(meanr - ranger), int(meanr + ranger))

            # go out along radius and get pixel values
            pupil_edge = np.zeros([360, len(r)])
            for i in range(0, len(r)):
                pupil_edge[th,:] = eye_frame[int(current_centY+r[i]*(np.deg2rad(np.sin(th)))), int(current_centX+r[i]*(np.deg2rad(np.cos(th))))]

            # fit sigmoind to pupil edge at this theta
            d = pupil_edge[th,:]
            init_params = [100,200,10,0.5]
            # non-linear regression, fit to sigmoid
            popt, pcov = curve_fit(curve_func, xdata=range(0,len(d)), ydata=d, p0=init_params)
            # confidence interval of the parameters
            delta = nlparci(popt, pcov)
            ci.append(delta)
            params.append(popt)

        print(d)
        fit_thresh = 1
        params = np.array(params)

        print(np.shape(params))
        # extract radius variable from parameters
        rfit = params[:,2] - 1

        # if confidence interval in estimate is > fit_thresh pix, set to to NaN
        # then, remove if luminance goes the wrong way (e.g. from reflectance)
        for th in range(0,np.size(ci, 0)):
            rfit[th,:] = np.where(ci[th] > fit_thresh, rfit[:,th], np.nan)
            rfit[th,:] = np.where(ci[th] < 0, rfit[:,th], np.nan)

        # median filter
        rfit = signal.medfilt(rfit,3)

        filtsize = 30
        # subtract baseline (boxcar average using conv)
        # this is because our points aren't perfectly centered on ellipse
        rfit_conv = np.array(np.zeros(np.shape(rfit)))
        for f in range(0,np.size(rfit, axis=0)):
            rfit_conv[f,:] = rfit[f,:] - np.convolve(rfit[f,:], np.ones(filtsize)/filtsize, 'same')

        # edges have artifact from conv, so set to NaNs
        # could fix this by padding data with wraparound at 0 and 360deg before conv
        rfit_conv[:,range(0,int(filtsize/2+1))] = np.nan
        rfit_conv[:,range(0,int(filtsize/2-1))] = np.nan

        plot_color1 = (0,255,255)
        rmin = 0.5 * (current_longaxis + current_shortaxis) - ranger
        for th in range(1,361):
            x1 = np.array(mean_cent[0] + (rmin + rfit) * np.cos(th)).astype(int)[0,0]
            y1 = np.array(mean_cent[1] + (rmin + rfit) * np.sin(th)).astype(int)[0,0]
            x2 = np.array(mean_cent[0] - (rmin + rfit) * np.cos(th)).astype(int)[0,0]
            y2 = np.array(mean_cent[1] - (rmin + rfit) * np.sin(th)).astype(int)[0,0]
            eye_frame = cv2.line(eye_frame, (x1,y1), (x2,y2), plot_color1, thickness=4)

        if eyevid.get(cv2.CAP_PROP_POS_FRAMES) == 1:
            rfit_conv_xr = xr.DataArray(rfit_conv)
            rfit_conv_xr['frame'] = eyevid.get(cv2.CAP_PROP_POS_FRAMES)
            rfit_conv_xr = xr.DataArray.rename(rfit_conv_xr, {'dim_1':'deg'})
        if eyevid.get(cv2.CAP_PROP_POS_FRAMES) > 1:
            rfit_conv_temp = xr.DataArray(rfit_conv)
            rfit_conv_temp['frame'] = eyevid.get(cv2.CAP_PROP_POS_FRAMES)
            rfit_conv_temp = xr.DataArray.rename(rfit_conv_temp, {'dim_1':'deg'})
            rfit_conv_xr = xr.concat([rfit_conv_xr, rfit_conv_temp], dim='frame', fill_value=np.nan)

        vidout.write(eye_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        vidout.release()
        cv2.destroyAllWindows()

    # correlation across timepoints
    timepoint_corr_rfit = np.corrcoef(rfit_conv_xr.values[:,0,:], rowvar=True)
    rfit_conv_vals = np.array(rfit_conv_xr.values[:,0,:], dtype=object)
    plt.figure()
    plt.plot(timepoint_corr_rfit)
    plt.title('correlation of radius fit across timepoints')
    plt.savefig(fig_dir + 'corr_radius_fit.png', dpi=300)
    plt.close()

    # calculate mean as template
    template = np.nanmean(rfit_conv_vals, 1)
    plt.figure()
    plt.plot(template)
    plt.title('mean as template')
    plt.savefig(fig_dir + 'mean_template.png', dpi=300)
    plt.close()

    for test_ind in range(0, np.size(rfit_conv_vals, 1)):
        try:
            nanxcorr_cc, nanxcorr_lags = nanxcorr(rfit_conv_vals[:,test_ind], template, 30)
            nanxcorr_bool = True
            break
        except:
            nanxcorr_bool = False

    print(nanxcorr_cc)

    if nanxcorr_bool is True:
        plt.figure()
        plt.plot(np.array(nanxcorr_cc, dtype=float))
        plt.title('nanxcorr of rfit conv and template')
        plt.savefig(fig_dir + 'nanxcorr_rfitconv_template.png', dpi=300)
        plt.close()

    # xcorr of two random timepoints
    t1 = 3; t2 = 4;
    nancorr_2time_cc = nanxcorr(rfit_conv_xr.sel(frame=t1).values, rfit_conv_xr.sel(frame=t2).values, 30)
    print(np.shape(nancorr_2time_cc))
    plt.figure()
    plt.plot(np.array(nancorr_2time_cc, dtype=float))
    plt.title('xcorr of time ' + str(t1) + ' and ' + str(t2))
    plt.savefig(fig_dir + 'xcorr_of_two_times.png', dpi=300)
    plt.close()

    # iterative fit to alignment
    # start with mean as template
    # on each iteration, shift individual frames to max xcorr with template
    # then recalculate mean template
    n = np.size(rfit_conv, 1)
    pupil_update = rfit_conv
    total_shift = np.zeros(n,0)
    c = total_shift;
    for rep in range(0,12):
        print('starting on rep ' + str(rep))
        # calculate and plot template
        template = np.nanmedian(pupil_update, 1)
        plt.figure(3,4)
        curent_sub = int('34' + str(rep))
        plt.subplot(curent_sub)
        plt.plot(template)
        plt.title('iter ' + str(rep))

        vidsavepath = os.path.join(fig_dir, str(file_name + '_puprot_worldshift_' + eyeext + '.avi'))
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        vidout = cv2.VideoWriter(vidsavepath, fourcc, 20.0, (int(eyevid.get(cv2.CAP_PROP_FRAME_WIDTH))*2, int(eyevid.get(cv2.CAP_PROP_FRAME_HEIGHT))*2))

        set_size = (int(eyevid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(eyevid.get(cv2.CAP_PROP_FRAME_HEIGHT)))

        while(1):
            # read the frames for this pass through while loop
            eye_ret, eye_frame = eyevid.read()

            frame_num = eyevid.get(cv2.CAP_PROP_POS_FRAMES)

            if not eye_ret:
                break

            if eyevid.get(cv2.CAP_PROP_POS_FRAMES) > 50:
                break

            cur = str(eyevid.get(cv2.CAP_PROP_POS_FRAMES))
            tot = str(eyevid.get(cv2.CAP_PROP_FRAME_COUNT))
            progress = str(int((eyevid.get(cv2.CAP_PROP_POS_FRAMES) / eyevid.get(cv2.CAP_PROP_FRAME_COUNT))*100))
            print('working on frame ' + cur + ' of ' + tot + ' (' + progress + '%)')

            # loop over each frame, take xcorr, and shift accordingly
            xc, lags = nanxcorr(template,pupilUpdate[:,frame_num],10)
            c[frame_num] = np.maximum(xc)
            peaklag = np.argmax(xc)
            peak[frame_num] = lags[peaklag]
            total_shift[frame_num] = total_shift[frame_num] + peak[frame_num]
            pupil_update[:,frame_num] = circ_shift[pupil_update[:,i],peak[frame_num],0]

            win = 3
            shift_nan = -total_shift
            for j in range(0,len(c)):
                shift_nan[j] = np.where(c[j]<0.4, shift_nan[j], np.nan)
            shift_smooth = np.convolve(shift_nan, np.ones(win,1) / win)

            for sp in range(1,5):
                curent_sub = int('22' + str(sp))
                plt.figure(2,2)
                plt.subplot(current_sub)
                if sp == 2 or sp == 4:
                    plt.plot(meancent[0][frame_num] + d + np.cos(shift_smooth[frame_num] + 90), meancent[1][frame_num] + d * np.sin(shift_smooth[frame_num] + 90))
                if sp == 3 or sp ==4:
                    for th in range(1,361):
                        x1 = np.array(mean_cent[0] + (rmin + rfit) * np.cos(th)).astype(int)[0,0]
                        y1 = np.array(mean_cent[1] + (rmin + rfit) * np.sin(th)).astype(int)[0,0]
                        x2 = np.array(mean_cent[0] - (rmin + rfit) * np.cos(th)).astype(int)[0,0]
                        y2 = np.array(mean_cent[1] - (rmin + rfit) * np.sin(th)).astype(int)[0,0]
                        plt.plot([x1,y1],[x2,y2], 'b.')
                plt.title('corr=' + c[frame_num])

            vidout.write(eye_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            vidout.release()
            cv2.destroyAllWindows()

    end = time.time()
    time_to_run = ((end - start_time)/60)
    time_to_run = str(time_to_run)
    print('time to run (min): ' + time_to_run)
