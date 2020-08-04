"""
FreelyMovingEphys world tracking utilities
track_world.py

Last modified August 03, 2020
"""

# package imports
import xarray as xr
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import cv2
from scipy import signal
from scipy.optimize import curve_fit
import scipy.stats

# module imports
from util.read_data import open_h5, open_time, read_paths, read1path

# get the mean confidence interval
def find_ci(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h, m-h, m+h

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

# find pupil edge and align over time to calculate cyclotorsion
# all inputs must be deinterlaced
def find_pupil_rotation(data_path, file_name, eyeext, topext, worldext, eye_ds, save_path, world_interp_method, ranger):
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

    eye_raw_theta = eye_ell_params.sel(ellipse_params='theta')
    eye_raw_phi = eye_ell_params.sel(ellipse_params='phi')
    eye_raw_longaxis= eye_ell_params.sel(ellipse_params='longaxis')
    eye_raw_shortaxis = eye_ell_params.sel(ellipse_params='shortaxis')

    eyeTSminusstart = [(t-start_time).seconds for t in eyeTS]
    worldTSminusstart = [(t-start_time).seconds for t in worldTS]

    print('opening videos')
    worldvid = cv2.VideoCapture(worldvidpath)
    topvid = cv2.VideoCapture(top1vidpath)
    eyevid = cv2.VideoCapture(eyevidpath)

    # setup the file to save out of this
    vidsavepath = os.path.join(fig_dir, str(file_name + '_worldshift_' + eyeext + '.avi'))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    vidout = cv2.VideoWriter(vidsavepath, fourcc, 20.0, (int(eyevid.get(cv2.CAP_PROP_FRAME_WIDTH))*2, int(eyevid.get(cv2.CAP_PROP_FRAME_HEIGHT))*2))

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

        cur = str(eyevid.get(cv2.CAP_PROP_POS_FRAMES))
        tot = str(eyevid.get(cv2.CAP_PROP_FRAME_COUNT))
        progress = str(int((eyevid.get(cv2.CAP_PROP_POS_FRAMES) / eyevid.get(cv2.CAP_PROP_FRAME_COUNT))*100))
        print('working on frame ' + cur + ' of ' + tot + ' (' + progress + '%)')

        wrld_frame = cv2.cvtColor(wrld_frame, cv2.COLOR_BGR2GRAY)
        eye_frame = cv2.cvtColor(eye_frame, cv2.COLOR_BGR2GRAY)
        top_frame = cv2.cvtColor(top_frame, cv2.COLOR_BGR2GRAY)

        # get ellisepe parameters for this time
        current_time = eyevid.get(cv2.CAP_PROP_POS_FRAMES)
        current_theta = eye_theta.sel(frame=current_time).values[0]
        current_phi = eye_phi.sel(frame=current_time).values[0]
        current_longaxis = eye_longaxis.sel(frame=current_time).values[0]
        current_shortaxis = eye_shortaxis.sel(frame=current_time).values[0]

        dlc_pts_thistime = eye_pts.sel(frame=current_time)
        dlc_names = dlc_pts_thistime.coords['point_loc'].values
        dlc_x_names = [name for name in dlc_names if '_x' in name]
        dlc_y_names = [name for name in dlc_names if '_y' in name]

        # get nanmean of x and y in (y, x) tuple as center of ellipse
        x_val = []; y_val = []
        for ptpairnum in range(0, len(dlc_x_names)):
            x_val.append(dlc_pts_thistime.sel(point_loc=dlc_x_names[ptpairnum]).values)
            y_val.append(dlc_pts_thistime.sel(point_loc=dlc_y_names[ptpairnum]).values)
        mean_cent = (int(np.nanmean(x_val)), int(np.nanmean(y_val)))

        ci = []; params = []
        for th in range(0, 360):
            meanr = 0.5 * (current_longaxis + current_shortaxis)

            r = range(int(meanr - ranger), int(meanr + ranger))

            # go out along radius and get pixel values
            pupil_edge = np.zeros([360, len(r)])
            for i in range(0, len(r)):
                pupil_edge[th,:] = eye_frame[int(mean_cent[1]+r[i]*(np.sin(th))), int(mean_cent[0]+r[i]*(np.cos(th)))]

            # fit sigmoind to pupil edge at this theta
            d = pupil_edge[th,:]
            init_params = [100,200,10,0.5]
            # non-linear regression
            popt, pcov = curve_fit(curve_func, xdata=range(0,len(d)), ydata=d, p0=init_params)
            # confidence interval of the parameters
            ypred, delta, ypred_lowerci, ypred_upperci = find_ci(popt)
            ci.append(delta)
            params.append(popt)

        fit_thresh = 1
        params = np.array(params)
        # extract radius variable from parameters
        rfit = np.vstack([np.array(np.zeros(len(ci))), (params[:,2] - 1)])

        # if confidence interval in estimate is > fit_thresh pix, set to to NaN
        # then, remove if luminance goes the wrong way (e.g. from reflectance)
        for j in range(0,len(ci)):
            rfit[:,j] = np.where(ci[j] > fit_thresh, rfit[:,j], np.nan)
            rfit[:,j] = np.where(ci[j] < 0, rfit[:,j], np.nan)

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

        plot_color1 = (0, 255, 255)
        rmin = 0.5 * (current_longaxis + current_shortaxis) - ranger
        th = range(0,360)
        x = mean_cent[0] + (rmin + rfit) * np.cos(th)
        y = mean_cent[1] + (rmin + rfit) * np.sin(th)
        x = np.array(x); y = np.array(y);
        eye_frame = cv2.line(eye_frame, (int(np.array(x[0,0])), int(np.array(x[1,0]))), (int(np.array(y[0,0])), int(np.array(y[1,0]))), plot_color1, thickness=4)

#         template = np.nanmean(rfit_conv[:,100:120,1])

        # ...saftey check plots of cross correlation will go here...

        vidout.write(eye_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        vidout.release()
        cv2.destroyAllWindows()

#     plt.figure()
#     plt.imagesc(coorcoef)
