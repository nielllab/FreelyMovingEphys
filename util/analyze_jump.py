"""
analyze_jump.py

jump tracking utilities

Oct. 26, 2020
"""

# package imports
import xarray as xr
import cv2
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm

# module imports
from util.read_data import nanxcorr, find_start_end

# get cross-correlation
def jump_cc(REye_ds, LEye_ds, top_ds, side_ds, config):
    # open pdf file to save plots in
    pdf = PdfPages(os.path.join(config['data_path'], (config['trial_name'] + '_jump_cc.pdf')))
    # to append data to (for making plots of pooled data):
    all_theta = []; all_phi = []; all_div = []
    all_th_gaze = []; all_th_div = []; all_th_phi = []
    # loop through every frame in the right eye view
    print('analyzing frames')
    for frame_num in tqdm(range(0, len(REye_ds))):

        REye_now = REye_ds.sel(frame=frame_num)
        LEye_now = LEye_ds.sel(frame=frame_num)
        Side_now = side_ds.sel(frame=frame_num)

        RTheta = (REye_now.sel(ellipse_param='theta') - np.nanmedian(REye_now.sel(ellipse_param='theta'))).values
        RPhi = (REye_now.sel(ellipse_param='phi') - np.nanmedian(REye_now.sel(ellipse_param='phi'))).values
        LTheta = (LEye_now.sel(ellipse_param='theta') -  np.nanmedian(LEye_now.sel(ellipse_param='theta'))).values
        LPhi = (LEye_now.sel(ellipse_param='phi') - np.nanmedian(LEye_now.sel(ellipse_param='phi'))).values
        head_theta = Side_now.sel(head_param='theta').values

        # zero-center head theta, and get rid of wrap-around effect (mod 360)
        th = head_theta * 180 / np.pi; th = (th + 360) % 360
        th = th - np.nanmean(th); th = -th

        # eye divergence (theta)
        div = 0.5 * (RTheta - LTheta)
        # gaze (mean theta of eyes)
        gaze_th = (RTheta + LTheta) * 0.5
        # gaze (mean phi of eyes)
        gaze_phi = (RPhi + LPhi) * 0.5

        # calculate xcorrs
        th_gaze, lags = nanxcorr(th, gaze_th, 30)
        th_div, lags = nanxcorr(th, div, 30)
        th_phi, lags = nanxcorr(th, gaze_phi, 30)

        # for pooled data
        all_theta.append(th); all_phi.append(gaze_phi); all_div.append(div)
        all_th_gaze.append(th_gaze); all_th_div.append(th_div); all_th_phi.append(th_phi)

        # plots
        plt.figure()
        plt.title(config['trial_name'])
        plt.ylabel('deg'); plt.xlabel('frames')
        plt.legend(['head_theta', 'eye_theta','eye_divergence','eye_phi'])
        plt.plot(th); plt.plot(gaze_th); plt.plot(div); plt.plot(gaze_phi)
        pdf.savefig()
        plt.close()

        plt.figure()
        plt.title('head theta xcorr')
        plt.plot(lags, th_gaze); plt.plot(lags, th_div); plt.plot(lags, th_phi)
        plt.legend(['gaze', 'div', 'phi'])
        pdf.savefig()
        plt.close()

        plt.figure()
        plt.ylabel('eye div deg'); plt.xlabel('head th deg')
        plt.plot([-40,40],[40,-40], 'r:')
        plt.xlim([-40,40]); plt.ylim([-40,40])
        plt.scatter(th, div, '.')
        pdf.savefig()
        plt.close()

        plt.figure()
        plt.ylabel('eye phi deg'); plt.xlabel('head th deg')
        plt.plot([-40,40],[-40,40], 'r:')
        plt.xlim([-40,40]); plt.ylim([-40,40])
        plt.scatter(th, gaze_phi, '.')
        pdf.savefig()
        plt.close()

    # plot pooled data
    # head theta, phi
    plt.figure()
    plt.plot(all_theta, all_phi, '.')
    plt.xlabel('head theta'); fig2.ylabel('phi')
    plt.xlim([-60,60]); plt.ylim([-60,60])
    pdf.savefig()
    plt.close()
    # head theta, eye theta divergence
    plt.figure()
    plt.plot(all_theta, all_div, '.')
    plt.xlabel('head theta'); plt.ylabel('eye theta div')
    plt.xlim([-60,60]); plt.ylim([-60,60])
    pdf.savefig()
    plt.close()
    # xcorr with head angle
    plt.figure()
    plt.errorbar(lags, np.nanmean(all_th_gaze), np.std(all_th_gaze)/np.sqrt(np.size(all_th_gaze)))
    plt.errorbar(lags, np.nanmean(all_th_div), np.std(all_th_div)/np.sqrt(np.size(all_th_div)))
    plt.errorbar(lags, np.nanmean(all_th_phi), np.std(all_th_phi)/np.sqrt(np.size(all_th_phi)))
    plt.ylim([-1,1]); fig4.ylabel('correlation'); fig4.title('xcorr with head angle')
    plt.legend(['mean theta', 'mean theta divergence', 'mean phi'])
    pdf.savefig()
    plt.close()

    pdf.close()

# create movies of pursuit with eye positions
def jump_gaze_trace(REye, LEye, TOP, SIDE, Svid, config):
    
    REye_params = REye.REYE_ellipse_params
    LEye_params = LEye.LEYE_ellipse_params
    Side_pts = SIDE.SIDE_pts
    Side_params = SIDE.SIDE_params

    # find the first shared frame for the four video feeds and play them starting at that shared frame
    # td_startframe, td_endframe, left_startframe, left_endframe, right_startframe, right_endframe, side_startframe, side_endframe, first_real_time, last_real_time = find_start_end(TOP, LEye, REye, SIDE)

    sidecap = cv2.VideoCapture(Svid).set(cv2.CAP_PROP_POS_FRAMES, int(side_startframe))

    width = int(sidecap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(sidecap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    savepath = os.path.join(config['data_path'], (config['trial_name'] + '_side_gaze_trace.avi'))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out_vid = cv2.VideoWriter(savepath, fourcc, 20.0, (width, height))

    for frame_num in tqdm(range(0,int(sidecap.get(cv2.CAP_PROP_FRAME_COUNT)))):
        # read in videos
        SIDE_ret, SIDE_frame = sidecap.read()

        if not SIDE_ret:
            break
        if not TOP_ret:
            break
        if not REye_ret:
            break
        if not LEye_ret:
            break

        # get current ellipse parameters
        framenow = sidecap.get(cv2.CAP_PROP_POS_FRAMES)
        REye_now = REye_params.sel(frame=framenow)
        LEye_now = LEye_params.sel(frame=framenow)
        SIDE_par_now = Sids_params.sel(frame=framenow)
        SIDE_pts_now = Sids_pts.sel(frame=framenow)

        # interpolate time
        REye_interp = REye_now.interplike(other=TOP, method='linear')
        LEye_interp = LEye_now.interplike(other=TOP, method='linear')
        SIDE_par_interp = SIDE_par_now.interplike(other=TOP, method='linear')
        SIDE_pts_interp = SIDE_pts_now.interplike(other=TOP, method='linear')

        # # scale
        # R = R * SIDE_interp.sel(head_param='scaleR') / 50
        # L = L * SIDE_interp.sel(head_param='scaleL') / 50

        # split apart parameters
        RTheta = REye_interp.sel(ellipse_param='theta').values
        RPhi = REye_interp.sel(ellipse_param='phi').values
        LTheta = LEye_interp.sel(ellipse_param='theta').values
        LPhi = LEye_interp.sel(ellipse_param='phi').values
        head_theta = SIDE_par_interp.values

        # zero-center head theta, and get rid of wrap-around effect (mod 360)
        # add pi/8 since this is roughly head tilt in movies relative to mean theta
        th = head_theta - np.nanmedian(head_theta) + np.pi + np.pi/8

        # eye divergence (theta)
        div = 0.5 * (RTheta - LTheta)

        # gaze (mean theta of eyes)
        gaze_th = (RTheta + LTheta) * 0.5
        # gaze (mean phi of eyes)
        gaze_phi = (RPhi + LPhi) * 0.5

        # plot mouse head poisiton with small blue 'tracers'
        for i in range(0,15):
            frame_before = framenow - i
            SIDE_before = SIDE_pts_interp.sel(frame=frame_before)
            head_x = SIDE_before.sel(point_loc='LEye_x').values
            head_y = SIDE_before.sel(point_loc='LEye_y').values
            SIDE_frame = cv2.point(SIDE_frame, (head_x,head_y), (0,0,255), 1)

        # blue circle over the current position of the eye
        SIDE_current = SIDE_pts_interp.sel(frame=frame_before)
        eyecent_x = SIDE_current.sel(point_loc='LEye_x').values
        eyecent_y = SIDE_current.sel(point_loc='LEye_y').values
        SIDE_frame = cv2.point(SIDE_frame, (eyecent_x,eyecent_y), (0,0,255), 6)

        # calculate and plot head vector
        hx = 200 * np.cos(th)
        hy = 200 * np.sin(th)
        # plot head vector
        headV_x1 = SIDE_pts_interp.sel(head_params='LEye_x').values
        headV_y1 = SIDE_pts_interp.sel(head_params='LEye_y').values
        headV_x2 = SIDE_pts_interp.sel(head_params='LEye_x').values + hx
        headV_y2 = SIDE_pts_interp.sel(head_params='LEye_y').values + hy
        # black line of the head vector
        SIDE_frame = cv2.line(SIDE_frame, (headV_x1,headV_y1), (headV_x2,headV_y2), (255,255,255), thickness=2)

        # calculate gaze direction (head and eyes)
        # subtract off the pi/8 that was added above
        rth = th - div * np.pi/180 - np.pi/8
        gazeV_x1 = SIDE_pts_interp.sel(head_params='LEye_x').values
        gazeV_y1 = SIDE_pts_interp.sel(head_params='LEye_y').values
        gazeV_x2 = SIDE_pts_interp.sel(head_params='LEye_x').values + 200 * np.cos(rth)
        gazeV_y2 = SIDE_pts_interp.sel(head_params='LEye_y').values + 200 * np.sin(rth)
        # cyan line of gaze direction
        SIDE_frame = cv2.line(SIDE_frame, (gazeV_x1,gazeV_y1), (gazeV_x2,gazeV_y2), (0,255,255), thickness=2)

        vidout.write(SIDE_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        out_vid.release()
        cv2.destroyAllWindows()
