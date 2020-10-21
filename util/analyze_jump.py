"""
analyze_jump.py

Jump tracking utilities

Sept. 07, 2020
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

# module imports
from util.track_world import nanxcorr

# get cross-correlation
def jump_cc(global_data_path, global_save_path, trial_name, REye_ds, LEye_ds, top_ds, side_ds):
    # open pdf file to save plots in
    pp = PdfPages(os.path.join(global_save_path, (key + '_jump_cc.pdf')))
    # to append data to (for making plots of pooled data):
    all_theta = []; all_phi = []; all_div = []
    all_th_gaze = []; all_th_div = []; all_th_phi = []
    # loop through every frame in the right eye view
    for frame_num in range(0, len(REye_ds)):

        REye_now = REye_ds.sel(frame=frame_num)
        LEye_now = LEye_ds.sel(frame=frame_num)

        RTheta = REye_now.sel(ellipse_param='theta') - np.nanmedian(REye_now.sel(ellipse_param='theta'))
        RPhi = REye_now.sel(ellipse_param='phi') - np.nanmedian(REye_now.sel(ellipse_param='phi'))
        LTheta = LEye_now.sel(ellipse_param='theta') -  np.nanmedian(LEye_now.sel(ellipse_param='theta'))
        LPhi = LEye_now.sel(ellipse_param='phi') - np.nanmedian(LEye_now.sel(ellipse_param='phi'))
        head_theta = SIDE_interp.sel(head_param='theta').values
        RTheta = RTheta.values; RPhi = RPhi.values
        LTheta = LTheta.values; LPhi = LPhi.values

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

        # plot
        fig1 = plt.figure(constrained_layout=True)
        gs = fig1.add_gridspec(2, 3)
        f1_ax1 = fig1.add_subplot(gs[0, 2])
        f1_ax1.set_title(trial_name)
        f1_ax1.ylabel('deg'); f1_ax1.xlabel('frames')
        f1_ax1.legend(['head_theta', 'eye_theta','eye_divergence','eye_phi'])
        f1_ax1.plot(th); f1_ax1.plot(gaze_th); f1_ax1.plot(div); f1_ax1.plot(gaze_phi);
        f1_ax2 = fig1.add_subplot(gs[1, 0])
        f1_ax2.set_title('head theta xcorr')
        f1_ax2.plot(lags, th_gaze); f1_ax2.plot(lags, th_div); f1_ax2.plot(lags, th_phi)
        f1_ax2.legend(['gaze', 'div', 'phi'])
        f1_ax3 = fig1.add_subplot(gs[1, 1])
        f1_ax3.ylabel('eye div deg'); f1_ax3.xlabel('head th deg')
        f1_ax3.plot([-40,40],[40,-40], 'r:')
        f1_ax3.xlim([-40,40]); f1_ax3.ylim([-40,40])
        f1_ax3.scatter(th, div, '.')
        f1_ax4 = fig1.add_subplot(gs[1, 2])
        f1_ax4.ylabel('eye phi deg'); f1_ax4.xlabel('head th deg')
        f1_ax4.plot([-40,40],[-40,40], 'r:')
        f1_ax4.xlim([-40,40]); f1_ax4.ylim([-40,40])
        f1_ax4.scatter(th, gaze_phi, '.')
        fig1.savefig(pp, format='pdf')

    # plot pooled data
    # head theta, phi
    fig2 = plt.figure()
    fig2.plot(all_theta, all_phi, '.')
    fig2.xlabel('head theta'); fig2.ylabel('phi')
    fig2.xlim([-60,60]); plt.ylim([-60,60])
    fig2.savefig(pp, format='pdf')
    # head theta, eye theta divergence
    fig3 = plt.figure()
    fig3.plot(all_theta, all_div, '.')
    fig3.xlabel('head theta'); fig3.ylabel('eye theta div')
    fig3.xlim([-60,60]); plt.ylim([-60,60])
    fig3.savefig(pp, format='pdf')
    # xcorr with head angle
    fig4 = plt.plot()
    fig4.errorbar(lags, np.nanmean(all_th_gaze), np.std(all_th_gaze)/np.sqrt(np.size(all_th_gaze)))
    fig4.errorbar(lags, np.nanmean(all_th_div), np.std(all_th_div)/np.sqrt(np.size(all_th_div)))
    fig4.errorbar(lags, np.nanmean(all_th_phi), np.std(all_th_phi)/np.sqrt(np.size(all_th_phi)))
    fig4.ylim([-1,1]); fig4.ylabel('correlation'); fig4.title('xcorr with head angle')
    fig4.legend(['mean theta', 'mean theta divergence', 'mean phi'])

    pp.close()

# create movies of pursuit with eye positions
def jump_gaze_trace(datapath, savepath, trialname, REye, LEye, TOP, SIDE, Rvid, Lvid, Svid, Tvid):
    # setup the file to save out of this

    savepath = str(savepath) + '/' + str(trial_name) + '_gaze_trace.avi'
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out_vid = cv2.VideoWriter(savepath, fourcc, 20.0, (width, height))

    # set colors
    plot_color0 = (225, 255, 0)
    plot_color1 = (0, 255, 255)

    # find the first shared frame for the four video feeds and play them starting at that shared frame
    td_startframe, td_endframe, left_startframe, left_endframe, right_startframe, right_endframe, side_startframe, side_endframe, first_real_time, last_real_time = find_start_end(TOP, LEye, REye, SIDE)

    sidecap = cv2.VideoCapture(Svid).set(cv2.CAP_PROP_POS_FRAMES, int(side_startframe))
    leftcap = cv2.VideoCapture(Lvid).set(cv2.CAP_PROP_POS_FRAMES, int(left_startframe))
    rightcap = cv2.VideoCapture(Rvid).set(cv2.CAP_PROP_POS_FRAMES, int(right_startframe))
    topcap = cv2.VideoCapture(Tvid).set(cv2.CAP_PROP_POS_FRAMES, int(td_startframe))

    while(1):
        # read in videos
        SIDE_ret, SIDE_frame = sidevid.read()
        TOP_ret, TOP_frame = sidevid.read()
        REye_ret, REye_frame = sidevid.read()
        LEye_ret, LEye_frame = sidevid.read()

        if not SIDE_ret:
            break
        if not TOP_ret:
            break
        if not REye_ret:
            break
        if not LEye_ret:
            break

        # get current ellipse parameters
        framenow = SIDE_frame.get(cv2.CAP_PROP_POS_FRAMES)
        REye_now = REye.sel(frame=framenow)
        LEye_now = LEye.sel(frame=framenow)
        SIDE_now = SIDE.sel(frame=framenow)

        # interpolate time
        REye_interp = REye_now.interplike(other=TOP, method='linear')
        LEye_interp = LEye_now.interplike(other=TOP, method='linear')
        SIDE_interp = SIDE_now.interplike(other=TOP, method='linear')

        # scale
        R = R * SIDE_interp.sel(head_param='scaleR') / 50
        L = L * SIDE_interp.sel(head_param='scaleL') / 50

        # split apart parameters
        RTheta = REye_interp.sel(ellipse_param='theta').values
        RPhi = REye_interp.sel(ellipse_param='phi').values
        LTheta = LEye_interp.sel(ellipse_param='theta').values
        LPhi = LEye_interp.sel(ellipse_param='phi').values
        head_theta = SIDE_interp.sel(head_param='theta').values

        # zero-center head theta, and get rid of wrap-around effect (mod 360)
        # add pi/8 since this is roughly head tilt in movies relative to mean theta
        th = head_theta - np.nanmedian(head_theta) + np.pi + np.pi/8

        # eye divergence (theta)
        div = 0.5 * (RTheta - LTheta)

        # gaze (mean theta of eyes)
        gaze_th = (RTheta + LTheta) * 0.5
        # gaze (mean phi of eyes)
        gaze_phi = (RPhi + LPhi) * 0.5

        # plot mouse head poisiton with 'tracers'
        for i in range(0,15):
            frame_before = framenow - i
            SIDE_before = SIDE.sel(frame=frame_before)
            head_x = SIDE_before.sel(head_params='x').values
            head_y = SIDE_before.sel(head_params='y').values
            SIDE_frame = cv2.point(SIDE_frame, (head_x,head_y), (0,0,255), 1)
        # blue circle over the current position of the eye
        SIDE_frame = cv2.point(SIDE_frame, (head_x1,head_y1), (0,0,255), 6)

        # calculate and plot head vector
        hx = 200 * np.cos(th)
        hy = 200 * np.sin(th)
        # plot head vector
        headV_x1 = SIDE_interp.sel(head_params='x').values
        headV_y1 = SIDE_interp.sel(head_params='y').values
        headV_x2 = SIDE_interp.sel(head_params='x').values + hx
        headV_y2 = SIDE_interp.sel(head_params='y').values + hy
        # black line of the head vector
        SIDE_frame = cv2.line(SIDE_frame, (headV_x1,headV_y1), (headV_x2,headV_y2), (255,255,255), thickness=2)

        # calculate gaze direction (head and eyes)
        # subtract off the pi/8 that was added above
        rth = th - div * np.pi/180 - np.pi/8
        gazeV_x1 = SIDE_interp.sel(head_params='x').values
        gazeV_y1 = SIDE_interp.sel(head_params='y').values
        gazeV_x2 = SIDE_interp.sel(head_params='x').values + 200 * np.cos(rth)
        gazeV_y2 = SIDE_interp.sel(head_params='y').values + 200 * np.sin(rth)
        # cyan line of gaze direction
        SIDE_frame = cv2.line(SIDE_frame, (gazeV_x1,gazeV_y1), (gazeV_x2,gazeV_y2), (0,255,255), thickness=2)

        vidout.write(SIDE_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        out_vid.release()
        cv2.destroyAllWindows()
