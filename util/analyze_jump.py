"""
analyze_jump.py

jump tracking analysis

Dec. 16, 2020
"""
# package imports
import xarray as xr
import cv2
import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.animation import FFMpegWriter
from tqdm import tqdm
# module imports
from util.paths import find
from util.aux_funcs import nanxcorr
from util.time import find_start_end

# get figures and process data for individual jump recordings
def jump_cc(REye_ds, LEye_ds, top_ds, side_ds, time, meta, config):
    # handle jump timing metadata
    jump_num = config['recording_name'].split('_')[-1].lstrip('0') # get the jump number without preceing 0
    vals = [] # the values in the dictionary for this jump
    cam_points = [] # theentries in time metadata dictionary
    for cam_point in time:
        cam_values = time[cam_point]
        vals.append(cam_values[jump_num])
        cam_points.append(cam_point)
    time_dict = {cam_points[i] : vals[i] for i in range(len(cam_points))} # make the dictionary for only this jump
    # then do it for the recording's general metadata and merge the time metadata in
    trial_info = meta['trial_info']
    jump_info = trial_info[int(jump_num)-1]
    jump_info.update(time_dict)

    # open pdf file to save plots in
    pdf = PdfPages(os.path.join(config['trial_head'], (config['recording_name'] + '_jump_cc.pdf')))

    # organize data
    REye = REye_ds.REYE_ellipse_params
    LEye = LEye_ds.LEYE_ellipse_params
    head_pitch = side_ds.SIDE_theta

    # zero-center theta and phi for each eye, and flip sign of phi
    RTheta = np.rad2deg(REye.sel(ellipse_params='theta')) - np.rad2deg(np.nanmedian(REye.sel(ellipse_params='theta')))
    RPhi = (np.rad2deg(REye.sel(ellipse_params='phi')) - np.rad2deg(np.nanmedian(REye.sel(ellipse_params='phi'))))
    LTheta = np.rad2deg(LEye.sel(ellipse_params='theta')) -  np.rad2deg(np.nanmedian(LEye.sel(ellipse_params='theta')))
    LPhi = (np.rad2deg(LEye.sel(ellipse_params='phi')) - np.rad2deg(np.nanmedian(LEye.sel(ellipse_params='phi'))))

    # zero-center head pitch, and get rid of wrap-around effect (mod 360)
    pitch = np.rad2deg(head_pitch)
    pitch = ((pitch+360) % 360)
    pitch = pitch - np.nanmean(pitch) # have to mean center as last step so that it's comparable on plots--is it okay for this to be different in video plotting? I would think so

    # interpolate over eye paramters to match head pitch
    RTheta_interp = RTheta.interp_like(pitch, method='linear')
    RPhi_interp = RPhi.interp_like(pitch, method='linear')
    LTheta_interp = LTheta.interp_like(pitch, method='linear')
    LPhi_interp = LPhi.interp_like(pitch, method='linear')

    # plot to check interpolation
    # plt.subplots(4,1)
    # plt.subplot(411)
    # plt.plot(RTheta_interp.values); plt.plot(RTheta); plt.plot(pitch)
    # plt.legend(['interp','raw','head_pitch']); plt.title('RTheta')
    # plt.subplot(412)
    # plt.plot(RPhi_interp.values); plt.plot(RPhi); plt.plot(pitch)
    # plt.title('RPhi')
    # plt.subplot(413)
    # plt.plot(LTheta_interp.values); plt.plot(LTheta); plt.plot(pitch)
    # plt.title('LTheta')
    # plt.subplot(414)
    # plt.plot(LPhi_interp.values); plt.plot(LPhi); plt.plot(pitch)
    # plt.title('LPhi')
    # plt.tight_layout()
    # pdf.savefig()
    # plt.close()

    # eye divergence (theta)
    div = (RTheta_interp - LTheta_interp) * 0.5
    # gaze (mean theta of eyes)
    gaze_th = (RTheta_interp + LTheta_interp) * 0.5
    # gaze (mean phi of eyes)
    gaze_phi = (RPhi_interp + LPhi_interp) * 0.5

    # correct lengths when off
    pitch_len = len(pitch.values); gaze_th_len = len(gaze_th.values); div_len = len(div.values); gaze_phi_len = len(gaze_phi.values)
    min_len = np.min([pitch_len, gaze_th_len, div_len, gaze_phi_len])
    max_len = np.max([pitch_len, gaze_th_len, div_len, gaze_phi_len])
    if max_len != min_len:
        pitch = pitch.isel(frame=range(0,min_len))
        gaze_th = gaze_th.isel(frame=range(0,min_len))
        div = div.isel(frame=range(0,min_len))
        gaze_phi = gaze_phi.isel(frame=range(0,min_len))

    # calculate xcorrs
    th_gaze, lags = nanxcorr(pitch.values, gaze_th.values, 30)
    th_div, lags = nanxcorr(pitch.values, div.values, 30)
    th_phi, lags = nanxcorr(pitch.values, gaze_phi.values, 30)

    # make an xarray of this trial's data to be used in pooled analysis
    trial_outputs = pd.DataFrame([pitch, gaze_th, div, gaze_phi,th_gaze,th_div,th_phi]).T
    trial_outputs.columns = ['head_pitch','mean_eye_th','eye_th_div','mean_eye_phi','th_gaze','th_div','th_phi']
    trial_xr = xr.DataArray(trial_outputs, dims=['frame','jump_params'])
    trial_xr.attrs['jump_metadata'] = str(jump_info)

    # plots
    plt.figure()
    plt.title(config['recording_name'])
    plt.ylabel('deg'); plt.xlabel('frames')
    plt.plot(pitch); plt.plot(gaze_th); plt.plot(div); plt.plot(gaze_phi)
    plt.legend(['head_pitch', 'eye_theta','eye_divergence','eye_phi'])
    pdf.savefig()
    plt.close()

    plt.figure()
    plt.subplots(2,1)
    plt.subplot(211)
    plt.plot(LTheta_interp)
    plt.plot(RTheta_interp)
    plt.plot(gaze_th)
    plt.title('theta interp_like pitch')
    plt.legend(['left','right','mean'])
    plt.subplot(212)
    plt.plot(LPhi_interp)
    plt.plot(RPhi_interp)
    plt.plot(gaze_phi)
    plt.title('phi interp_like pitch')
    plt.legend(['left','right','mean'])
    plt.tight_layout()
    pdf.savefig()

    plt.figure()
    plt.title('head_pitch xcorr')
    plt.plot(lags, th_gaze); plt.plot(lags, th_div); plt.plot(lags, th_phi)
    plt.legend(['gaze', 'div', 'phi'])
    pdf.savefig()
    plt.close()

    plt.figure()
    plt.ylabel('eye div deg'); plt.xlabel('head pitch deg')
    plt.plot([-30,30],[30,-30], 'r:')
    plt.xlim([-30,30]); plt.ylim([-30,30])
    plt.scatter(pitch, div)
    pdf.savefig()
    plt.close()

    plt.figure()
    plt.ylabel('eye phi deg'); plt.xlabel('head pitch deg')
    plt.plot([-30,30],[-30,30], 'r:')
    plt.xlim([-30,30]); plt.ylim([-30,30])
    plt.scatter(pitch, gaze_phi)
    pdf.savefig()
    plt.close()

    # if config['save_avi_vids'] is True:

    # make an animated plot of these parameters
    print('saving animated plots')
    fig, (ax1,ax2,ax3,ax4) = plt.subplots(4,1)
    ax1.plot(pitch, 'b-'); ax1.set_title('pitch')
    ax2.plot(gaze_th, 'b-'); ax2.set_title('mean th')
    ax3.plot(gaze_phi, 'b-'); ax3.set_title('mean phi')
    ax4.plot(div, 'b-'); ax4.set_title('th div')
    plt.tight_layout()
    pdf.savefig()

    ani_save_path = os.path.join(config['trial_head'], (config['recording_name'] + '_params_animation.avi'))
    writer = FFMpegWriter(fps=60, bitrate=-1)
    with writer.saving(fig, ani_save_path, 150):
        for t in tqdm(range(0,len(pitch))):
            ln1 = ax1.vlines(t,-40,40)
            ln2 = ax2.vlines(t,-40,40)
            ln3 = ax3.vlines(t,-40,40)
            ln4 = ax4.vlines(t,-40,40)
            writer.grab_frame()
            ln1.remove(); ln2.remove(); ln3.remove(); ln4.remove()

        # plt.close('all')
    pdf.close()
    return trial_xr

# make plots using the pooled jumping data
def pooled_jump_analysis(pooled, config):

    pdf = PdfPages(os.path.join(config['data_path'], 'pooled_jump_plots.pdf'))
    
    # convert to dataarray so that indexing can be done accross recordings
    # this is only needed if there's more than one trial read in, so a try/except is used
    try:
        pooled_da = pooled.to_array()
    except AttributeError:
        pooled_da = pooled
    # then, get data out for each parameter
    all_pitch = pooled_da.sel(jump_params='head_pitch').values
    all_phi = pooled_da.sel(jump_params='mean_eye_phi').values
    all_div = pooled_da.sel(jump_params='eye_th_div').values
    all_th_gaze = pooled_da.sel(jump_params='th_gaze', frame=range(60)).values
    all_th_div = pooled_da.sel(jump_params='th_div', frame=range(60)).values
    all_th_phi = pooled_da.sel(jump_params='th_phi', frame=range(60)).values
    lags = range(-30, 30)
    
    # head theta, phi
    plt.figure()
    plt.plot(all_pitch, all_phi, 'k.')
    plt.xlabel('head theta'); plt.ylabel('phi')
    plt.xlim([-60,60]); plt.ylim([-60,60])
    pdf.savefig()
    plt.close()
    # head theta, eye theta divergence
    plt.figure()
    plt.plot(all_pitch, all_div, 'k.')
    plt.xlabel('head theta'); plt.ylabel('eye theta div')
    plt.xlim([-60,60]); plt.ylim([-60,60])
    pdf.savefig()
    plt.close()
    # xcorr with head angle
    plt.figure()
    plt.errorbar(lags, np.mean(all_th_gaze,0), yerr=(np.std(np.array(all_th_gaze,dtype=np.float64),0)/np.sqrt(np.size(all_th_gaze,0))))
    plt.errorbar(lags, np.mean(all_th_div,0), yerr=(np.std(np.array(all_th_div,dtype=np.float64),0)/np.sqrt(np.size(all_th_div,0))))
    plt.errorbar(lags, np.mean(all_th_phi,0), yerr=(np.std(np.array(all_th_phi,dtype=np.float64),0)/np.sqrt(np.size(all_th_phi,0))))
    plt.ylim([-1,1]); plt.ylabel('correlation'); plt.title('xcorr with head pitch')
    plt.legend(['mean theta', 'theta divergence', 'mean phi'])
    pdf.savefig()
    plt.close()

    pdf.close()

# create movies of pursuit with eye positions
def jump_gaze_trace(REye, LEye, TOP, SIDE, Svid, config):
    
    REye_params = REye.REYE_ellipse_params
    LEye_params = LEye.LEYE_ellipse_params
    Side_pts = SIDE.SIDE_pts
    Side_params = SIDE.SIDE_theta

    # find the first shared frame for the four video feeds and play them starting at that shared frame
    # td_startframe, td_endframe, left_startframe, left_endframe, right_startframe, right_endframe, side_startframe, side_endframe, first_real_time, last_real_time = find_start_end(TOP, LEye, REye, SIDE)

    sidecap = cv2.VideoCapture(Svid) #.set(cv2.CAP_PROP_POS_FRAMES, int(side_startframe))

    width = int(sidecap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(sidecap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    savepath = os.path.join(config['trial_head'], (config['recording_name'] + '_side_gaze_trace.avi'))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    vid_out = cv2.VideoWriter(savepath, fourcc, 60.0, (width, height))

    # interpolate time
    REye_interp = REye_params.interp_like(other=TOP, method='linear')
    LEye_interp = LEye_params.interp_like(other=TOP, method='linear')
    SIDE_par_interp = Side_params.interp_like(other=TOP, method='linear')
    SIDE_pts_interp = Side_pts.interp_like(other=TOP, method='linear')

    for frame_num in tqdm(range(0,int(sidecap.get(cv2.CAP_PROP_FRAME_COUNT)))):
        # read in videos
        SIDE_ret, SIDE_frame = sidecap.read()

        if not SIDE_ret:
            break

        # get current ellipse parameters
        REye = REye_interp.sel(frame=frame_num)
        LEye = LEye_interp.sel(frame=frame_num)
        SIDE_par_now = SIDE_par_interp.sel(frame=frame_num)
        SIDE_pts_now = SIDE_pts_interp.sel(frame=frame_num)

        # split apart parameters
        RTheta = REye.sel(ellipse_params='theta').values
        RPhi = REye.sel(ellipse_params='phi').values
        LTheta = LEye.sel(ellipse_params='theta').values
        LPhi = LEye.sel(ellipse_params='phi').values
        head_theta = SIDE_par_now.values

        # zero-center head theta, and get rid of wrap-around effect (mod 360)
        # add pi/8 since this is roughly head tilt in movies relative to mean theta
        th = head_theta - (np.nanmedian(head_theta) + np.pi + np.pi/8)

        # eye divergence (theta)
        div = (RTheta - LTheta) * 0.5
        # gaze (mean theta of eyes)
        gaze_th = (RTheta + LTheta) * 0.5
        # gaze (mean phi of eyes)
        gaze_phi = (RPhi + LPhi) * 0.5

        # plot mouse head position with small blue 'tracers'
        for i in range(0,20):
            frame_before = frame_num - i
            if frame_before >= 0:
                head_x = SIDE_pts_interp.sel(point_loc='LEye_x', frame=frame_before).values
                head_y = SIDE_pts_interp.sel(point_loc='LEye_y', frame=frame_before).values
                try:
                    SIDE_frame = cv2.circle(SIDE_frame, (int(head_x),int(head_y)), 2, (255,0,0), -1)
                except ValueError:
                    pass

        # blue circle over the current position of the eye
        eyecent_x = SIDE_pts_now.sel(point_loc='LEye_x').values
        eyecent_y = SIDE_pts_now.sel(point_loc='LEye_y').values
        try:
            SIDE_frame = cv2.circle(SIDE_frame, (int(eyecent_x),int(eyecent_y)), 4, (255,0,0), -1)
        except ValueError:
            pass

        # calculate and plot head vector
        headV_x1 = SIDE_pts_now.sel(point_loc='LEye_x').values
        headV_y1 = SIDE_pts_now.sel(point_loc='LEye_y').values
        headV_x2 = SIDE_pts_now.sel(point_loc='LEye_x').values + 200 * np.cos(th)
        headV_y2 = SIDE_pts_now.sel(point_loc='LEye_y').values + 200 * np.sin(th)
        # black line of the head vector
        try:
            SIDE_frame = cv2.line(SIDE_frame, (int(headV_x1),int(headV_y1)), (int(headV_x2),int(headV_y2)), (0,0,0), thickness=2)
        except ValueError:
            pass

        # calculate gaze direction (head and eyes)
        # subtract off the pi/8 that was added above
        rth = th - div * np.pi/180 - np.pi/8
        # rth = (th - div) + np.pi/8
        gazeV_x1 = SIDE_pts_now.sel(point_loc='LEye_x').values
        gazeV_y1 = SIDE_pts_now.sel(point_loc='LEye_y').values
        gazeV_x2 = SIDE_pts_now.sel(point_loc='LEye_x').values + 200 * np.cos(rth)
        gazeV_y2 = SIDE_pts_now.sel(point_loc='LEye_y').values + 200 *np.sin(rth)
        # cyan line of gaze direction
        try:
            SIDE_frame = cv2.line(SIDE_frame, (int(gazeV_x1),int(gazeV_y1)), (int(gazeV_x2),int(gazeV_y2)), (255,255,0), thickness=2)
        except ValueError:
            pass

        vid_out.write(SIDE_frame)

    vid_out.release()

def animated_gaze_plot(REye, LEye, Top, SIDE, Side_vid_path, LEye_vid_path, REye_vid_path, Top_vid_path, config):
    # get the parameters out for eyes and side
    REye_params = REye.REYE_ellipse_params
    REye_pts = REye.REYE_pts
    LEye_params = LEye.LEYE_ellipse_params
    LEye_pts = LEye.LEYE_pts
    Side_pts = SIDE.SIDE_pts
    Side_params = SIDE.SIDE_theta
    Top_pts = Top.Top_pts
    Top_head_th = Top.Top_head_angle
    Top_body_th = Top.Top_body_angle

    # read in each camera and get out the size that it'll need to be set to
    ani_save_path = os.path.join(config['trial_head'], (config['recording_name'] + '_params_animation.avi'))
    ani_vid = cv2.VideoCapture(ani_save_path)
    Top_vid = cv2.VideoCapture(Top_vid_path)
    REye_vid = cv2.VideoCapture(REye_vid_path)
    LEye_vid = cv2.VideoCapture(LEye_vid_path)
    Side_vid = cv2.VideoCapture(Side_vid_path)
    set_width = int(REye_vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    set_height = int(REye_vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

    savepath = os.path.join(config['trial_head'], (config['recording_name'] + '_animated_gaze_plot.avi'))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    vid_out = cv2.VideoWriter(savepath, fourcc, 60.0, (set_width*4, set_height* 3))

    for frame_num in tqdm(range(0,int(Top_vid.get(cv2.CAP_PROP_FRAME_COUNT)))):
        # read in videos
        SIDE_ret, SIDE_frame = Side_vid.read()
        TOP_ret, TOP_frame = Top_vid.read()
        LEYE_ret, LEYE_frame = LEye_vid.read()
        REYE_ret, REYE_frame = REye_vid.read()
        ani_ret, ani_frame = ani_vid.read()

        for ret in [SIDE_ret, TOP_ret, LEYE_ret, REYE_ret, ani_ret]:
            if not ret:
                break

        # get current parameters
        try:
            REye_now = REye_params.sel(frame=frame_num)
            LEye_now = LEye_params.sel(frame=frame_num)
            REye_pts_now = REye_pts.sel(frame=frame_num)
            LEye_pts_now = LEye_pts.sel(frame=frame_num)
            SIDE_par_now = Side_params.sel(frame=frame_num)
            SIDE_pts_now = Side_pts.sel(frame=frame_num)
            TOP_pts_now = Top_pts.sel(frame=frame_num)
            TOP_head_th_now = Top_head_th.sel(frame=frame_num)
            TOP_body_th_now = Top_body_th.sel(frame=frame_num)
        except KeyError:
            break

        # split apart eye ellipse parameters
        RTheta = REye_now.sel(ellipse_params='theta').values
        RPhi = REye_now.sel(ellipse_params='phi').values
        LTheta = LEye_now.sel(ellipse_params='theta').values
        LPhi = LEye_now.sel(ellipse_params='phi').values
        head_pitch = SIDE_par_now.values
        
        # plot ellipse and dlc points on the REYE frame
        try:
            # get out ellipse parameters and plot them on the video
            ellipse_axes = (int(REye_now.sel(ellipse_params='longaxis').values), int(REye_now.sel(ellipse_params='shortaxis').values))
            ellipse_phi = int(np.rad2deg(REye_now.sel(ellipse_params='ellipse_phi').values))
            ellipse_cent = (int(REye_now.sel(ellipse_params='X0').values), int(REye_now.sel(ellipse_params='Y0').values))
            REYE_frame = cv2.ellipse(REYE_frame, ellipse_cent, ellipse_axes, ellipse_phi, 0, 360, (255,0,0), 2) # ellipse in blue
        except (ValueError, KeyError):
            pass
        try:
            for k in range(0, len(REye_pts_now), 3):
                pt_cent = (int(REye_pts_now.isel(point_loc=k).values), int(REye_pts_now.isel(point_loc=k+1).values))
                if REye_pts_now.isel(point_loc=k+2).values < config['lik_thresh']: # bad points in red
                    REYE_frame = cv2.circle(REYE_frame, pt_cent, 3, (0,0,255), -1)
                elif REye_pts_now.isel(point_loc=k+2).values >= config['lik_thresh']: # good points in green
                    REYE_frame = cv2.circle(REYE_frame, pt_cent, 3, (0,255,0), -1)
        except (ValueError, KeyError):
            pass

        # then do the same on the left eye... should probably do this in a more compact way
        try:
            # get out ellipse parameters and plot them on the video
            ellipse_axes = (int(LEye_now.sel(ellipse_params='longaxis').values), int(LEye_now.sel(ellipse_params='shortaxis').values))
            ellipse_phi = int(np.rad2deg(LEye_now.sel(ellipse_params='ellipse_phi').values))
            ellipse_cent = (int(LEye_now.sel(ellipse_params='X0').values), int(LEye_now.sel(ellipse_params='Y0').values))
            LEYE_frame = cv2.ellipse(LEYE_frame, ellipse_cent, ellipse_axes, ellipse_phi, 0, 360, (255,0,0), 2) # ellipse in blue
        except (ValueError, KeyError):
            pass
        try:
            for k in range(0, len(LEye_pts_now), 3):
                pt_cent = (int(LEye_pts_now.isel(point_loc=k).values), int(LEye_pts_now.isel(point_loc=k+1).values))
                if LEye_pts_now.isel(point_loc=k+2).values < config['lik_thresh']: # bad points in red
                    LEYE_frame = cv2.circle(LEYE_frame, pt_cent, 3, (0,0,255), -1)
                elif LEye_pts_now.isel(point_loc=k+2).values >= config['lik_thresh']: # good points in green
                    LEYE_frame = cv2.circle(LEYE_frame, pt_cent, 3, (0,255,0), -1)
        except (ValueError, KeyError):
            pass
        
        # now plot pts on the topdown cam
        try:
            for k in range(0, len(TOP_pts_now['point_loc']), 3):
                try:
                    td_pts_x = TOP_pts_now.isel(point_loc=k).values
                    td_pts_y = TOP_pts_now.isel(point_loc=k + 1).values
                    center_xy = (int(td_pts_x), int(td_pts_y))
                    TOP_frame = cv2.circle(TOP_frame, center_xy, 6, (255,0,0), -1) # topdown pts in blue

                    # backX = TOP_pts_now.sel(point_loc='BackNeck_x').values
                    # backY = TOP_pts_now.sel(point_loc='BackNeck_y').values

                    # head_x1 = (backX).astype(int)
                    # head_y1 = (backY).astype(int)
                    # head_x2 = (backX + 30 * np.cos(float(TOP_head_th_now))).astype(int)
                    # head_y2 = (backY + 30 * np.sin(float(TOP_head_th_now))).astype(int)
                    # TOP_frame = cv2.line(TOP_frame, (head_x1,head_y1), (head_x2,head_y2), (0,0,0), thickness=4) # line for head angle in black

                    backX = TOP_pts_now.sel(point_loc='MidSpine2_x').values
                    backY = TOP_pts_now.sel(point_loc='MidSpine2_y').values

                    body_x1 = (backX).astype(int)
                    body_y1 = (backY).astype(int)
                    body_x2 = (backX + 200 * np.cos(float(TOP_body_th_now))).astype(int)
                    body_y2 = (backY + 200 * np.sin(float(TOP_body_th_now))).astype(int)
                    TOP_frame = cv2.line(TOP_frame, (body_x1,body_y1), (body_x2,body_y2), (0,0,0), thickness=4) # line for body angle in black also
                except (ValueError, OverflowError) as e:
                    pass
        except KeyError:
            pass

        # lastly, plot head pitch, gaze, etc. on side video

        # zero-center head theta, and get rid of wrap-around effect (mod 360)
        # add pi/8 since this is roughly head tilt in movies relative to mean theta
        pitch = np.rad2deg(head_pitch)
        pitch = ((pitch+360) % 360)
        # pitch = (180 + pitch)

        # eye divergence (theta)
        div = (RTheta - LTheta) * 0.5
        # gaze (mean theta of eyes)
        gaze_th = (RTheta + LTheta) * 0.5
        # gaze (mean phi of eyes)
        gaze_phi = (RPhi + LPhi) * 0.5

        # plot mouse head position with small blue 'tracers'
        for i in range(0,20):
            frame_before = frame_num - i
            if frame_before >= 0:
                head_x = Side_pts.sel(point_loc='LEye_x', frame=frame_before).values
                head_y = Side_pts.sel(point_loc='LEye_y', frame=frame_before).values
                try:
                    SIDE_frame = cv2.circle(SIDE_frame, (int(head_x),int(head_y)), 2, (255,0,0), -1)
                except ValueError:
                    pass

        # blue circle over the current position of the eye
        eyecent_x = SIDE_pts_now.sel(point_loc='LEye_x').values
        eyecent_y = SIDE_pts_now.sel(point_loc='LEye_y').values
        try:
            SIDE_frame = cv2.circle(SIDE_frame, (int(eyecent_x),int(eyecent_y)), 4, (255,0,0), -1)
        except ValueError:
            pass

        # calculate and plot head vector
        headV_x1 = SIDE_pts_now.sel(point_loc='LEye_x').values
        headV_y1 = SIDE_pts_now.sel(point_loc='LEye_y').values
        headV_x2 = SIDE_pts_now.sel(point_loc='LEye_x').values + 200 * np.cos(np.deg2rad(pitch))
        headV_y2 = SIDE_pts_now.sel(point_loc='LEye_y').values + 200 * np.sin(np.deg2rad(pitch))
        # black line of the head vector
        try:
            SIDE_frame = cv2.line(SIDE_frame, (int(headV_x1),int(headV_y1)), (int(headV_x2),int(headV_y2)), (0,0,0), thickness=2)
        except ValueError:
            pass

        # calculate gaze direction (head and eyes)
        gaze_direc = np.deg2rad(pitch) - div
        # rth = (th - div) + np.pi/8
        gazeV_x1 = SIDE_pts_now.sel(point_loc='LEye_x').values
        gazeV_y1 = SIDE_pts_now.sel(point_loc='LEye_y').values
        gazeV_x2 = SIDE_pts_now.sel(point_loc='LEye_x').values + 200 * np.cos(gaze_direc)
        gazeV_y2 = SIDE_pts_now.sel(point_loc='LEye_y').values + 200 * np.sin(gaze_direc)
        # cyan line of gaze direction
        try:
            SIDE_frame = cv2.line(SIDE_frame, (int(gazeV_x1),int(gazeV_y1)), (int(gazeV_x2),int(gazeV_y2)), (255,255,0), thickness=2)
        except ValueError:
            pass

        TOP_resize = cv2.resize(TOP_frame, (set_width*2, set_height))
        SIDE_resize = cv2.resize(SIDE_frame, (set_width*2, set_height))
        LEYE_resize = cv2.resize(LEYE_frame, (set_width, set_height))
        ani_resized = cv2.resize(ani_frame, (set_width*2, set_height*3))
        # also resize the LEYE in case it's off from REYE
        row_of_eyes = np.concatenate((REYE_frame, LEYE_resize), axis=1)
        eyes_and_side = np.concatenate((row_of_eyes, SIDE_resize), axis=0)
        all_vids = np.concatenate((eyes_and_side, TOP_resize), axis=0)

        animated_all = np.concatenate((all_vids, ani_resized),axis=1)

        vid_out.write(animated_all)

    vid_out.release()
