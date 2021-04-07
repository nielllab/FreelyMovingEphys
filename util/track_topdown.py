"""
track_topdown.py

topdown tracking utilities
"""
import pandas as pd
import numpy as np
import xarray as xr
import os
import tkinter
import math
import cv2
from skimage import measure
from itertools import product
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm

from util.time import open_time
from util.format_data import split_xyl

def body_angle(pt_input, config, trial_name, top_view):
    """
    get body angle of mouse in topdown view
    """
    print('getting body angle...')
    angs = []
    for step in tqdm(range(0,np.size(pt_input, 1))):
        step_pts = pt_input.isel(frame=step)
        try:
            x1 = step_pts.sel(point_loc='spine_x')
            x2 = step_pts.sel(point_loc='spine2_x')
            y1 = step_pts.sel(point_loc='spine_y')
            y2 = step_pts.sel(point_loc='spine2_y')
        except KeyError:
            x1 = step_pts.sel(point_loc='MidSpine1_x')
            x2 = step_pts.sel(point_loc='MidSpine2_x')
            y1 = step_pts.sel(point_loc='MidSpine1_y')
            y2 = step_pts.sel(point_loc='MidSpine2_y')
        x_dist = x1 - x2
        y_dist = y1 - y2
        th = np.arctan2(y_dist, x_dist)
        angs.append(float(th))
    body_ang = xr.DataArray(angs, dims=['frame'])

    return body_ang

def head_angle1(pt_input, config, trial_name, top_view):
    """
    get head angle of mouse in topdown view
    """
    angs = []
    print('getting head angle...')
    for step in tqdm(range(0,np.size(pt_input, 1))):
        step_pts = pt_input.isel(frame=step)
        try:
            x1 = step_pts.sel(point_loc='rightear_x')
            x2 = step_pts.sel(point_loc='leftear_x')
            y1 = step_pts.sel(point_loc='rightear_y')
            y2 = step_pts.sel(point_loc='leftear_y')
        except KeyError:
            x1 = step_pts.sel(point_loc='Nose_x')
            x2 = step_pts.sel(point_loc='BackNeck_x')
            y1 = step_pts.sel(point_loc='Nose_y')
            y2 = step_pts.sel(point_loc='BackNeck_y')
        x_dist = x1 - x2
        y_dist = y1 - y2
        th = np.arctan2(y_dist, x_dist)
        angs.append(float(th))
    head_theta = xr.DataArray(angs, dims=['frame'])

    return head_theta

def body_props(top_pts, mouse_theta, config, trial_name, top_view):
    """
    using the topdown data, get properties of mouse movement (and cricket, if config file says there is one)
    """
    # set names of points
    cricketbodyX = 'cricket_body_x'; cricketbodyY = 'cricket_body_y'
    mousenoseX = 'spine_x'; mousenoseY = 'spine_y'

    filt = np.ones([3]) / np.sum(np.ones([3]))

    if config['has_cricket_labeled'] is True:
        # cricket speed
        vx_c = np.diff(top_pts.sel(point_loc=cricketbodyX).values)
        vy_c = np.diff(top_pts.sel(point_loc=cricketbodyY).values)
        vx_c = np.convolve(vx_c, filt, mode='same')
        vy_c = np.convolve(vx_c, filt, mode='same')
        cricket_speed = np.sqrt(vx_c**2, vy_c**2)

        # cricket range
        rx = top_pts.sel(point_loc=cricketbodyX).values - top_pts.sel(point_loc=mousenoseX).values
        ry = top_pts.sel(point_loc=cricketbodyY).values - top_pts.sel(point_loc=mousenoseY).values
        c_range = np.sqrt(rx**2, ry**2)

        # azimuth
        cricket_theta = np.arctan2(ry,rx)
        az = mouse_theta - cricket_theta

    # head angular velocity
    d_theta = np.diff(mouse_theta.values)
    d_theta = np.where(d_theta > np.pi, d_theta+2*np.pi, d_theta-2*np.pi)
    theta_fract = np.sum(~pd.isnull(mouse_theta.values))/len(mouse_theta.values)
    # long_theta_fract = np.sum(~pd.isnull(mouse_theta['mean_head_theta'].values))/len(mouse_theta['mean_head_theta'].values)

    # head velocity
    vx_m = np.diff(top_pts.sel(point_loc=mousenoseX).values) # currently use nose x/y -- is this different if we use the center of the head?
    vy_m = np.diff(top_pts.sel(point_loc=mousenoseY).values)
    vx_m = np.convolve(vx_m, filt, mode='same')
    vy_m = np.convolve(vy_m, filt, mode='same')
    mouse_speed = np.sqrt(vx_m**2, vy_m**2)

    if config['has_cricket_labeled'] is True:
        # plot of the cricket and mouse properties
        plt.subplots(2,3)
        plt.subplot(231)
        plt.plot(cricket_speed)
        plt.xlabel('frames')
        plt.ylabel('pixels/sec')
        plt.title('cricket speed')
        plt.subplot(232)
        plt.plot(c_range)
        plt.xlabel('frame')
        plt.ylabel('pixels')
        plt.title('range (cricket body to mouse nose)')
        plt.subplot(233)
        plt.plot(az)
        plt.xlabel('frame')
        plt.ylabel('radians')
        plt.title('azimuth')
        plt.subplot(234)
        plt.plot(d_theta)
        plt.xlabel('frame')
        plt.ylabel('radians/frame')
        plt.title('back of head angular velocity')
        plt.subplot(235)
        plt.plot(mouse_speed)
        plt.xlabel('frame')
        plt.ylabel('pixels/sec')
        plt.title('mouse speed')
        plt.savefig(os.path.join(config['trial_path'], (trial_name + '_' + top_view + '_props.png')), dpi=300)
        plt.close()

        props_out = pd.DataFrame({'cricket_speed':list(cricket_speed), 'range':list(c_range)[:-1], 'azimuth':list(az)[:-1], 'd_theta':list(d_theta), 'mouse_speed':list(mouse_speed)})
        prop_names = ['cricket_speed', 'range', 'azimuth', 'd_theta', 'mouse_speed']
        props_out_xr = xr.DataArray(props_out, coords=[('frame',range(0,np.size(cricket_speed,0))), ('prop',prop_names)])

    elif config['has_cricket_labeled'] is False:
        props_out = pd.DataFrame({'d_theta':list(d_theta), 'mouse_speed':list(mouse_speed)})
        prop_names = ['d_theta', 'mouse_speed']
        props_out_xr = xr.DataArray(props_out, coords=[('frame',range(0,np.size(mouse_speed,0))), ('prop',prop_names)])

        plt.subplots(1,2)
        plt.subplot(121)
        plt.plot(d_theta)
        plt.xlabel('frame')
        plt.ylabel('radians/frame')
        plt.title('back of head angular velocity')
        plt.subplot(122)
        plt.plot(mouse_speed)
        plt.xlabel('frame')
        plt.ylabel('pixels/sec')
        plt.title('mouse speed')
        plt.savefig(os.path.join(config['trial_path'], (trial_name + '_' + top_view + '_props.png')), dpi=300)
        plt.close()

    return props_out_xr

def topdown_tracking(topdown_data, config, trial_name, top_view):
    topdown_pt_names = list(topdown_data['point_loc'].values)
    topdown_interp = xr.DataArray.interpolate_na(topdown_data, dim='frame', use_coordinate='frame', method='linear')

    try:
        # for ephys top down network: FreelyMovingTOP_wGear-dylan-2020-10-08
        nose_x_pts = topdown_interp.sel(point_loc='nose_x')
        nose_y_pts = topdown_interp.sel(point_loc='nose_y')
    except KeyError:
        # for jumping topdown network: Jumping2-Elliott-2020-06-30
        nose_x_pts = topdown_interp.sel(point_loc='Nose_x')
        nose_y_pts = topdown_interp.sel(point_loc='Nose_y')
        
    # if config['save_figs'] is True:
    #     plt.figure()
    #     plt.title('mouse nose x/y path before likelihood threshold')
    #     plt.plot(np.squeeze(nose_x_pts), np.squeeze(nose_y_pts))
    #     plt.plot((np.squeeze(nose_x_pts)[0]), (np.squeeze(nose_y_pts)[0]), 'go') # starting point
    #     plt.plot((np.squeeze(nose_x_pts)[-1]), (np.squeeze(nose_y_pts)[-1]), 'ro')  # ending point
    #     plt.savefig(os.path.join(config['trial_path'], (trial_name + '_' + top_view + '_nose_trace.png')), dpi=300)
    #     plt.close()

    # threshold points using the input paramater (thresh) to find all times when all points are good (only want high values)
    likeli_loop_count = 0
    for pt_num in range(0, len(topdown_pt_names)):
        current_pt_loc = topdown_pt_names[pt_num]
        if 'likelihood' in current_pt_loc:
            # find the associated x and y points of the selected likelihood
            # assumes order is x, y, likelihood, will cause problems if isn't true of data...
            assoc_x_pos = topdown_pt_names[pt_num - 2]
            assoc_x_pt = topdown_interp.sel(point_loc=assoc_x_pos)
            assoc_y_pos = topdown_pt_names[pt_num - 1]
            assoc_y_pt = topdown_interp.sel(point_loc=assoc_y_pos)

            # select only the likelihood data for this point
            likeli_pt = topdown_interp.sel(point_loc=current_pt_loc)

            # set x/y coords to NaN where the likelihood is below threshold value
            assoc_x_pt[likeli_pt < config['lik_thresh']] = np.nan
            assoc_y_pt[likeli_pt < config['lik_thresh']] = np.nan

            likeli_thresh_1loc = xr.concat([assoc_x_pt, assoc_y_pt, likeli_pt], dim='point_loc')

            if likeli_loop_count == 0:
                likeli_thresh_allpts = likeli_thresh_1loc
            elif likeli_loop_count > 0:
                likeli_thresh_allpts = xr.concat([likeli_thresh_allpts, likeli_thresh_1loc], dim='point_loc', fill_value=np.nan)

            likeli_loop_count = likeli_loop_count + 1

    # make a plot of the mouse's path, where positions that fall under threshold will be NaNs
    # nose_x_thresh_pts = likeli_thresh_allpts.sel(point_loc='nose_x')
    # nose_y_thresh_pts = likeli_thresh_allpts.sel(point_loc='nose_y')
    # mask the NaNs, but only for the figure (don't want to lose time information for actual analysis)
    # nose_x_thresh_nonan_pts = nose_x_thresh_pts[np.isfinite(nose_x_thresh_pts)]
    # nose_y_thresh_nonan_pts = nose_y_thresh_pts[np.isfinite(nose_y_thresh_pts)]

    # if config['save_vids'] is True:
    #     plt.figure()
    #     plt.title('mouse nose x/y path after likelihood threshold')
    #     plt.plot(np.squeeze(nose_x_thresh_nonan_pts), np.squeeze(nose_y_thresh_nonan_pts))
    #     plt.plot((np.squeeze(nose_x_thresh_nonan_pts)[0]), (np.squeeze(nose_y_thresh_nonan_pts)[0]), 'go')  # starting point
    #     plt.plot((np.squeeze(nose_x_thresh_nonan_pts)[-1]), (np.squeeze(nose_y_thresh_nonan_pts)[-1]), 'ro')  # ending point
    #     plt.savefig(os.path.join(config['save_path'], (trial_name + '_' + top_view + '_nose_trace_thresh.png')), dpi=300)
    #     plt.close()

    likeli_thresh_allpts['trial'] = trial_name + '_' + top_view

    # points_out = likeli_thresh_allpts.assign_coords(timestamps=('frame', toptime))

    return likeli_thresh_allpts#, nose_x_thresh_pts, nose_y_thresh_pts

def plot_top_vid(vid_path, dlc_data, head_ang, config, trial_name, top_view):
    """
    plot points on topdown video and save as .avi
    """

    # read topdown video in
    vidread = cv2.VideoCapture(vid_path)
    width = int(vidread.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vidread.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # setup the file to save out of this
    savepath = os.path.join(config['trial_path'], (trial_name + '_' + top_view + '_plot.avi'))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out_vid = cv2.VideoWriter(savepath, fourcc, 60.0, (width, height))

    # set colors
    plot_color0 = (225, 255, 0)
    plot_color1 = (0, 255, 255)

    if config['num_save_frames'] > int(vidread.get(cv2.CAP_PROP_FRAME_COUNT)):
        num_save_frames = int(vidread.get(cv2.CAP_PROP_FRAME_COUNT))
    else:
        num_save_frames = config['num_save_frames']

    for frame_num in tqdm(range(0,num_save_frames)):
        # read the frame for this pass through while loop
        ret, frame = vidread.read()

        if not ret:
            break

        if dlc_data is not None:

            try:
                for k in range(0, len(dlc_data['point_loc']), 3):
                    topdownTS = dlc_data.isel(frame=frame_num)
                    if config['run_top_angles'] is True:
                        current_ang = head_ang.isel(frame=frame_num)
                    try:
                        td_pts_x = topdownTS.isel(point_loc=k).values
                        td_pts_y = topdownTS.isel(point_loc=k + 1).values
                        center_xy = (int(td_pts_x), int(td_pts_y))
                        frame = cv2.circle(frame, center_xy, 6, plot_color0, -1)

                        if config['run_top_angles'] is True:
                            backX = topdownTS.sel(point_loc='base_implant_x').values
                            backY = topdownTS.sel(point_loc='base_implant_y').values

                            x1 = (backX * np.cos(float(current_ang))).astype(int)
                            y1 = (backY * np.sin(float(current_ang))).astype(int)
                            x2 = (backX + 30 * np.cos(float(current_ang))).astype(int)
                            y2 = (backY + 30 * np.sin(float(current_ang))).astype(int)
                            frame = cv2.line(frame, (x1,y1), (x2,y2), plot_color1, thickness=4)
                    except (ValueError, OverflowError) as e:
                        pass
            except KeyError:
                pass

            out_vid.write(frame)

        elif dlc_data is None:
            out_vid.write(frame)

    out_vid.release()
