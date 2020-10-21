"""
track_topdown.py

topdown tracking utilities

Oct. 21, 2020
"""

# package imports
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

# module imports
from util.read_data import split_xyl, open_time

# matrix rotation, used to find head angle
def rotmat(theta):
    m = [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    return m

def body_angle(pt_input, config, trial_name, top_view):
    angs = []
    for step in range(0,len(pt_input)):
        step_pts = pt_input.isel(frame=step)
        x1 = step_pts.sel(point_loc='spine_x')
        x2 = step_pts.sel(point_loc='spine2_x')
        y1 = step_pts.sel(point_loc='spine_y')
        y2 = step_pts.sel(point_loc='spine2_y')
        ang = -1/((y2-y1)/(x2-x1))
        angs.append(ang)
    head_theta = xr.DataArray(angs)

    return body_ang

def head_angle1(pt_input, config, trial_name, top_view):
    angs = []
    for step in range(0,len(pt_input)):
        step_pts = pt_input.isel(frame=step)
        x1 = step_pts.sel(point_loc='rightear_x')
        x2 = step_pts.sel(point_loc='leftear_x')
        y1 = step_pts.sel(point_loc='rightear_y')
        y2 = step_pts.sel(point_loc='leftear_y')
        ang = -1/((y2-y1)/(x2-x1))
        angs.append(ang)
    head_theta = xr.DataArray(angs)

    return head_theta

# using the topdown data, get properties of mouse movement (and cricket, if config file says there is one)
def body_props(top_pts, mouse_theta, config, trial_name, top_view):

    # set names of points
    cricketbodyX = 'cricket_body_x'; cricketbodyY = 'cricket_body_y'
    mousenoseX = 'spine_x'; mousenoseY = 'spine_y'

    if config['cricket'] is True:
        # cricket speed
        vx_c = np.diff(top_pts.sel(point_loc=cricketbodyX).values)
        vy_c = np.diff(top_pts.sel(point_loc=cricketbodyY).values)
        filt = np.ones([3]) / np.sum(np.ones([3]))
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

    if config['cricket'] is True:
        # a very large plot of the cricket and mouse properties
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

    elif config['cricket'] is False:
        props_out = pd.DataFrame({'d_theta':list(d_theta), 'mouse_speed':list(mouse_speed)})
        prop_names = ['d_theta', 'mouse_speed']
        props_out_xr = xr.DataArray(props_out, coords=[('frame',range(0,np.size(cricket_speed,0))), ('prop',prop_names)])

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

# find angle of head at all time points
def head_angle(pt_input, nose_x, nose_y, config, trial_name, top_view):

    pt_names = list(pt_input['point_loc'].values)

    if config['cricket'] is True:
        pt_input = pt_input[:-3,:]
        pt_names = pt_names[:-3]

    x_vals, y_vals, likeli_pts = split_xyl(pt_names, pt_input, lik_thresh)

    data = np.stack([x_vals.T, y_vals.T])

    centroid = np.squeeze(np.mean(data, axis=1))

    centered = np.zeros(np.shape(data), dtype=object)
    theta_good = np.zeros(np.shape(centroid[0,:]), dtype=object)
    aligned_good = np.zeros(np.shape(data), dtype=object)

    # get centered points
    for h in range(0,len(data[0,:,0])):
        centered[:,h,:] = data[:,h,:] - centroid

    # last good frame will be used as reference frame
    num_good_for_frames = []
    for testframe in range(0, np.size(data, axis=2)):
        num_ideal_points_in_test = np.size(data[0, :, testframe], axis=0)
        num_good = np.count_nonzero(~np.isnan(data[1, :, testframe]))
        if num_ideal_points_in_test == num_good:
            ref = centered[:,:,testframe]
        num_good_for_frames.append(num_good)

    # if there are no NaNs and it's a perfect timepoint, loop through a range of thetas and rotate the frame by that much
    # then, calculate how well it matches the reference
    for frame in range(0, np.size(centroid, axis=1)):
        num_ideal_points = np.size(data[0, :, frame], axis=0)
        num_real_pts = np.count_nonzero(~np.isnan(data[1, :, frame]))
        if num_real_pts == num_ideal_points:
            c = centered[:,:,frame]
            theta = np.linspace(0, (2 * math.pi), 101)
            theta = theta[1:-1]
            rms = np.zeros(len(theta))
            for i in range(0, len(theta)):
                c_rot = np.matmul(c.T, rotmat(theta[i])) # rotation
                rms[i] = np.nansum((ref - c_rot.T) ** 2)  # root mean squared
            # find index for theta with smallest error and rotate by this amount
            theta_good[frame] = np.argmin(rms)
            aligned_good[:,:,frame] = np.matmul(c.T, rotmat(theta_good[frame])).T
        elif num_real_pts < num_ideal_points:
            theta_good[frame] = np.nan
            aligned_good[:,:,frame] = np.nan

    # if there are no values in aligned_good, then don't bother with the rest of the process
    try:
        # calculate mean head from good points across trials
        mean_head = np.nanmean(aligned_good, axis=2)
        # rotate mean head to align to x-axis
        longaxis = mean_head[:, [0, num_ideal_points-1]] # line from middle of head to nose
        longtheta = np.arctan2(np.diff(longaxis[:,1]).astype(float), np.diff(longaxis[:,0]).astype(float))[0] # angle of line at mean head angle
        headrot = rotmat(-longtheta)
        aligned = np.zeros(np.shape(aligned_good.T), dtype=object)
        for frame in range(0,np.size(aligned_good, axis=2)):
            aligned[frame,:,:] = np.matmul(aligned_good[:,:,frame].T, headrot)
        aligned = aligned.T

        mean_head1 = np.nanmean(aligned, axis=2)

        mean_stack = np.stack(([mean_head1[0,:]**2, mean_head1[1,:]**2]), axis=1)
        mean_dist = mean_stack[:,0] + mean_stack[:,1]
        for i in range(0,len(mean_dist)):
            mean_dist[i] = np.sqrt(mean_dist[i])
        cent = np.zeros([2, np.size(aligned_good.T, axis=0)], dtype=object)

        # get all centroids
        for frame in range(0, np.size(centroid, axis=1)):
            c = data[:,:,frame]
            meshx, meshy = np.meshgrid((np.floor(np.amin(c[0,:])), np.ceil(np.amax(c[0,:]))), (np.floor(np.amin(c[1,:])), np.ceil(np.amax(c[1,:]))), sparse=False)

            # for each head point calculate how far the pixels are from it, calculate error of how
            # far this is from where it should be, and add these up
            err = 0
            for i in range(0,num_ideal_points):
                if ~np.isnan(c[0,i]):
                    r = np.sqrt((meshx-c[0,i])**2 + (meshy-c[1,i])**2) # distance
                    err = err + (mean_dist[i] - r)**2 # error
            # find minimum, then get x and y values and set as centeroid
            ind = np.argmin(err)
            unravel_ind = np.unravel_index(ind,np.shape(err), order='C')
            cent[0,frame] = meshx[unravel_ind]
            cent[1,frame] = meshy[unravel_ind]

        # center all points using calculated centroid
        for i in range(0,num_ideal_points):
            centered[:,i,:] = data[:,i,:] - cent

        # now, align all timepoints
        allaligned = np.zeros(np.shape(centered), dtype=object)
        alltheta = np.zeros(np.shape(centroid[0,:]), dtype=object)

        for frame in range(0, np.size(centroid, axis=1)):
            num_ideal_points = np.size(data[0, :, frame], axis=0)
            num_real_pts = np.count_nonzero(~np.isnan(data[1, :, frame]))
            c = centered[:,:,frame]
            if num_real_pts >= 3:
                theta = np.linspace(0, (2 * np.pi), 101)
                theta = theta[1:-1]
                del rms
                rms = np.zeros(len(theta))
                for i in range(0, len(theta)):
                    c_rot = np.matmul(c.T, rotmat(theta[i])) # rotation
                    rms[i] = np.nansum((mean_head1 - c_rot.T) ** 2)  # root mean squared
                # find index for theta with smallest error and rotate by this amount
                alltheta[frame] = np.argmin(rms)
                allaligned[:,:,frame] = np.matmul(c.T, rotmat(alltheta[frame])).T
            elif num_real_pts < 3:
                alltheta[frame] = np.nan
                allaligned[:,:,frame] = np.nan

        # head angle was negative of what we want, this fixes that
        alltheta = 2 * np.pi - alltheta
        # range -pi to pi
        alltheta = np.where(alltheta > np.pi, alltheta, alltheta-2*np.pi)

        # build the xarray to store out theta values
        thetaout = xr.DataArray(alltheta, coords={'frame':range(0,len(alltheta))}, dims=['frame'])
        thetaout['mean_head_theta'] = longtheta

    except ZeroDivisionError:
        thetaout = xr.DataArray(np.zeros(np.shape(theta_good)))

    if config['save_vids'] is True:
        fig1 = plt.figure(constrained_layout=True)
        gs = fig1.add_gridspec(5,2)
        f1_ax1 = fig1.add_subplot(gs[0, :])
        f1_ax1.set_title(trial_name + top_view + 'points')
        f1_ax1.plot(data[:,0,:], data[:,1,:])
        f1_ax2 = fig1.add_subplot(gs[1, 0])
        f1_ax2.set_title('number of good points')
        f1_ax2.plot(num_good_for_frames)
        f1_ax3 = fig1.add_subplot(gs[1, 1])
        f1_ax3.set_title('fraction bad timepoints')
        f1_ax3.plot(num_good_for_frames)
        f1_ax3.set_xlabel('point num'); f1_ax3.set_ylim([0,1])
        # f1_ax4 = fig1.add_subplot(gs[2, 0])
        # f1_ax4.scatter(cent[0,:], cent[1,:])
        # f1_ax4.set_title('all points')
        # f1_ax5 = fig1.add_subplot(gs[2, 1])
        # f1_ax5.set_title('only good points')
        # f1_ax5.scatter(centroid[0,:], centroid[1,:])
        f1_ax6 = fig1.add_subplot(gs[3, :])
        f1_ax6.plot(alltheta)
        f1_ax6.set_title('final theta')
        f1_ax6.set_ylabel('theta'); f1_ax6.set_xlabel('frame')
        f1_ax7 = fig1.add_subplot(gs[4, :])
        f1_ax7.plot(nose_x); f1_ax7.plot(nose_y)
        f1_ax7.legend('x','y')
        f1_ax7.set_ylabel('position'); f1_ax6.set_xlabel('frame')
        plt.savefig(os.path.join(config['trial_path'], (trial_name + '_' + top_view + '_head_alignment.png')), dpi=300)
        plt.close()

    return thetaout

# track topdown position by calling other functions, takes in ONE trial at a time
def topdown_tracking(topdown_data, config, trial_name, top_view):
    topdown_pt_names = list(topdown_data['point_loc'].values)
    topdown_interp = xr.DataArray.interpolate_na(topdown_data, dim='frame', use_coordinate='frame', method='linear')

    nose_x_pts = topdown_interp.sel(point_loc='nose_x')
    nose_y_pts = topdown_interp.sel(point_loc='nose_y')
    if config['save_figs'] is True:
        plt.figure()
        plt.title('mouse nose x/y path before likelihood threshold')
        plt.plot(np.squeeze(nose_x_pts), np.squeeze(nose_y_pts))
        plt.plot((np.squeeze(nose_x_pts)[0]), (np.squeeze(nose_y_pts)[0]), 'go') # starting point
        plt.plot((np.squeeze(nose_x_pts)[-1]), (np.squeeze(nose_y_pts)[-1]), 'ro')  # ending point
        plt.savefig(os.path.join(config['trial_path'], (trial_name + '_' + top_view + '_nose_trace.png')), dpi=300)
        plt.close()

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

# plot points on topdown video and save as .avi
def plot_top_vid(vid_path, dlc_data, head_ang, config, trial_name, top_view):

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

    for frame_num in tqdm(range(0,int(vidread.get(cv2.CAP_PROP_FRAME_COUNT)))):
        # read the frame for this pass through while loop
        ret, frame = vidread.read()

        if not ret:
            break

        if dlc_data is not None:
            # get current frame number to be displayed, so that it can be used to slice DLC data
            frame_time = vidread.get(cv2.CAP_PROP_POS_FRAMES)

            try:
                for k in range(0, len(dlc_data['point_loc']), 3):
                    topdownTS = dlc_data.sel(frame=frame_time)
                    current_ang = head_ang.sel(frame=frame_time)
                    try:
                        td_pts_x = topdownTS.isel(point_loc=k).values
                        td_pts_y = topdownTS.isel(point_loc=k + 1).values
                        center_xy = (int(td_pts_x), int(td_pts_y))
                        if k == 0:
                            # plot them on the fresh topdown frame
                            frame = cv2.circle(frame, center_xy, 6, plot_color0, -1)
                        elif k >= 3:
                            # plot them on the topdown frame with all past topdown points
                            frame = cv2.circle(frame, center_xy, 6, plot_color0, -1)

                        backX = topdownTS.sel(point_loc='baseimplant_x').values
                        backY = topdownTS.sel(point_loc='baseimplant_y').values

                        x1 = (backX * np.cos(float(current_ang))).astype(int)
                        y1 = (backY * np.sin(float(current_ang))).astype(int)
                        x2 = (backX + 30 * np.cos(float(current_ang))).astype(int)
                        y2 = (backY + 30 * np.sin(float(current_ang))).astype(int)
                        frame = cv2.line(frame, (x1,y1), (x2,y2), plot_color1, thickness=4)
                    except ValueError:
                        pass
            except KeyError:
                pass

            out_vid.write(frame)

        elif dlc_data is None:
            out_vid.write(frame)

    out_vid.release()
