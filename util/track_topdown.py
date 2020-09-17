"""
FreelyMovingEphys topdown tracking utilities
track_topdown.py

Last modified July 15, 2020
"""

# package imports
import pandas as pd
import numpy as np
import matplotlib
import xarray as xr
import os
from matplotlib import pyplot as plt
import tkinter
import math

# function imports
from util.read_data import split_xyl

# matrix rotation, used to find head angle
def rotmat(theta):
    m = [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    return m

# find angle of head at all time points
def head_angle(pt_input, pt_names, lik_thresh, savepath, cricket, trial_name):
    fig_dir = savepath + '/' + trial_name + '/'
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    if cricket is True:
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
    for testframe in range(0, np.size(data, axis=2)):
        testptnum = np.size(data[:, :, testframe], axis=1)
        num_good = np.count_nonzero(~np.isnan(data[1, :, testframe]))
        if testptnum == num_good:
            ref = centered[:,:,testframe]

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

    # calculate mean head from good points across trials
    mean_head = np.nanmean(aligned_good, axis=2)

    # rotate mean head to align to x-axis
    longaxis = mean_head[:, [num_ideal_points-1, 1]] # line from middle of head to nose
    longtheta = np.arctan2(np.diff(longaxis[1]).astype(float), np.diff(longaxis[0]).astype(float))[0] # angle of line
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

    # get all cetroids
    for frame in range(0, np.size(centroid, axis=1)):
        c = data[:,:,frame]
        mesh1 = np.floor(np.amin(c[0,:]))
        mesh2 = np.ceil(np.amax(c[0,:]))
        mesh3 = np.floor(np.amin(c[1,:]))
        mesh4 = np.ceil(np.amax(c[1,:]))
        meshx, meshy = np.meshgrid((mesh1, mesh2), (mesh3, mesh4), sparse=False)

        # for each head point calculate how far the pixels are from it, calculate error of how
        # far this is from where it should be, and add these up
        err = 0
        for i in range(0,num_ideal_points):
            if ~np.isnan(c[0,i]):
                r = np.sqrt((meshx-c[0,i])**2 + (meshy-c[1,i])**2) # distance
                err = err + (mean_dist[i] - r)**2 # error
        # find minimum, then get x and y values and set as centeroid
        ind = np.argmin(err)
        indi, indj = np.unravel_index(ind,np.shape(err))
        cent[0,frame] = meshx[indi,indj]
        cent[1,frame] = meshy[indi,indj]

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
            theta = np.linspace(0, (2 * math.pi), 101)
            theta = theta[1:-1]
            del rms
            rms = np.zeros(len(theta))
            for i in range(0, len(theta)):
                c_rot = np.matmul(c.T, rotmat(theta[i])) # rotation
                rms[i] = np.nansum((mean_head - c_rot.T) ** 2)  # root mean squared
            # find index for theta with smallest error and rotate by this amount
            alltheta[frame] = np.argmin(rms)
            allaligned[:,:,frame] = np.matmul(c.T, rotmat(alltheta[frame])).T
        elif num_real_pts < 3:
            alltheta[frame] = np.nan
            allaligned[:,:,frame] = np.nan

    # head angle was negative of what we want, so this fixes that
    alltheta = 2 * math.pi - alltheta
    # range -pi to pi
    alltheta = np.where(alltheta > math.pi, alltheta, alltheta-2*math.pi)

    # plots of head theta
    plt.figure(figsize=(15,15))
    plt.plot(alltheta)
    plt.xlabel('frame')
    plt.ylabel('angle')
    plt.title('head theta over frames')
    plt.savefig(fig_dir + 'head_angle_trace.png', dpi=300)
    plt.close()

    thetaout = xr.DataArray(alltheta)

    return thetaout

# track topdown position by calling other functions, takes in ONE trial at a time
def topdown_tracking(topdown_data, topdown_pt_names, savepath, trial_name, lik_thresh, coord_cor, topdown_pt_num, cricket):
    # make directory for figure saving, if it does not already exist
    fig_dir = savepath + '/' + trial_name + '/'
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    topdown_interp = xr.DataArray.interpolate_na(topdown_data, dim='frame', use_coordinate='frame', method='linear')

    # for each point location in the topdown input data, select y head points and subtract them from int to fix coordinates
    y_names = [name for name in topdown_pt_names if '_y' in name]
    x_names = [name for name in topdown_pt_names if '_x' in name]
    l_names = [name for name in topdown_pt_names if 'lik' in name]
    y_data = topdown_interp.sel(point_loc=y_names) - coord_cor
    x_data = topdown_interp.sel(point_loc=x_names)
    l_data = topdown_interp.sel(point_loc=l_names)
    topdown_coordcor = xr.concat([x_data, y_data, l_data], dim='point_loc', fill_value=np.nan)

    nose_x_pts = topdown_coordcor.sel(point_loc='nose_x')
    nose_y_pts = topdown_coordcor.sel(point_loc='nose_y')
    plt.figure(figsize=(15, 15))
    plt.title('mouse nose x/y path before likelihood threshold')
    plt.plot(np.squeeze(nose_x_pts), np.squeeze(nose_y_pts))
    plt.plot((np.squeeze(nose_x_pts)[0]), (np.squeeze(nose_y_pts)[0]), 'go') # starting point
    plt.plot((np.squeeze(nose_x_pts)[-1]), (np.squeeze(nose_y_pts)[-1]), 'ro')  # ending point
    plt.savefig(fig_dir + 'nose_trace.png', dpi=300)
    plt.close()

    # threshold points using the input paramater (thresh) to find all times when all points are good (only want high values)
    likeli_loop_count = 0
    for pt_num in range(0, len(topdown_pt_names)):
        current_pt_loc = topdown_pt_names[pt_num]
        if 'likelihood' in current_pt_loc:
            # find the associated x and y points of the selected likelihood
            # assumes order is x, y, likelihood, will cause problems if isn't true of data...
            assoc_x_pos = topdown_pt_names[pt_num - 2]
            assoc_x_pt = topdown_coordcor.sel(point_loc=assoc_x_pos)
            assoc_y_pos = topdown_pt_names[pt_num - 1]
            assoc_y_pt = topdown_coordcor.sel(point_loc=assoc_y_pos)

            # select only the likelihood data for this point
            likeli_pt = topdown_coordcor.sel(point_loc=current_pt_loc)

            # set x/y coords to NaN where the likelihood is below threshold value
            assoc_x_pt[likeli_pt < lik_thresh] = np.nan
            assoc_y_pt[likeli_pt < lik_thresh] = np.nan

            likeli_thresh_1loc = xr.concat([assoc_x_pt, assoc_y_pt, likeli_pt], dim='point_loc')

            if likeli_loop_count == 0:
                likeli_thresh_allpts = likeli_thresh_1loc
            elif likeli_loop_count > 0:
                likeli_thresh_allpts = xr.concat([likeli_thresh_allpts, likeli_thresh_1loc], dim='point_loc',
                                                 fill_value=np.nan)

            likeli_loop_count = likeli_loop_count + 1

    # make a plot of the mouse's path, where positions that fall under threshold will be NaNs
    nose_x_thresh_pts = likeli_thresh_allpts.sel(point_loc='nose_x')
    nose_y_thresh_pts = likeli_thresh_allpts.sel(point_loc='nose_y')
    # mask the NaNs, but only for the figure (don't want to lose time information for actual analysis)
    nose_x_thresh_nonan_pts = nose_x_thresh_pts[np.isfinite(nose_x_thresh_pts)]
    nose_y_thresh_nonan_pts = nose_y_thresh_pts[np.isfinite(nose_y_thresh_pts)]
    plt.figure(figsize=(15, 15))
    plt.title('mouse nose x/y path after likelihood threshold')
    plt.plot(np.squeeze(nose_x_thresh_nonan_pts), np.squeeze(nose_y_thresh_nonan_pts))
    plt.plot((np.squeeze(nose_x_thresh_nonan_pts)[0]), (np.squeeze(nose_y_thresh_nonan_pts)[0]), 'go')  # starting point
    plt.plot((np.squeeze(nose_x_thresh_nonan_pts)[-1]), (np.squeeze(nose_y_thresh_nonan_pts)[-1]), 'ro')  # ending point
    plt.savefig(fig_dir + 'nose_trace_thresh.png', dpi=300)
    plt.close()

    likeli_thresh_allpts['trial'] = trial_name

    return likeli_thresh_allpts