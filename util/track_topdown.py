"""
FreelyMovingEphys topdown tracking utilities
track_topdown.py

Last modified July 08, 2020
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

# matrix rotation, used to find head angle
def rotmat(theta):
    m = [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    return m

# find angle of head at all time points
# def head_angle():



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