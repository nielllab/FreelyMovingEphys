#####################################################################################
"""
topdown_preening.py of FreelyMovingEphys
(formerly: topdown_preening.py)

Takes in data in the form of an xarray DataArray from mouse arena top-down camera.
There ought to be DataArray coordinates for frame of video, experiment trial, and
point location on the mouse. Within each point location there should be an x, y, and
likelihood value (see example data structure below).
In the function preen_topdown_data(), y-coordinates are corrected and x/y position
values are thresholded so that the likelihood value next-door to the x/y positions
falling below the threshold value, thresh, will result in the x/y positions being set
to being NaN.
There are three helper functions in this file: (1) drop_leading_and_lagging_nans() will
eliminate the NaNs that come before and after the first and last real values in a
particular trial and front so that plots can be mde without plotting NaNs for the
start and finish points. (2) xr_looped_append() is a somewhat sloppy attempt to append
xarray DataArrays together without there being an xr.append function in existence.
There are inelegant consequences to this function's current structure, noted in
in-line comments. (3) interp_nans_if_any interpolates across NaNs in the data.
Two figures are produced for each trial (if figures=True): one tracing the mouse’s
nose x/y before the x/y positions are changed to NaNs where the likelihood falls below
threshold, and one after they are replaced with NaNs.

Example topdown input data layout:

<xarray.DataArray (trial: 2, frame: 4563, point_loc: 30)>
array([...values of data here...])
Coordinates:
  * frame      (frame) int64 0 1 2 3 4 5 6 ... 4557 4558 4559 4560 4561 4562
  * point_loc  (point_loc) object 'nose x' ... 'cricket Body likelihood'
  * trial      (trial) object 'mouse_J462c_trial_2_090519_0' 'mouse_J462c_trial_1_090519_3'

Code adapted from GitHub repository /niell-lab-analysis/freely moving/alignHead.m

TO DO:
- eliminate the xr_looped_append() function and write a replacement [low priority]
- add to description information about round_msec() function

last modified: June 23, 2020 by Dylan Martins (dmartins@uoregon.edu)
"""
#####################################################################################
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import xarray as xr
import os
from matplotlib import pyplot as plt
import tkinter
import math

from utilities.data_reading import test_trial_presence
from utilities.data_cleaning import split_xyl

####################################################
def xr_looped_append(loop_over_array, new_array, trial_or_loc_id, dim_concat_to, loop_num):
    if loop_num == 0:
        loop_over_array = xr.DataArray(new_array)
        loop_over_array[dim_concat_to] = trial_or_loc_id
    elif loop_num > 0:
        new_array = xr.DataArray(new_array)
        new_array[dim_concat_to] = trial_or_loc_id
        loop_over_array = xr.concat([loop_over_array, new_array], dim=dim_concat_to, fill_value=np.nan)
    else:
        print('xr_looped_append issue with loop number')
    return loop_over_array

####################################################
def interp_nans_if_any(whole_data, loc_names):
    '''
    Runs NaN interpolation of xarray DataArray
    :param whole_data: DataArray of positions
    :param coords_array: array of coord names of whole_data
    :return: interpolated DataArray of same coords and dims
    '''
    for loc_num in range(0, len(loc_names)):
        # get the name of the current point location, index into whole_data to get the data that will be interpolated
        # then, delete unneeded coords of data_to_interp
        this_loc_name = loc_names[loc_num]
        data_to_interp = whole_data.sel(point_loc=this_loc_name)
        del data_to_interp['point_loc'], data_to_interp['trial']

        # do the linear interpolation
        interp_item = xr.DataArray.interpolate_na(data_to_interp, dim='time', use_coordinate='time', method='linear')

        # rebuild the post-interpolation DataArray
        if loc_num == 0:
            interp_data = xr_looped_append(interp_item, interp_item, this_loc_name, 'point_loc', loc_num)
        if loc_num > 0:
            interp_data = xr_looped_append(interp_data, interp_item, this_loc_name, 'point_loc', loc_num)

    return interp_data

####################################################

def preen_topdown_data(all_topdown_data, trial_list, pt_names, savepath_input, coord_correction_val=1200, num_points=8, thresh=0.99, showfig=False, savefig=False):
    '''
    Aligns the head of a mouse from DLC output files which are passed in from load_from_DLC.py

    :param all_topdown_data: one xarray DataArray containing the coordinates of all points for all trials
    :param trial_list: list of unique strings identifying each trial in all_topdown_data
    :param pt_names: list of names of point locations on mouse
    :param coord_correction_val: value to subtract from y-axis coordinates
    :param num_points: number of points recorded from DLC
    :param thresh: likelihood threshold value
    :param figures: bool, determines whether or not figures will be printed out for each trial
    :return:

    other notes
    default num_points set to 8; 9&10 would include cricket points in Angie's prey capture data
    '''

    # run through each trial individually
    for trial_num in range(0, len(trial_list)):
        # get the name of the current trial
        current_trial = trial_list[trial_num]
        test_trial = test_trial_presence(all_topdown_data, current_trial)
        if test_trial is True:
            with all_topdown_data.sel(trial=current_trial) as topdown_data:
                # interpolate across NaNs fro each point_loc, then piece dataset back together
                topdown_interp = interp_nans_if_any(topdown_data, pt_names)
                # make a copy of topdown_interp so that corrected y values can be concated into it

                # for each point location in the topdown input data, select y head points and subtract them from int to fix coordinates
                for pt_num in range(0, len(pt_names)):
                    pt_str = pt_names[pt_num]
                    if ' x' in pt_str:
                        orig_x_topdown = topdown_interp.sel(point_loc=pt_str)
                        if pt_num == 0:
                            topdown_coordcor = xr_looped_append(orig_x_topdown, orig_x_topdown, pt_str, 'point_loc', pt_num)
                        if pt_num > 0:
                            topdown_coordcor = xr_looped_append(topdown_coordcor, orig_x_topdown, pt_str, 'point_loc', pt_num)

                    if ' y' in pt_str:
                        # select data from each y point and sutract from the values a coordinate correction int, default=1200
                        orig_y_topdown = topdown_interp.sel(point_loc=pt_str)
                        nonan_y_topdown = xr.DataArray.dropna(orig_y_topdown, dim='time')
                        cor_y_topdown = nonan_y_topdown - coord_correction_val
                        # next block is a bit goofy... pass cor_y_topdown twice into xr_looped_append, it's only uesd the second time
                        # this is becuse it's one funciton to both create a DataArray and concat to it
                        # see above function, xr_looped_append, which works but remains inelegant
                        if pt_num == 0:
                            topdown_coordcor = xr_looped_append(cor_y_topdown, cor_y_topdown, pt_str, 'point_loc', pt_num)
                        if pt_num > 0:
                            topdown_coordcor = xr_looped_append(topdown_coordcor, cor_y_topdown, pt_str, 'point_loc', pt_num)

                    if 'likelihood' in pt_str:
                        orig_lik_topdown = topdown_interp.sel(point_loc=pt_str)
                        if pt_num == 0:
                            topdown_coordcor = xr_looped_append(orig_lik_topdown, orig_lik_topdown, pt_str, 'point_loc', pt_num)
                        if pt_num > 0:
                            topdown_coordcor = xr_looped_append(topdown_coordcor, orig_lik_topdown, pt_str, 'point_loc', pt_num)

                # THOUGHT: eliminate NaNs at the start and end of each trial and point so that plots show true start and finish
                # topdown_interp_leadlagdrop = drop_leading_and_lagging_nans(topdown_interp, pt_names)

                # make figure of nose position over time, with start and finish labeled in green and red respectively
                if savefig is True:
                    fig1_dir = savepath_input + '/' + current_trial + '/'
                    if not os.path.exists(fig1_dir):
                        os.makedirs(fig1_dir)
                    fig1_path = fig1_dir + 'nose_position_over_time.png'

                    # for now, just drop NaNs that remain in the topdown_interp xarray after interpolation
                    # coordcor_pts_wout_nans = drop_leading_and_lagging_nans(topdown_coordcor, pt_names)
                    nose_x_pts = topdown_coordcor.sel(point_loc='nose x')
                    nose_y_pts = topdown_coordcor.sel(point_loc='nose y')
                    plt.figure(figsize=(15, 15))
                    plt.title('mouse nose x/y path before likelihood threshold')
                    plt.plot(np.squeeze(nose_x_pts), np.squeeze(nose_y_pts))
                    plt.plot((np.squeeze(nose_x_pts)[0]), (np.squeeze(nose_y_pts)[0]), 'go') # starting point
                    plt.plot((np.squeeze(nose_x_pts)[-1]), (np.squeeze(nose_y_pts)[-1]), 'ro')  # ending point
                    plt.savefig(fig1_path, dpi=300)
                    plt.close()


                # threshold points using the input paramater (thresh) to find all times when all points are good (only want high values)
                likeli_loop_count = 0
                for pt_num in range(0, len(pt_names)):
                    current_pt_loc = pt_names[pt_num]
                    if 'likelihood' in current_pt_loc:
                        # find the associated x and y points of the selected liklihood
                        # assumes order is x, y, likelihood, will cause problems if isn't true of data...
                        assoc_x_pos = pt_names[pt_num - 2]
                        assoc_x_pt = topdown_coordcor.sel(point_loc=assoc_x_pos)
                        assoc_y_pos = pt_names[pt_num - 1]
                        assoc_y_pt = topdown_coordcor.sel(point_loc=assoc_y_pos)

                        # select only the likelihood data for this point
                        likeli_pt = topdown_coordcor.sel(point_loc=current_pt_loc)

                        # get number of frames to use for indexing through all positions
                        frame_coords = topdown_coordcor.coords['time']
                        frame_len = len(frame_coords)

                        # set x/y coords to NaN where the liklihood is below threshold value
                        assoc_x_pt[likeli_pt < thresh] = np.nan
                        assoc_y_pt[likeli_pt < thresh] = np.nan

                        likeli_thresh_1loc = xr.concat([assoc_x_pt, assoc_y_pt, likeli_pt], dim='point_loc')

                        if likeli_loop_count == 0:
                            likeli_thresh_allpts = likeli_thresh_1loc
                        elif likeli_loop_count > 0:
                            likeli_thresh_allpts = xr.concat([likeli_thresh_allpts, likeli_thresh_1loc], dim='point_loc', fill_value=np.nan)

                        likeli_loop_count = likeli_loop_count + 1

                if savefig is True:
                    fig2_dir = savepath_input + '/' + current_trial + '/'
                    if not os.path.exists(fig2_dir):
                        os.makedirs(fig2_dir)
                    fig2_path = fig2_dir + 'nose_position_over_time_thresh.png'

                    # make a plot of the mouse's path, where positions that fall under threshold will be NaNs
                    nose_x_thresh_pts = likeli_thresh_allpts.sel(point_loc='nose x')
                    nose_y_thresh_pts = likeli_thresh_allpts.sel(point_loc='nose y')
                    # mask the NaNs, but only for the figure (don't want to lose time information for actual analysis)
                    nose_x_thresh_nonan_pts = nose_x_thresh_pts[np.isfinite(nose_x_thresh_pts)]
                    nose_y_thresh_nonan_pts = nose_y_thresh_pts[np.isfinite(nose_y_thresh_pts)]
                    plt.figure(figsize=(15, 15))
                    plt.title('mouse nose x/y path after likelihood threshold')
                    plt.plot(np.squeeze(nose_x_thresh_nonan_pts), np.squeeze(nose_y_thresh_nonan_pts))
                    plt.plot((np.squeeze(nose_x_thresh_nonan_pts)[0]), (np.squeeze(nose_y_thresh_nonan_pts)[0]), 'go') # starting point
                    plt.plot((np.squeeze(nose_x_thresh_nonan_pts)[-1]), (np.squeeze(nose_y_thresh_nonan_pts)[-1]), 'ro') # ending point
                    plt.savefig(fig2_path, dpi=300)
                    plt.close()

                x_vals, y_vals, likeli_pts = split_xyl(pt_names, topdown_coordcor, 0.99)
                timestamp_list = list(x_vals.index.values)

                if savefig is True:
                    frame_slice = timestamp_list[0]
                    x_to_plot = x_vals.loc[[frame_slice]]
                    y_to_plot = y_vals.loc[[frame_slice]]

                    fig3_dir = savepath_input + '/' + current_trial + '/'
                    if not os.path.exists(fig3_dir):
                        os.makedirs(fig3_dir)
                    fig3_path = fig3_dir + 'dlc_topdown_pts_at_time_' + str(frame_slice) + '.png'

                    plt.figure(figsize=(15, 10))
                    plt.plot(int(x_to_plot.iloc[0,0]), int(y_to_plot.iloc[0,0]), 'bo')
                    plt.plot(int(x_to_plot.iloc[0,1]), int(y_to_plot.iloc[0,1]), 'go')
                    plt.plot(int(x_to_plot.iloc[0,2]), int(y_to_plot.iloc[0,2]), 'ro')
                    plt.plot(int(x_to_plot.iloc[0,3]), int(y_to_plot.iloc[0,3]), 'co')
                    plt.plot(int(x_to_plot.iloc[0,4]), int(y_to_plot.iloc[0,4]), 'mo')
                    plt.plot(int(x_to_plot.iloc[0,5]), int(y_to_plot.iloc[0,5]), 'yo')
                    plt.plot(int(x_to_plot.iloc[0,6]), int(y_to_plot.iloc[0,6]), 'ko')
                    plt.title('topdown dlc points at time ' + str(frame_slice) + ' of ' + str(current_trial))
                    plt.savefig(fig3_path, dpi=300)
                    plt.close()

                back_head_centroid = []
                front_head_centroid = []
                atan_head = []
                for time in range(0, len(timestamp_list)):
                    frame_slice = timestamp_list[time]
                    x_to_plot = x_vals.loc[[frame_slice]]
                    y_to_plot = y_vals.loc[[frame_slice]]
                    sum_back_x_pts = sum([int(x_to_plot.iloc[0,6]), int(x_to_plot.iloc[0,3])])
                    sum_back_y_pts = sum([int(y_to_plot.iloc[0, 6]), int(y_to_plot.iloc[0, 3])])

                    back_head_centroid_timepoint = (sum_back_x_pts / 2, sum_back_y_pts / 2)
                    front_head_centroid_timepoint = (int(x_to_plot.iloc[0,0]), int(y_to_plot.iloc[0,0]))
                    atan_head_timepoint = math.atan((front_head_centroid_timepoint[1] - back_head_centroid_timepoint[1]) / (front_head_centroid_timepoint[0] - back_head_centroid_timepoint[0])) * 180 / math.pi

                    back_head_centroid.append(back_head_centroid_timepoint)
                    front_head_centroid.append(front_head_centroid_timepoint)
                    atan_head.append(atan_head_timepoint)

                if savefig is True:
                    fig4_dir = savepath_input + '/' + current_trial + '/'
                    if not os.path.exists(fig4_dir):
                        os.makedirs(fig4_dir)
                    fig4_path = fig4_dir + 'dlc_topdown_head_angle_timepoint.png'

                    plt.figure(figsize=(15, 10))
                    plt.plot([back_head_centroid_timepoint[0], front_head_centroid_timepoint[0]], [back_head_centroid_timepoint[1], front_head_centroid_timepoint[1]], 'bo-')
                    plt.title('topdown head angle at start of trial')
                    plt.savefig(fig4_path, dpi=300)
                    plt.close()

                if savefig is True:
                    fig5_dir = savepath_input + '/' + current_trial + '/'
                    if not os.path.exists(fig5_dir):
                        os.makedirs(fig5_dir)
                    fig5_path = fig5_dir + 'dlc_topdown_head_angle.png'

                    plt.figure(figsize=(15, 10))
                    plt.plot(atan_head)
                    plt.title('topdown head angle')
                    plt.xlabel('frame')
                    plt.ylabel('angle')
                    plt.savefig(fig5_path, dpi=300)
                    plt.close()

                # this trial's data with no NaNs both post-thresholding and post-y-coordinate correction
                # mask the NaNs
                complete_topdown_out = xr.DataArray.dropna(likeli_thresh_allpts, dim='time', how='all')
                complete_topdown_out['trial'] = current_trial

                # the below block does not work because it requires that all points are present to get the angle of the head, which is not true
                # in this set of data
                # need to implement changes so that it aligns from any head points that are avalible

                # back_head_centroid = xr.DataArray(back_head_centroid, coords=[('time', topdown_coordcor.coords['time']), ('x/y', range(0,2))])
                # front_head_centroid = xr.DataArray(front_head_centroid, coords=[('time', topdown_coordcor.coords['time']), ('x/y', range(0,2))])
                # atan_head = xr.DataArray(atan_head, coords=[('time', topdown_coordcor.coords['time'])])
                # complete_topdown_out = xr.concat([topdown_likeli_thresh_nonan, back_head_centroid, front_head_centroid, atan_head], dim='time')

                # append this trial to all others now that processing is done
                if trial_num == 0:
                    all_topdown_output = complete_topdown_out
                elif trial_num > 0:
                    all_topdown_output = xr.concat([all_topdown_output, complete_topdown_out], dim='trial', fill_value=np.nan)

    return all_topdown_output
