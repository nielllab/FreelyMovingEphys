#####################################################################################
"""
topdown_preening.py

Take in the xarray of all trial's topdown DeepLabCut points, threshold them based on
likelihood, correct their coordinates if needed, get out the head angle by calling
functions in head_angle.py, and return the data in an xarray DataArray.

last modified: June 28, 2020
"""
#####################################################################################
# import packages
import pandas as pd
import numpy as np
import matplotlib
import xarray as xr
import os
from matplotlib import pyplot as plt
import tkinter
import math

matplotlib.use('TkAgg')

from utilities.data_reading import test_trial_presence
from utilities.data_cleaning import split_xyl
from utilities.head_angle import align_head

####################################################

def preen_topdown_data(all_topdown_data, trial_list, pt_names, savepath_input, coord_correction_val=1200, num_points=8, thresh=0.99, savefig=False):

    # run through each trial individually
    for trial_num in range(0, len(trial_list)):
        # get the name of the current trial
        current_trial = trial_list[trial_num]
        test_trial = test_trial_presence(all_topdown_data, current_trial)
        if test_trial is True:
            with all_topdown_data.sel(trial=current_trial) as topdown_data:
                # interpolate across NaNs fro each point_loc, then piece dataset back together
                topdown_interp = xr.DataArray.interpolate_na(topdown_data, dim='frame', use_coordinate='frame', method='linear')

                # for each point location in the topdown input data, select y head points and subtract them from int to fix coordinates
                y_names = [name for name in pt_names if '_y' in name]
                x_names = [name for name in pt_names if '_x' in name]
                l_names = [name for name in pt_names if 'lik' in name]
                y_data = topdown_interp.sel(point_loc=y_names) - coord_correction_val
                x_data = topdown_interp.sel(point_loc=x_names)
                l_data = topdown_interp.sel(point_loc=l_names)
                topdown_coordcor = xr.concat([x_data, y_data, l_data], dim='point_loc', fill_value=np.nan)

                # make figure of nose position over time, with start and finish labeled in green and red respectively
                if savefig is True:
                    fig1_dir = savepath_input + '/' + current_trial + '/'
                    if not os.path.exists(fig1_dir):
                        os.makedirs(fig1_dir)
                    fig1_path = fig1_dir + 'nose_position_over_time.png'

                    # for now, just drop NaNs that remain in the topdown_interp xarray after interpolation
                    # coordcor_pts_wout_nans = drop_leading_and_lagging_nans(topdown_coordcor, pt_names)
                    nose_x_pts = topdown_coordcor.sel(point_loc='nose_x')
                    nose_y_pts = topdown_coordcor.sel(point_loc='nose_y')
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
                        # find the associated x and y points of the selected likelihood
                        # assumes order is x, y, likelihood, will cause problems if isn't true of data...
                        assoc_x_pos = pt_names[pt_num - 2]
                        assoc_x_pt = topdown_coordcor.sel(point_loc=assoc_x_pos)
                        assoc_y_pos = pt_names[pt_num - 1]
                        assoc_y_pt = topdown_coordcor.sel(point_loc=assoc_y_pos)

                        # select only the likelihood data for this point
                        likeli_pt = topdown_coordcor.sel(point_loc=current_pt_loc)

                        # set x/y coords to NaN where the likelihood is below threshold value
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
                    nose_x_thresh_pts = likeli_thresh_allpts.sel(point_loc='nose_x')
                    nose_y_thresh_pts = likeli_thresh_allpts.sel(point_loc='nose_y')
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

                # x_vals, y_vals, likeli_pts = split_xyl(pt_names, topdown_coordcor, 0.99)
                # timestamp_list = list(x_vals.index.values)

                # if savefig is True:
                #     frame_slice = timestamp_list[0]
                #     x_to_plot = x_vals.loc[[frame_slice]]
                #     y_to_plot = y_vals.loc[[frame_slice]]
                #
                #     fig3_dir = savepath_input + '/' + current_trial + '/'
                #     if not os.path.exists(fig3_dir):
                #         os.makedirs(fig3_dir)
                #     fig3_path = fig3_dir + 'dlc_topdown_pts_at_time_' + str(frame_slice) + '.png'
                #
                #     plt.figure(figsize=(15, 10))
                #     plt.plot(int(x_to_plot.iloc[0,0]), int(y_to_plot.iloc[0,0]), 'bo')
                #     plt.plot(int(x_to_plot.iloc[0,1]), int(y_to_plot.iloc[0,1]), 'go')
                #     plt.plot(int(x_to_plot.iloc[0,2]), int(y_to_plot.iloc[0,2]), 'ro')
                #     plt.plot(int(x_to_plot.iloc[0,3]), int(y_to_plot.iloc[0,3]), 'co')
                #     plt.plot(int(x_to_plot.iloc[0,4]), int(y_to_plot.iloc[0,4]), 'mo')
                #     plt.plot(int(x_to_plot.iloc[0,5]), int(y_to_plot.iloc[0,5]), 'yo')
                #     plt.plot(int(x_to_plot.iloc[0,6]), int(y_to_plot.iloc[0,6]), 'ko')
                #     plt.title('topdown dlc points at time ' + str(frame_slice) + ' of ' + str(current_trial))
                #     plt.savefig(fig3_path, dpi=300)
                #     plt.close()

                # align the head of the mouse from the topdown view, even if some points are missing
                # theta_all, aligned_all = align_head(topdown_coordcor, timestamp_list, pt_names)

                # if savefig is True:
                #     fig5_dir = savepath_input + '/' + current_trial + '/'
                #     if not os.path.exists(fig5_dir):
                #         os.makedirs(fig5_dir)
                #     fig5_path = fig5_dir + 'dlc_topdown_head_angle.png'
                #
                #     plt.figure(figsize=(15, 10))
                #     plt.plot(theta_all)
                #     plt.title('topdown head angle')
                #     plt.xlabel('frame')
                #     plt.ylabel('angle')
                #     plt.savefig(fig5_path, dpi=300)
                #     plt.close()

                # this trial's data with no NaNs both post-thresholding and post-y-coordinate correction
                # mask the NaNs
                likeli_thresh_allpts['trial'] = current_trial

                # append this trial to all others now that processing is done
                if trial_num == 0:
                    all_topdown_output = likeli_thresh_allpts
                elif trial_num > 0:
                    all_topdown_output = xr.concat([all_topdown_output, likeli_thresh_allpts], dim='trial', fill_value=np.nan)

    return all_topdown_output
