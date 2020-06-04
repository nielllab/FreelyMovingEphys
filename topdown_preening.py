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

last modified: June 3, 2020 by Dylan Martins (dmartins@uoregon.edu)
"""
#####################################################################################
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import xarray as xr

####################################################
def drop_leading_and_lagging_nans(data, loc_names):
    '''
    Drop the NaNs that start and end a time seris
    :param data: xarray DataArray of all point locations
    :param loc_names: list of names of each of the point locations
    :return: xarray DataArray for individual points without NaNs at start and end (it will start and end with the first and last numbers)
    '''
    for loc_num in range(0, len(loc_names)):
        # get name of each tagged point in 'data', then index into 'data' to get that tagged point
        this_loc_name = loc_names[loc_num]
        loc_data = data.sel(point_loc=this_loc_name)
        # find first and last non-NaN value and drop everything that comes before and after these
        # ends up with an xarray with real start and end points instead of filled NaN values
        true_where_valid = pd.notna(loc_data)
        index_of_valid = [i for i, x in enumerate(true_where_valid) if x]
        if index_of_valid != []:
            # index into valid positions and select valid data
            first_valid = index_of_valid[0]
            last_valid = index_of_valid[-1]
            valid_data = data.sel(frame=range(first_valid, last_valid))
        elif index_of_valid == []:
            print('no NaNs could be found')
    return valid_data

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
        interp_item = xr.DataArray.interpolate_na(data_to_interp, dim='frame', use_coordinate='frame', method='linear')

        # rebuild the post-interpolation DataArray
        if loc_num == 0:
            interp_data = xr_looped_append(interp_item, interp_item, this_loc_name, 'point_loc', loc_num)
        if loc_num > 0:
            interp_data = xr_looped_append(interp_data, interp_item, this_loc_name, 'point_loc', loc_num)

    return interp_data

####################################################
def preen_topdown_data(all_topdown_data, trial_list, pt_names, coord_correction_val=1200, num_points=8, thresh=0.99, figures=False, save_figs=True):
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
                    nonan_y_topdown = xr.DataArray.dropna(orig_y_topdown, dim='frame')
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
            if figures==True:
                # for now, just drop NaNs that remain in the topdown_interp xarray after interpolation
                coordcor_pts_wout_nans = drop_leading_and_lagging_nans(topdown_coordcor, pt_names)
                nose_x_pts = coordcor_pts_wout_nans.sel(point_loc='nose x')
                nose_y_pts = coordcor_pts_wout_nans.sel(point_loc='nose y')
                plt.figure()
                plt.title('mouse nose x/y path before likelihood threshold')
                plt.plot(np.squeeze(nose_x_pts), np.squeeze(nose_y_pts))
                plt.plot((np.squeeze(nose_x_pts)[0]), (np.squeeze(nose_y_pts)[0]), 'go') # starting point
                plt.plot((np.squeeze(nose_x_pts)[-1]), (np.squeeze(nose_y_pts)[-1]), 'ro')  # ending point
                plt.show()

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

                    # select only the liklihood data for this point
                    likeli_pt = topdown_coordcor.sel(point_loc=current_pt_loc)

                    # get number of frames to use for indexing through all positions
                    frame_coords = topdown_coordcor.coords['frame']
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

            if figures == True:
                # make a plot of the mouse's path, where positions that fall under threshold will be NaNs
                nose_x_thresh_pts = likeli_thresh_allpts.sel(point_loc='nose x')
                nose_y_thresh_pts = likeli_thresh_allpts.sel(point_loc='nose y')
                # mask the NaNs, but only for the figure (don't want to lose time information for actual analysis)
                nose_x_thresh_nonan_pts = nose_x_thresh_pts[np.isfinite(nose_x_thresh_pts)]
                nose_y_thresh_nonan_pts = nose_y_thresh_pts[np.isfinite(nose_y_thresh_pts)]
                plt.figure()
                plt.title('mouse nose x/y path after likelihood threshold')
                plt.plot(np.squeeze(nose_x_thresh_nonan_pts), np.squeeze(nose_y_thresh_nonan_pts))
                plt.plot((np.squeeze(nose_x_thresh_nonan_pts)[0]), (np.squeeze(nose_y_thresh_nonan_pts)[0]), 'go')  # starting point
                plt.plot((np.squeeze(nose_x_thresh_nonan_pts)[-1]), (np.squeeze(nose_y_thresh_nonan_pts)[-1]), 'ro')  # ending point
                plt.show()

            # this trial's data with no NaNs both post-thresholding and post-y-coordinate correction
            # mask the NaNs
            topdown_likeli_thresh_nonan = xr.DataArray.dropna(likeli_thresh_allpts, dim='frame', how='all')
            topdown_likeli_thresh_nonan['trial'] = current_trial

            # append this trial to all others now that processing is done
            if trial_num == 0:
                all_topdown_output = topdown_likeli_thresh_nonan
            elif trial_num > 0:
                all_topdown_output = xr.concat([all_topdown_output, topdown_likeli_thresh_nonan], dim='trial', fill_value=np.nan)

    return all_topdown_output