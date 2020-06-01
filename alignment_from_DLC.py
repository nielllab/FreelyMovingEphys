#####################################################################################
"""
alignment_from_DLC.py of FreelyMovingEphys

Reads in .csv file from deepLabCut, computes head position.

Adapted from /niell-lab-analysis/freely moving/alignHead.m

last modified: June 1, 2020
"""
#####################################################################################
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import xarray as xr

####################################################
def drop_leading_and_lagging_nans(data, loc_names):
    for loc_num in range(0, len(loc_names)):
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
            interp_data = xr.DataArray(interp_item)
            interp_data['point_loc'] = this_loc_name
        elif loc_num > 0:
            interp_item_to_concat = xr.DataArray(interp_item)
            interp_item_to_concat['point_loc'] = this_loc_name
            interp_data = xr.concat([interp_data, interp_item_to_concat], dim='point_loc', fill_value='NaN')
    return interp_data

####################################################
def align_head_from_DLC(all_topdown_data, trial_list, pt_names, coord_correction_val=1200, num_points=8, thresh=0.99, figures=False):
    '''
    Aligns the head of a mouse from DLC output files which are passed in from load_all_csv.py

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

            # get all the point location names
            coords_array = topdown_data.coords['point_loc']
            del coords_array['point_loc'] # deletes the coordinates but retains the array of coord names

            # for each point location in the topdown input data, select y head points and subtract them from 1200 to fix coordinates
            for pt_loc in range(0, len(coords_array)):
                pt_str = coords_array[pt_loc]
                if ' y' in pt_str:
                    topdown_data.pt_str = topdown_data.pt_str - coord_correction_val

            # interpolate across NaNs fro each point_loc, then piece dataset back together
            topdown_interp = interp_nans_if_any(topdown_data, pt_names)

            # THOUGHT: eliminate NaNs at the start and end of each trial and point so that plots show true start and finish
            # topdown_interp_leadlagdrop = drop_leading_and_lagging_nans(topdown_interp, pt_names)

            # make figure of nose position over time, with start and finish labeled in green and red respectively
            if figures==True:
                # for now, just drop NaNs that remain in the topdown_interp xarray after interpolation
                nose_x_pts = topdown_interp.sel(point_loc='nose x', drop='NaN')
                nose_y_pts = topdown_interp.sel(point_loc='nose y', drop='NaN')
                plt.figure()
                plt.plot(np.squeeze(nose_x_pts), np.squeeze(nose_y_pts))
                plt.plot((np.squeeze(nose_x_pts)[0]), (np.squeeze(nose_y_pts)[0]), 'go') # starting point
                plt.plot((np.squeeze(nose_x_pts)[-1]), (np.squeeze(nose_y_pts)[-1]), 'ro')  # ending point
                plt.show()

            # threshold points using the input paramater, thresh, to find all times when all points are good (only want high values)
            only_good_topdown_data = topdown_data
            sum_of_bad_topdown_data = topdown_data
            for pt_loc in range(0, len(coords_array)):
                current_pt_loc = coords_array[pt_loc]
                pt_str = current_pt_loc
                if 'likelihood' in pt_str:
                    # keep first argument, turn second argument to NaN, and drop second argument in bad_topdown_data_dropped
                    only_good_topdown_data.pt_str = topdown_data.where(topdown_data.pt_str >= thresh, topdown_data.pt_str < thresh)
                    sum_of_bad_topdown_data.pt_str = pd.nansum(only_good_topdown_data)
                    print('only good')
                    print(only_good_topdown_data)
                    print('sum of bad')
                    print(sum_of_bad_topdown_data)

    # # this block isn't working quite yet
    # only_good_pts = num_good_topdown_pts == num_points
    # bad_frac = 1 - (num_good_topdown_pts / num_points)
    # print(bad_frac)
    #
    # # NOTE: This figure doesn't work quite yet either
    # if figures==True:
    #     plt.subplots(1,2)
    #     plt.subplot(121)
    #     plt.plot(num_good_topdown_pts)
    #     plt.title('number of good timepoints by frame')
    #     plt.ylabel('# good points')
    #     plt.xlabel('frame #')
    #     plt.ylim([0, num_points])
    #     plt.subplot(122)
    #     plt.bar(x=range(0, num_points), height=bad_frac)
    #     plt.title('fraction of timepoints below threshold')
    #     plt.ylabel('fraction of bad timepoints')
    #     plt.xlabel('point #')
    #     plt.show()

    return topdown_data

###############################################################
    # good_points = np.where(raw_data > thresh)
    # print(good_points)
    # num_good_points = np.sum(good_points)
    #
    # plt.figure(1)
    # plt.plot(num_good_points)
    # plt.ylabel('number of good frames')
    # plt.xlabel('frame')
    # plt.show()
    #
    # # find centroid at each time point
    # centroid = pd.DataFrame.mean(points)
    # centered = pd.DataFrame([])
    # for point_count in range(0,num_points):
    #     centered = [points - centroid for i in points]
    #     frames = pd.DataFrame([point_count, centered])
    #     centered.append([frames])
    # print('centered shape: ' + str(np.shape(centered)))
    #
    # # choose a reference image; in the future: bootstrapping, choose 10 random ones and average the results
    # reference_number = min(np.where(num_good_points == num_points))
    # reference_frame = centered[reference_number]
    # print(reference_frame)

    # rotate all data to align to bootstrapped reference image
        # select good points
        # loop through range of thetas, rotate image by that much, and calculate how well it matches the reference
        # find the smallest error, and rotate the image by that amount

    # calculate head mean, rotate mean head to align to x-axis

    # calculate centroid
        # for each head point, calculate how far from mean head position
        # calculate error of how far it is from where it should be, then add these up
        # find minimum, get x and y, set as centroid
        # center all points using calculated centroid

    # align time
    # find at least four good times
        # loop over thetas, rotate points, calculate error
        # sum good points
        # loop over thetas, rotate points, calculate error (THIS CAN BE A SEPERATE CALLED FUNCTIONS)
        # find minimum and rotate points accoridngly

        # then a bunch of cricket things that don't apply to our freely moving ephys project (should this still be built in as an option in case prey capture is used?)

    #return aligned_x, aligned_y, aligned_speed, theta, dtheta,