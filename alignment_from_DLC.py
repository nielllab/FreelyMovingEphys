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
def align_head_from_DLC(all_topdown_data, trial_list, pt_names, coord_correction_val=1200, num_points=8, thresh=0.99, figures=False, save_figs=True):
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
            all_pt_loop_count = 0
            for pt_num in range(0, len(pt_names)):
                current_pt_loc = pt_names[pt_num]
                likeli_loop_count = 0
                if ' x' in current_pt_loc:
                    print('found ' + current_pt_loc)
                elif ' y' in current_pt_loc:
                    print('found ' + current_pt_loc)
                elif 'likelihood' in current_pt_loc:
                    print('found ' + current_pt_loc)
                    # find the associated x and y points of the selected liklihood
                    # assumes order is x, y, likelihood, will cause problems if isn't true of data...
                    assoc_x_pos = pt_names[pt_num - 2]
                    assoc_x_pt = topdown_coordcor.sel(point_loc=assoc_x_pos)
                    assoc_y_pos = pt_names[pt_num - 1]
                    assoc_y_pt = topdown_coordcor.sel(point_loc=assoc_y_pos)

                    # select only the liklihood data for this point
                    likeli_pt = topdown_coordcor.sel(point_loc=current_pt_loc)

                    # get number of frames to use for
                    frame_coords = topdown_coordcor.coords['frame']
                    frame_len = len(frame_coords)

                    # go into each coordinate in the positinoal data that has been parced out; if the likelihood fell
                    # below the threshold in 'likeli_thresh_pt', the x and y positions associated with that frame of the
                    # video will be set to a np.nan
                    print('running through individual positions in x, y, and likelihood')
                    for row in range(0, frame_len):
                        likeli_pos = likeli_pt[row]
                        if likeli_pos < thresh or likeli_pos == np.nan: # threshold the likelihood
                            x_out = np.nan
                            y_out = np.nan
                        assoc_x_pt[row] = x_out
                        assoc_y_pt[row] = y_out
                        print('concat')
                        # concat together x, y, and likelihood
                        thresh_1row = xr.concat([assoc_x_pt, assoc_y_pt, likeli_pos], dim='frame')
                        print('append1')
                        # append 'thresh_1row' to a new DataArray along dim='frame'
                        if row == 0:
                            likeli_thresh_allrows = thresh_1row
                        elif row > 0:
                            likeli_thresh_allrows = xr.concat([likeli_thresh_allrows, thresh_1row], dim='frame', fill_value=np.nan)
                    print('append2')
                    if likeli_loop_count == 0:
                        likeli_thresh_allpts = likeli_thresh_allrows
                        all_pt_loop_count = all_pt_loop_count + 1
                    elif likeli_loop_count > 0:
                        likeli_thresh_allpts = xr.concat([likeli_thresh_allpts, likeli_thresh_allrows], dim='point_loc', fill_value=np.nan)
                    likeli_loop_count = likeli_loop_count + 1

            print(likeli_thresh_allpts)

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

            #
            # del likeli_thresh_topdown['temp_pts']
            # likeli_thresh_topdown.coords('point_loc')

                #     only_good_topdown_data =
                #     # get the number of NaNs to know how many bad timepoints there are
                #     sum_of_bad_topdown_data.pt_str = pd.nansum(only_good_topdown_data)
                #     print('only good')
                #     print(only_good_topdown_data)
                #     print('sum of bad')
                #     print(sum_of_bad_topdown_data)
                # elif ' x' in pt_str:
                #     print('only an x location, no likelihood to extract')
                # elif ' y' in pt_str:
                #     print('only an y location, no likelihood to extract')
                # else:
                #     print('no locations passed')

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

    return likeli_thresh_allpts

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