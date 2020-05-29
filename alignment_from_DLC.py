#####################################################################################
"""
alignment_from_DLC.py of FreelyMovingEphys

Reads in .csv file from deepLabCut, computes head position.

Adapted from /niell-lab-analysis/freely moving/alignHead.m

last modified: May 29, 2020
"""
#####################################################################################
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import xarray as xr

def align_head_from_DLC(topdown_data, num_points=8, thresh=0.98, figures=False): #default num_points set to 8; 9&10 would include cricket points in Angie's prey capture data

    # IMPLEMENT: for statement to index through each trial individually

    # get all the point location names
    coords_array = topdown_data.coords['point_loc']
    del coords_array['point_loc'] # deletes the coordinates but retains the array of coord names

    # for each point location in the topdown input data, select y head points and subtract them from 1200 to fix coordinates
    for pt_loc in range(0,len(coords_array)):
        pt_str = coords_array[pt_loc]
        if ' y' in pt_str:
            topdown_data.pt_str = topdown_data.pt_str - 1200

    # figure of nose position over time, with start and finish labeled in green and red respectively
    if figures==True:
        nose_x_pts = topdown_data.sel(point_loc='nose x')
        nose_y_pts = topdown_data.sel(point_loc='nose y')
        plt.figure()
        plt.plot(np.squeeze(nose_x_pts), np.squeeze(nose_y_pts))
        plt.plot((np.squeeze(nose_x_pts)[0]), (np.squeeze(nose_y_pts)[0]), 'go') # starting point
        plt.plot((np.squeeze(nose_x_pts)[-1]), (np.squeeze(nose_y_pts)[-1]), 'ro')  # ending point
        plt.show()

    # threshold points using the input paramater, thresh, to find all times when all points are good (only want high values)
    good_topdown_data = topdown_data
    num_good_topdown_pts = topdown_data
    for pt_loc in range(0, len(coords_array)):
        current_pt_loc = coords_array[pt_loc]
        pt_str = current_pt_loc
        if 'likelihood' in pt_str:
            good_topdown_data.pt_str = topdown_data.where(topdown_data.pt_str >= thresh, topdown_data.pt_str < thresh)
            num_good_topdown_pts.pt_str = good_topdown_data.where(good_topdown_data.pt_str, good_topdown_data.pt_str < thresh)

    # this block isn't working quite yet
    print(num_good_topdown_pts)
    only_good_pts = num_good_topdown_pts == num_points
    bad_frac = 1 - (num_good_topdown_pts / num_points)
    print(bad_frac)

    # NOTE: This figure doesn't work quite yet either
    if figures==True:
        plt.subplots(1,2)
        plt.subplot(121)
        plt.plot(num_good_topdown_pts)
        plt.title('number of good timepoints by frame')
        plt.ylabel('# good points')
        plt.xlabel('frame #')
        plt.ylim([0, num_points])
        plt.subplot(122)
        plt.bar(x=range(0, num_points), height=bad_frac)
        plt.title('fraction of timepoints below threshold')
        plt.ylabel('fraction of bad timepoints')
        plt.xlabel('point #')
        plt.show()

    return good_topdown_data

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