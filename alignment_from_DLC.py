#####################################################################################
"""
alignment_from_DLC.py of FreelyMovingEphys

Reads in .csv file from deepLabCut, computes head position.

Adapted from /niell-lab-analysis/freely moving/alignHead.m

last modified: May 18, 2020
"""
#####################################################################################
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import xarray as xr

def align_head_from_DLC(raw_data, num_points=8, thresh=0.99): #default num_points set to 8; 9&10 would include cricket points in Angie's prey capture data

    # select y head points and subtract them from 1200

    # threshold points using "thresh" (find times when all points are good); want high values only
    good_points = np.where(raw_data > thresh)
    print(good_points)
    num_good_points = np.sum(good_points)

    plt.figure(1)
    plt.plot(num_good_points)
    plt.ylabel('number of good frames')
    plt.xlabel('frame')
    plt.show()

    # find centroid at each time point
    centroid = pd.DataFrame.mean(points)
    centered = pd.DataFrame([])
    for point_count in range(0,num_points):
        centered = [points - centroid for i in points]
        frames = pd.DataFrame([point_count, centered])
        centered.append([frames])
    print('centered shape: ' + str(np.shape(centered)))

    # choose a reference image; in the future: bootstrapping, choose 10 random ones and average the results
    reference_number = min(np.where(num_good_points == num_points))
    reference_frame = centered[reference_number]
    print(reference_frame)

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

    return points
    #return aligned_x, aligned_y, aligned_speed, theta, dtheta,