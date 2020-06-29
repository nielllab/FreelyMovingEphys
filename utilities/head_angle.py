#####################################################################################
'''
head_angle.py

Takes in one timepoint's topdown positions and aligns the gets out the aligned position and angle of the head.
This is NOT currently used in by load_from_DLC.py or associated functions, and isn't yet finished being written.

Adapted from niell-lab-analysis/freely moving/alignHead.m

Last modified: June 24, 2020
'''
#####################################################################################

import math
import numpy as np

from utilities.data_cleaning import split_xyl

def rotmat(theta):
    m = [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    return m

def align_head(pt_input, timestamp_list, pt_names):

    x_vals, y_vals, likeli_pts = split_xyl(pt_names, pt_input, 0.99)

    # get a reference frame out
    ref_time_slice = timestamp_list[0]
    ref = np.concatenate([x_vals.loc[[ref_time_slice]], y_vals.loc[[ref_time_slice]]], axis=1)

    data = np.concatenate([x_vals.T, y_vals.T], axis=1)

    centroid = np.squeeze(np.mean(data, axis=0))
    centered = np.zeros(np.shape(centroid), dtype=object)
    theta_good = np.zeros(np.shape(centroid), dtype=object)
    aligned_good = np.zeros(np.shape(centroid), dtype=object)
    for time in range(0, 1 - len(timestamp_list)):
        num_points = len(data[time, :])
        num_real_pts = num_points - np.count_nonzero(~np.isnan(data[time, :]))

        centered_timepoint = np.squeeze(data[time, :]) - centroid
        centered[time] = centered_timepoint

        if num_real_pts == num_points:
            c = centered[time]
            # if there are no NaNs and it's a perfect timepoint, loop through a range of thetas and rotate the frame by that much
            # then, calculate how well it matches the reference
            theta = np.linspace(0, (2 * math.pi), 101)
            theta = theta[1:-1]
            rms = np.zeros(len(theta))
            for i in range(0, len(theta)):
                c_rot = c * rotmat(theta[i])  # rotation
                rms[i] = np.nansum((ref - c_rot) ** 2)  # root mean squared
            # find smallest error and rotate by this amount
            y, ind = min(rms)
            theta_good[time] = theta(ind)
            aligned_good[time] = c * rotmat(theta_good[time])
        elif num_real_pts < num_points:
            theta_good[time] = np.nan
            aligned_good[time] = np.nan

    # calculate mean head from good points across trials
    mean_head = np.mean(aligned_good)

    # calculate the x/y centroid that best matches the defined distances between marked points and the centroid
    mean_distance = np.sqrt((mean_head[0]**2) + (mean_head[1]**2))

    # make a mesh grid that covers x/y position of all head points at this time
    meshx, meshy = np.meshgrid((np.floor(min(data[:, 0])), np.ceil(max(data[:, 0]))), (np.floor(min(data[:, 0])), np.ceil(max(data[:, 0]))))

    # for each head point calculate how far the pixels are from it
    # then calculate error of how far this is from where it should be, then add these up
    for time in range(0, len(timestamp_list)):
        err = 0
        theta_all = []
        aligned_all = []
        for pt_time in range(0, num_points):
            pt = data[:, i]
            if pt != np.nan:
                r = np.sqrt((meshx - data(pt, 0))**2 + (meshy - data(pt, 0))**2) # distance
                err = err + (mean_distance[i] - r)**2 # error

            num_real_pts = num_points - np.count_nonzero(~np.isnan(pt[0]))

            # do the alignment if there are at least 4 good points
            if num_real_pts >= 4:
                c = centered[i]
                # if there are no NaNs and it's a perfect timepoint, loop through a range of thetas and rotate the frame by that much
                # then, calculate how well it matches the reference
                theta = np.linspace(0, (2 * math.pi), 101)
                theta = theta[1:-1]
                rms = np.zeros(len(theta))
                for i in range(0, len(theta)):
                    c_rot = c * rotmat(theta[i]) # rotation
                    rms[i] = np.nansum((ref - c_rot) **2) # root mean squared
                # find smallest error and rotate by this amount
                y, ind = min(rms)
                theta_out = 2 * math.pi - theta(ind)
                aligned_out = c * rotmat(theta(ind))
                theta_all.append(theta_out)
                aligned_all.append(aligned_out)
            elif num_real_pts < 4:
                theta_out = np.nan
                aligned_out = np.nan
                theta_all.append(theta_out)
                aligned_all.append(aligned_out)

    return theta_all, aligned_all