"""
track_side.py

side camera tracking utilities
"""
import xarray as xr
import pandas as pd
import numpy as np

def side_tracking(sidedlc, config):
    """
    interpolate and threshold side camera dlc data
    INPUTS:
        sidedlc -- .h5 to DLC data
        config -- options dict
    """
    # get point names
    side_pt_names = list(sidedlc['point_loc'].values)

    # likelihood threshold
    likeli_loop_count = 0
    for pt_num in range(0, len(side_pt_names)):

        current_pt_loc = side_pt_names[pt_num]
        
        # threshold based on likleihood
        if 'likelihood' in current_pt_loc:
            # find the associated x and y points of the selected likelihood
            # assumes order is x, y, likelihood, will cause problems if isn't true of data...
            assoc_x_pos = side_pt_names[pt_num - 2]
            assoc_x_pt = sidedlc.sel(point_loc=assoc_x_pos)
            assoc_y_pos = side_pt_names[pt_num - 1]
            assoc_y_pt = sidedlc.sel(point_loc=assoc_y_pos)

            # select only the likelihood data for this point
            likeli_pt = sidedlc.sel(point_loc=current_pt_loc)

            # set x/y coords to NaN where the likelihood is below threshold value
            assoc_x_pt[likeli_pt < config['lik_thresh']] = np.nan
            assoc_y_pt[likeli_pt < config['lik_thresh']] = np.nan

            likeli_thresh_1loc = xr.concat([assoc_x_pt, assoc_y_pt, likeli_pt], dim='point_loc')

            if likeli_loop_count == 0:
                sidepts_out = likeli_thresh_1loc
            elif likeli_loop_count > 0:
                sidepts_out = xr.concat([sidepts_out, likeli_thresh_1loc], dim='point_loc', fill_value=np.nan)

            likeli_loop_count = likeli_loop_count + 1

    return sidepts_out

def side_angle(sidepts):
    """
    get angle of head from side view
    outputs list of angles in radians
    """
    head_ang = []
    # loop through each frame and get the angle for each
    for frame_num in range(0,np.size(sidepts['frame'].values)):
        nosex = sidepts.sel(point_loc='Nose_x', frame=frame_num)
        nosey = sidepts.sel(point_loc='Nose_y', frame=frame_num)
        earx = sidepts.sel(point_loc='LEar_x', frame=frame_num)
        eary = sidepts.sel(point_loc='LEar_y', frame=frame_num)

        x_dist = (nosex - earx)
        y_dist = (nosey - eary)
        th = np.arctan2(y_dist, x_dist)

        head_ang.append(float(th))

    xr_out = xr.DataArray(head_ang, dims=['frame'])

    return xr_out