"""
track_side.py

side camera tracking utilities

Oct. 26, 2020
"""

# package imports
import xarray as xr
import pandas as pd
import numpy as np

# interpolate and threshold side camera dlc data
def side_tracking(side_data, config):

    side_pt_names = list(side_data['point_loc'].values)

    sideinterp = xr.DataArray.interpolate_na(side_data, dim='frame', use_coordinate='frame', method='linear')

    # likelihood threshold
    likeli_loop_count = 0
    for pt_num in range(0, len(side_pt_names)):
        current_pt_loc = side_pt_names[pt_num]
        if 'likelihood' in current_pt_loc:
            # find the associated x and y points of the selected likelihood
            # assumes order is x, y, likelihood, will cause problems if isn't true of data...
            assoc_x_pos = side_pt_names[pt_num - 2]
            assoc_x_pt = sideinterp.sel(point_loc=assoc_x_pos)
            assoc_y_pos = side_pt_names[pt_num - 1]
            assoc_y_pt = sideinterp.sel(point_loc=assoc_y_pos)

            # select only the likelihood data for this point
            likeli_pt = sideinterp.sel(point_loc=current_pt_loc)

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

# def side_angle():
#     return ang