#####################################################################################
"""
data_cleaning.py of FreelyMovingEphys/utilities/

Various functions to preen xarray data.

last modified: June 15, 2020 by Dylan Martins (dmartins@uoregon.edu)
"""
#####################################################################################
# imports
import xarray as xr
import numpy as np
import pandas as pd

#############################################
def split_xyl(eye_names, eye_data, thresh):
    '''
    Makes a separate pandas DataFrame out of x points and y points from left or right eye from xarray DataArray input
    Also thresholds the likelihood values using input parameter
    :param eye_names: names of each point in the eye data, a list
    :param eye_data: an xarray of x points, y points, and likelihood of
    :param thresh: likelihood threshold value, a decimal value equal to or less than 1
    :return: x_vals, y_vals, likeli_pts
    '''
    x_locs = []
    y_locs = []
    likeli_locs = []
    for loc_num in range(0, len(eye_names)):
        loc = eye_names[loc_num]
        if ' x' in loc:
            x_locs.append(loc)
        elif ' y' in loc:
            y_locs.append(loc)
        elif ' likeli' in loc:
            likeli_locs.append(loc)
        elif loc is None:
            print('loc is None')

    # get the xarray split up into x, y,and likelihood
    for loc_num in range(0, len(likeli_locs)):
        pt_loc = likeli_locs[loc_num]
        if loc_num == 0:
            likeli_pts = eye_data.sel(point_loc=pt_loc)
        elif loc_num > 0:
            likeli_pts = xr.concat([likeli_pts, eye_data.sel(point_loc=pt_loc)], dim='point_loc', fill_value=np.nan)
    for loc_num in range(0, len(x_locs)):
        pt_loc = x_locs[loc_num]
        # threshold from likelihood
        eye_data.sel(point_loc=pt_loc)[eye_data.sel(point_loc=pt_loc) < thresh] = np.nan
        if loc_num == 0:
            x_pts = eye_data.sel(point_loc=pt_loc)
        elif loc_num > 0:
            x_pts = xr.concat([x_pts, eye_data.sel(point_loc=pt_loc)], dim='point_loc', fill_value=np.nan)
    for loc_num in range(0, len(y_locs)):
        pt_loc = y_locs[loc_num]
        # threshold from likelihood
        eye_data.sel(point_loc=pt_loc)[eye_data.sel(point_loc=pt_loc) < thresh] = np.nan
        if loc_num == 0:
            y_pts = eye_data.sel(point_loc=pt_loc)
        elif loc_num > 0:
            y_pts = xr.concat([y_pts, eye_data.sel(point_loc=pt_loc)], dim='point_loc', fill_value=np.nan)

    # drop len=1 dims
    x_pts = xr.DataArray.squeeze(x_pts)
    y_pts = xr.DataArray.squeeze(y_pts)

    # convert to dataframe, transpose so points are columns, and drop trailing NaNs
    x_vals = pd.DataFrame.dropna(xr.DataArray.to_pandas(x_pts).T)
    y_vals = pd.DataFrame.dropna(xr.DataArray.to_pandas(y_pts).T)

    return x_vals, y_vals, likeli_pts

#############################################