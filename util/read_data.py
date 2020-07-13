"""
FreelyMovingEphys data reading utilities
read_data.py

Last modified July 12, 2020
"""

# package imports
import pandas as pd
import numpy as np
import xarray as xr
from glob import glob
import os

# read in .h5 DLC files and manage column names
def open_h5(path):
    try:
        pts = pd.read_hdf(path)
    except ValueError:
        # read in .h5 file when there is a key set in corral_files.py
        pts = pd.read_hdf(path, key='data')
    # organize columns of pts
    pts.columns = [' '.join(col[:][1:3]).strip() for col in pts.columns.values]
    pts = pts.rename(columns={pts.columns[n]: pts.columns[n].replace(' ', '_') for n in range(len(pts.columns))})
    pt_loc_names = pts.columns.values

    return pts, pt_loc_names

# for dlc_intake's use of open_h5, prevents errors if data does not exist
def try_open_h5(list_path):
    try:
        path = list_path[0]
    except KeyError:
        path = list_path
    try:
        pts, pt_loc_names = open_h5(path)
    except NotImplementedError:
        pts = None
        pt_loc_names = None

    return pts, pt_loc_names

# match the length of deinterlaced videos with DLC point structures and videos
def match_deinterlace(raw_time, timestep):
    out = []
    for i in raw_time:
        between_time = i + (timestep / 2)
        out.append(i)
        out.append(between_time)

    return out

# read in the timestamps for a camera and adjust to deinterlaced video length if needed
def open_time(path, num_timepoints_in_pts=None):
    # read in the timestamps
    TS_read = pd.read_csv(path, names=['time'])
    TS_read['time'] = pd.to_datetime(TS_read['time'])
    time_out = TS_read['time']

    if num_timepoints_in_pts is not None:
        # test length of the time just read in as it compares to the length of the data, correct for deinterlacing if needed
        timestep = TS_read['time'][1] - TS_read['time'][0]
        if num_timepoints_in_pts > len(TS_read['time']):
            time_out = match_deinterlace(TS_read['time'], timestep)
        elif num_timepoints_in_pts == len(TS_read['time']):
            time_out = TS_read['time']
        elif num_timepoints_in_pts < len(TS_read['time']):
            print('issue with read_time: more timepoints than there are data')
            time_out = TS_read['time']

    return time_out

def try_open_time(list_path, num_timepoints_in_pts=None):
    try:
        path = list_path[0]
    except KeyError:
        path = list_path
    # test to see if data exist, read in if exists
    time_out = open_time(path, num_timepoints_in_pts)

    return time_out

# convert xarray DataArray of DLC x and y positions and likelihood values into separate pandas data structures
def split_xyl(eye_names, eye_data, thresh):
    x_locs = []
    y_locs = []
    likeli_locs = []
    for loc_num in range(0, len(eye_names)):
        loc = eye_names[loc_num]
        if '_x' in loc:
            x_locs.append(loc)
        elif '_y' in loc:
            y_locs.append(loc)
        elif 'likeli' in loc:
            likeli_locs.append(loc)
    # get the xarray, split up into x, y,and likelihood
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
    x_pts = xr.DataArray.squeeze(x_pts)
    y_pts = xr.DataArray.squeeze(y_pts)
    # convert to dataframe, transpose so points are columns
    x_vals = xr.DataArray.to_pandas(x_pts).T
    y_vals = xr.DataArray.to_pandas(y_pts).T

    return x_vals, y_vals, likeli_pts

# make a path from pieces, search for files with a glob function, and then sort outputs for a list of glob keys
def find_paths(main_path, glob_keys):
    out = []
    for key in glob_keys:
        path_list = sorted(glob(os.path.join(main_path, key)))
        out.append(path_list)
    return out

# build an xarray DataArray of between zero and three camera inputs
def read_paths(path1=None, timepath1=None, path2=None, timepath2=None, path3=None, timepath3=None):
    if path1 is not None:
        view1, names1 = try_open_h5(path1)
    if path2 is not None:
        view2, names2 = try_open_h5(path2)
    if path3 is not None:
        view3, names3 = try_open_h5(path3)
    if timepath1 is not None:
        time1 = try_open_time(timepath1, len(view1))
    if timepath2 is not None:
        time2 = try_open_time(timepath2, len(view2))
    if timepath3 is not None:
        time3 = try_open_time(timepath3, len(view3))

    if view1 is not None:
        xdata = xr.DataArray(view1, dims=['frame', 'point_loc'])
        xdata['view'] = 'v1'
        alltime = pd.DataFrame(time1, columns=['v1'])
        if view2 is not None:
            v2 = xr.DataArray(view2, dims=['frame', 'point_loc'])
            v2['view'] = 'v2'
            xdata = xr.concat([xdata, v2], dim='view', fill_value=np.nan)
            v2t = pd.DataFrame(time2, columns=['v2'])
            alltime = alltime.join(v2t)
            if view3 is not None:
                v3 = xr.DataArray(view3, dims=['frame', 'point_loc'])
                v3['view'] = 'v3'
                xdata = xr.concat([xdata, v3], dim='view', fill_value=np.nan)
                v3t = pd.DataFrame(time3, columns=['v3'])
                alltime = alltime.join(v3t)
    elif view1 is None:
        xdata = None
        alltime = None
    if alltime is not None:
        xtime = xr.DataArray(alltime)

    return xdata, xtime, names1













