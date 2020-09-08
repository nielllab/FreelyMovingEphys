"""
read_data.py

Data reading utilities

Last modified September 07, 2020
"""

# package imports
import pandas as pd
import numpy as np
import xarray as xr
from glob import glob
import os
import fnmatch
import dateutil

# glob for subdirectories
def find(pattern, path):
    result = [] # initialize the list as empty
    for root, dirs, files in os.walk(path): # walk though the path directory, and files
        for name in files:  # walk to the file in the directory
            if fnmatch.fnmatch(name,pattern):  # if the file matches the filetype append to list
                result.append(os.path.join(root,name))
    return result # return full list of file of a given type

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

# read in the timestamps for a camera and adjust to deinterlaced video length if needed
def open_time(path, dlc_len=None):
    # read in the timestamps if they've come directly from cameras
    read_time = pd.read_csv(open(path, 'rU'), encoding='utf-8', engine='c', header=None)
    time_in = pd.to_timedelta(read_time.squeeze(), unit='us')

    # auto check if vids were deinterlaced
    if dlc_len is not None:
        # test length of the time just read in as it compares to the length of the data, correct for deinterlacing if needed
        timestep = np.median(np.diff(time_in, axis=0))
        if dlc_len > len(time_in):
            time_out = np.zeros(np.size(time_in, 0)*2)
            # shift each deinterlaced frame by 0.5 frame period forward/backwards relative to timestamp
            time_out[::2] = time_in + 0.25 * timestep
            time_out[1::2] = time_in - 0.25 * timestep
        elif dlc_len == len(time_in):
            time_out = time_in
        elif dlc_len < len(time_in):
            time_out = time_in
    elif dlc_len is None:
        time_out = time_in

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

# build an xarray DataArray of the a single camera's dlc point .h5 files and .csv timestamp corral_files
# function is used for any camera view regardless of type, though extension must be specified in 'view' argument
def h5_to_xr(pt_path, time_path, view):
    if pt_path is not None and pt_path != []:
        if isinstance(pt_path, list):
            TOP, names = open_h5(pt_path[0])
        else:
            TOP, names = open_h5(pt_path)
        if isinstance(time_path, list):
            time = open_time(time_path[0], len(TOP))
        else:
            time = open_time(time_path, len(TOP))
        TOPpts = xr.DataArray(TOP, dims=['frame', 'point_loc'])
        TOPpts.name = view
        TOPpts = TOPpts.assign_coords(timestamps=('frame', time[1:])) # indexing [1:] into time because first row is the empty header, 0
    elif pt_path is None or pt_path == []:
        TOPpts = None

    return TOPpts, names

# Sort out what the first timestamp in all DataArrays is so that videos can be set to start playing at the corresponding frame
def find_start_end(topdown_data, leftellipse_data, rightellipse_data, side_data):

    # bin the times
    topdown_binned = topdown_data.resample(time='10ms').mean()
    left_binned = leftellipse_data.resample(time='10ms').mean()
    right_binned = rightellipse_data.resample(time='10ms').mean()
    side_binned = side_data.resample(time='10ms').mean()

    # get binned times for each
    td_bintime = topdown_binned.coords['timestamps'].values
    le_bintime = left_binned.coords['timestamps'].values
    re_bintime = right_binned.coords['timestamps'].values
    sd_bintime = side_binned.coords['timestamps'].values

    print('topdown: ' + str(td_bintime[0]) + ' / ' + str(td_bintime[-1]))
    print('left: ' + str(le_bintime[0]) + ' / ' + str(le_bintime[-1]))
    print('right: ' + str(re_bintime[0]) + ' / ' + str(re_bintime[-1]))
    print('side: ' + str(sd_bintime[0]) + ' / ' + str(sd_bintime[-1]))

    # find the last timestamp to start a video
    first_real_time = max([td_bintime[0], le_bintime[0], re_bintime[0], sd_bintime[0]])

    # find the first end of a video
    last_real_time = min([td_bintime[-1], le_bintime[-1], re_bintime[-1], sd_bintime[-1]])

    # find which position contains the timestamp that matches first_real_time and last_real_time
    td_startframe = next(i for i, x in enumerate(td_bintime) if x == first_real_time)
    td_endframe = next(i for i, x in enumerate(td_bintime) if x == last_real_time)
    left_startframe = next(i for i, x in enumerate(le_bintime) if x == first_real_time)
    left_endframe = next(i for i, x in enumerate(le_bintime) if x == last_real_time)
    right_startframe = next(i for i, x in enumerate(re_bintime) if x == first_real_time)
    right_endframe = next(i for i, x in enumerate(re_bintime) if x == last_real_time)
    side_startframe = next(i for i, x in enumerate(sd_bintime) if x == first_real_time)
    side_endframe = next(i for i, x in enumerate(sd_bintime) if x == last_real_time)

    return td_startframe, td_endframe, left_startframe, left_endframe, right_startframe, right_endframe, side_startframe, side_endframe, first_real_time, last_real_time
