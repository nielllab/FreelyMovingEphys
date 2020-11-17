"""
read_data.py

functions for reading in and manipulating data and time

Nov. 17, 2020
"""

import pandas as pd
import numpy as np
import xarray as xr
from glob import glob
import os
import fnmatch
import dateutil
import cv2
from tqdm import tqdm
from datetime import datetime
import time
import argparse

# get user inputs
def pars_args():
    parser = argparse.ArgumentParser(description='deinterlace videos and adjust timestamps to match')
    parser.add_argument('-c', '--json_config_path', 
        default='~/Desktop/preprocessing_config.json',
        help='path to video analysis config file')
    args = parser.parse_args()
    
    return args

# glob for subdirectories
def find(pattern, path):
    result = [] # initialize the list as empty
    for root, dirs, files in os.walk(path): # walk though the path directory, and files
        for name in files:  # walk to the file in the directory
            if fnmatch.fnmatch(name,pattern):  # if the file matches the filetype append to list
                result.append(os.path.join(root,name))
    return result # return full list of file of a given type

# check if path exists, if not then create directory
def check_path(basepath, path):
    if path in basepath:
        return basepath
    elif not os.path.exists(os.path.join(basepath, path)):
        os.makedirs(os.path.join(basepath, path))
        print('Added Directory:'+ os.path.join(basepath, path))
        return os.path.join(basepath, path)
    else:
        return os.path.join(basepath, path)

# read in .h5 DLC files and manage column names
def open_h5(path):
    try:
        pts = pd.read_hdf(path)
    except ValueError:
        # read in .h5 file when there is a key set in corral_files.py
        pts = pd.read_hdf(path, key='data')
    # organize columns
    pts.columns = [' '.join(col[:][1:3]).strip() for col in pts.columns.values]
    pts = pts.rename(columns={pts.columns[n]: pts.columns[n].replace(' ', '_') for n in range(len(pts.columns))})
    pt_loc_names = pts.columns.values

    return pts, pt_loc_names

# open h5 file of a multianimal DLC project
def open_ma_h5(path):
    pts = pd.read_hdf(path)
    # flatten columns from MultiIndex 
    pts.columns = ['_'.join(col[:][1:]).strip() for col in pts.columns.values]

    return pts

# read in the timestamps for a camera and adjust to deinterlaced video length if needed
def open_time(path, dlc_len=None, force_shift=False):
    # read in the timestamps if they've come directly from cameras
    read_time = pd.read_csv(path, encoding='utf-8', engine='c', header=None).squeeze()
    if read_time[0] == 0: # in case header == 0, which is true of some files, drop that header which will have been read in as the first entry
        read_time = read_time[1:]
    time_in = []
    fmt = '%H:%M:%S.%f'
    if read_time.dtype!=np.float64:
        for current_time in read_time:
            currentT = str(current_time).strip()
            try:
                t = datetime.strptime(currentT,fmt)
            except ValueError as v:
                ulr = len(v.args[0].partition('unconverted data remains: ')[2])
                if ulr:
                    currentT = currentT[:-ulr]
            try:
                time_in.append((datetime.strptime(currentT, '%H:%M:%S.%f') - datetime.strptime('00:00:00.000000', '%H:%M:%S.%f')).total_seconds())
            except ValueError:
                time_in.append(np.nan)
        time_in = np.array(time_in)
    else:
        time_in = read_time.values

    # auto check if vids were deinterlaced
    if dlc_len is not None:
        # test length of the time just read in as it compares to the length of the data, correct for deinterlacing if needed
        timestep = np.nanmedian(np.diff(time_in, axis=0))
        if dlc_len > len(time_in):
            time_out = np.zeros(np.size(time_in, 0)*2)
            # shift each deinterlaced frame by 0.5 frame period forward/backwards relative to timestamp
            time_out[::2] = time_in - 0.25 * timestep
            time_out[1::2] = time_in + 0.25 * timestep
        elif dlc_len == len(time_in):
            time_out = time_in
        elif dlc_len < len(time_in):
            time_out = time_in
    elif dlc_len is None:
        time_out = time_in

    # force the times to be shifted if the user is sure it should be done
    if force_shift is True:
        # test length of the time just read in as it compares to the length of the data, correct for deinterlacing
        timestep = np.nanmedian(np.diff(time_in, axis=0))
        time_out = np.zeros(np.size(time_in, 0)*2)
        # shift each deinterlaced frame by 0.5 frame period forward/backwards relative to timestamp
        time_out[::2] = time_in - 0.25 * timestep
        time_out[1::2] = time_in + 0.25 * timestep

    return time_out

# read in the timestamps for a camera when they come from a csv file with data in it
# this does not read or open a file, it takes in a DataFrame column
# assumes that timestamps have 10 characters on the front end, %Y-%m-%dT and 6 on the back end, -08:00
# written to be used with ball rotation timestamps
def open_time1(read_time):
    time_in = []
    fmt = '%H:%M:%S.%f'
    if read_time.dtype!=np.float64:
        for current_time in read_time:
            current_time = current_time[11:-6]
            currentT = str(current_time).strip()
            try:
                t = datetime.strptime(currentT,fmt)
            except ValueError as v:
                ulr = len(v.args[0].partition('unconverted data remains: ')[2])
                if ulr:
                    currentT = currentT[:-ulr]
            try:
                time_in.append((datetime.strptime(currentT, '%H:%M:%S.%f') - datetime.strptime('00:00:00.000000', '%H:%M:%S.%f')).total_seconds())
            except ValueError:
                time_in.append(np.nan)
        time_in = np.array(time_in)
    else:
        time_in = read_time.values

    return time_in

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
    likeli_pts = xr.DataArray.squeeze(likeli_pts)
    # convert to dataframe, transpose so points are columns
    x_vals = xr.DataArray.to_pandas(x_pts).T
    y_vals = xr.DataArray.to_pandas(y_pts).T
    likeli_pts = xr.DataArray.to_pandas(likeli_pts).T

    return x_vals, y_vals, likeli_pts

# build an xarray DataArray of the a single camera's dlc point .h5 files and .csv timestamp corral_files
# function is used for any camera view regardless of type, though extension must be specified in 'view' argument
def h5_to_xr(pt_path, time_path, view, config):
    if pt_path is not None and pt_path != []:
        if 'TOP' in view and config['multianimal_TOP'] is True:
            # add a step to convert pickle files here?
            pts = open_ma_h5(pt_path)
        else:
            pts, names = open_h5(pt_path)
        time = open_time(time_path, len(pts))
        xrpts = xr.DataArray(pts, dims=['frame', 'point_loc'])
        xrpts.name = view
        try:
            xrpts = xrpts.assign_coords(timestamps=('frame', time[1:])) # indexing [1:] into time because first row is the empty header, 0
        except ValueError:
            # this is both messy and temporary -- trying to fix issue: ValueError: conflicting sizes for dimension 'frame': length 71521 on 'timestamps' and length 71522 on 'frame'
            timestep = time[1] - time[0]
            last_value = time[-1] + timestep
            new_time = np.append(time, pd.Series(last_value))
            xrpts = xrpts.assign_coords(timestamps=('frame', new_time[1:]))
    elif pt_path is None or pt_path == []:
        if time_path is not None and time_path != []:
            time = open_time(time_path)
            xrpts = xr.DataArray(np.zeros([len(time)-1]), dims=['frame'])
            xrpts = xrpts.assign_coords({'frame':range(0,len(xrpts))})
            xrpts = xrpts.assign_coords(timestamps=('frame', time[1:]))
            names = None
        elif time_path is None or time_path == []:
            xrpts = None; names = None

    return xrpts

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

# calculates xcorr ignoring NaNs without altering timing
# adapted from /niell-lab-analysis/freely moving/nanxcorr.m
def nanxcorr(x, y, maxlag=25):
    lags = range(-maxlag, maxlag)
    cc = []
    for i in range(0,len(lags)):
        # shift data
        yshift = np.roll(y, lags[i])
        # get index where values are usable in both x and yshift
        use = ~pd.isnull(x + yshift)
        # some restructuring
        x_arr = np.asarray(x, dtype=object); yshift_arr = np.asarray(yshift, dtype=object)
        x_use = x_arr[use]; yshift_use = yshift_arr[use]
        # normalize
        x_use = (x_use - np.mean(x_use)) / (np.std(x_use) * len(x_use))
        yshift_use = (yshift_use - np.mean(yshift_use)) / np.std(yshift_use)
        # get correlation
        cc.append(np.correlate(x_use, yshift_use))
    cc_out = np.hstack(np.stack(cc))
    return cc_out, lags

# add videos to xarray
# will downsample by ratio in config file and convert to black and white uint8
def format_frames(vid_path, config):
    print('formatting video into DataArray')
    vidread = cv2.VideoCapture(vid_path)
    all_frames = np.empty([int(vidread.get(cv2.CAP_PROP_FRAME_COUNT)),
                        int(vidread.get(cv2.CAP_PROP_FRAME_HEIGHT)*config['dwnsmpl']),
                        int(vidread.get(cv2.CAP_PROP_FRAME_WIDTH)*config['dwnsmpl'])], dtype=np.uint8)
    for frame_num in tqdm(range(0,int(vidread.get(cv2.CAP_PROP_FRAME_COUNT)))):
        ret, frame = vidread.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sframe = cv2.resize(frame, (0,0), fx=config['dwnsmpl'], fy=config['dwnsmpl'], interpolation=cv2.INTER_NEAREST)
        all_frames[frame_num,:,:] = sframe.astype(np.int8)

    formatted_frames = xr.DataArray(all_frames.astype(np.int8), dims=['frame', 'height', 'width'])
    formatted_frames.assign_coords({'frame':range(0,len(formatted_frames))})
    del all_frames

    return formatted_frames

# align xarrays by time and merge
# first input will start at frame 0, the second input will be aligned to the first using timestamps in nanoseconds
# so that the first frame in a new dimension, 'merge_time', will start at either a positive or negative integer which
# is shifted forward or back from 0
def merge_xr_by_timestamps(xr1, xr2):
    # round the nanoseseconds in each xarray
    round1 = np.around(xr1['timestamps'].data.astype(np.int), -4)
    round2 = np.around(xr2['timestamps'].data.astype(np.int), -4)
    df2 = pd.DataFrame(round2)
    # where we'll put the index of the closest match in round2 for each value in round1
    ind = []
    for step in range(0,len(round1)):
        ind.append(np.argmin(abs(df2 - round1[step])))
    # here, a positive value means that round2 is delayed by that value
    # and a negative value means that round2 is ahead by that value
    delay_behind_other = int(round(np.mean([(i-ind[i]) for i in range(0,len(ind))])))
    # set the two dataarrays up with aligned timestamps
    new1 = xr1.expand_dims({'merge_time':range(0,len(xr1))})
    new2 = xr2.expand_dims({'merge_time':range(delay_behind_other, len(ind)+delay_behind_other)}).drop('timestamps')
    # merge into one dataset
    ds_out = xr.merge([new1,new2], dim='merge_time')

    return ds_out

# use np.vstack with arrays of variable length by filling jaggged ends with NaNs
def nan_vstack(array1, array2):
    lengths = []
    for array in [array1, array2]:
        lengths.append(len(array))
    set_length = max(lengths)
    array_num = 0
    for array in [array1, array2]:
        array_num = array_num + 1
        if len(array) < set_length:
            len_diff = set_length - len(array)
            nan_array = np.empty(len_diff)
            nan_array[:] = np.nan
            array = np.concatenate([array, nan_array])
        if array_num == 1:
            array1 = array
        elif array_num == 2:
            stack = np.vstack([array1, array])
    return stack