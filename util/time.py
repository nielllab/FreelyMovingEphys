"""
time.py
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

def open_time(path, dlc_len=None, force_shift=False):
    """
    read in the timestamps for a camera and adjust to deinterlaced video length if needed
    INPUTS
        path: path to a timestamp .csv file
        dlc_len: int, number of frames in the DLC data (used to decide if interpolation is needed, but this can be left as None to ignore)
        force_shift:  bool, whether or not to interpolate timestamps without checking
    OUTPUTS
        time_out: timestamps as numpy array
    """
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

def open_time1(read_time):
    """
    read in the timestamps for a camera when they come from a csv file containing other data
    this does not read or open a file, it takes in a DataFrame column
    written to be used with ball rotation timestamps
    INPUTS
        read_time: column of a dataframe to read in and format
    OUTPUTS
        time_in: numpy array of timestamps
    """
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

    return time_in

def find_start_end(topdown_data, leftellipse_data, rightellipse_data, side_data):
    """
    find the first timestamp in all DataArrays is so that videos can be set to start playing at the corresponding frame
    """
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

def merge_xr_by_timestamps(xr1, xr2):
    """
    align xarrays by time and merge
    first input will start at frame 0, the second input will be aligned to the first using timestamps in nanoseconds
    so that the first frame in a new dimension, 'merge_time', will start at either a positive or negative integer which
    is shifted forward or back from 0
    """
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