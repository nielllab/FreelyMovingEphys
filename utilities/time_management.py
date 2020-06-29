#####################################################################################
"""
time_management.py

Functions for dealing with time and time stamps of cameras.

last modified: June 28, 2020
"""
#####################################################################################

# import packages
import pandas as pd
from datetime import datetime
from datetime import timedelta
import xarray as xr
import numpy as np

####################################################
def match_deinterlace(raw_time, timestep):
    # match the length of deinterlaced videos with DLC point structures and videos that are twice the length of the timestamp files
    out = []
    for i in raw_time:
        between_time = i + (timestep / 2)
        out.append(i)
        out.append(between_time)
    return out

####################################################
def read_time(data, len_main):
    # read in time values, correct their lengths
    # takes in the time data and the lenght of the main point data it's associated with
    # len_main is used to sort out if the time file is too short because of deinterlacing of video
    TS_read = pd.read_csv(data, names=['time'])
    TS_read['time'] = pd.to_datetime(TS_read['time'])
    TS_read['time'] = TS_read['time'] - TS_read['time'][0]

    if len_main > len(TS_read['time']):
        time_out = match_deinterlace(TS_read['time'], TS_read['time'][1])
    elif len_main == len(TS_read['time']):
        time_out = TS_read['time']
    elif len_main < len(TS_read['time']):
        print('issue with read_time: more timepoints than there are data')
        time_out = TS_read['time']

    return time_out

####################################################

# sort out what the first timestamp in all DataArrays is so that videos can be set to start playing at the corrosponding frame
# currently, this does not account for worldcam timestamps
def find_first_time(topdown_data, leftellipse_data, rightellipse_data):
    # bin times
    print(topdown_data, leftellipse_data, rightellipse_data)
    topdown_binned = topdown_data.resample(time='10ms').mean()
    left_binned = leftellipse_data.resample(time='10ms').mean()
    right_binned = rightellipse_data.resample(time='10ms').mean()

    # get first time for each
    td_first_bintime = topdown_binned.coords['time'].values
    le_first_bintime = left_binned.coords['time'].values
    re_first_bintime = right_binned.coords['time'].values

    # find the last timestamp to start a video
    first_real_time = max([td_first_bintime[0], le_first_bintime[0], re_first_bintime[0]])

    # find the first end of a video
    last_real_time = min([td_first_bintime[-1], le_first_bintime[-1], re_first_bintime[-1]])

    # find which position contains the timestamp that matches first_real_time and last_real_time
    td_startframe = [i for i, x in enumerate(topdown_binned.coords['time'].values) if x == first_real_time]
    td_endframe = [i for i, x in enumerate(topdown_binned.coords['time'].values) if x == last_real_time]
    left_startframe = [i for i, x in enumerate(left_binned.coords['time'].values) if x == first_real_time]
    left_endframe = [i for i, x in enumerate(left_binned.coords['time'].values) if x == last_real_time]
    right_startframe = [i for i, x in enumerate(right_binned.coords['time'].values) if x == first_real_time]
    right_endframe = [i for i, x in enumerate(right_binned.coords['time'].values) if x == last_real_time]

    return td_startframe, td_endframe, left_startframe, left_endframe, right_startframe, right_endframe, first_real_time, last_real_time
