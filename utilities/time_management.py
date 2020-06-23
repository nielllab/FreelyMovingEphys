#####################################################################################
"""
time_management.py of FreelyMovingEphys

Functions for dealing with time and time stamps of data structures for the FreelyMovingEphys
repository. read_time() reads in time functions and returns a list of timestamps associated
with each video frame. It also returns the starting timestamp for each data input.

last modified: June 23, 2020 by Dylan Martins (dmartins@uoregon.edu)
"""
#####################################################################################

import pandas as pd
from datetime import datetime

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
    TS_read = pd.read_csv(data)
    TS_read.columns=['index', 'time']
    TS_read = TS_read['time']
    TS = []
    for row_num in range(0, len(TS_read)):
        TS_row = str(TS_read.loc[row_num])[:-2] # [8:23] indexing is used to remove an extra decimal in milliseconds
        # that python can't recognize because it's too precise and to get rid of the 'time' column heading which is, for
        # some reason, included in the indexed datetime string of every line
        TS_datetime = datetime.strptime(TS_row, '%H:%M:%S.%f')
        TS.append(TS_datetime)
        start = TS[0]
        end = TS[-1]
    # add an explapolated timestep to the end because the actual timesteps are always one too short
    # not an ideal solution, but otherwise there are problems
    topdown_timestep = TS[-1] - TS[-2]
    TS.append(TS[-1] + topdown_timestep)
    if len_main > len(TS):
        time_out = match_deinterlace(TS, topdown_timestep)
    elif len_main == len(TS):
        time_out = TS
    elif len_main < TS:
        print('issue with read_time: more timepoints than there are data')
    return time_out, start, end

####################################################

