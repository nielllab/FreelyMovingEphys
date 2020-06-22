#####################################################################################
"""
time_management.py of FreelyMovingEphys

Functions for dealing with time and time stamps of data structures for the FreelyMovingEphys
repository. read_time() reads in time functions and returns a list of timestamps associated
with each video frame. It also returns the starting timestamp for each data input.

last modified: June 11, 2020 by Dylan Martins (dmartins@uoregon.edu)
"""
#####################################################################################

import pandas as pd
from datetime import datetime

####################################################
def read_time(data):
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
    return TS, start, end

####################################################


