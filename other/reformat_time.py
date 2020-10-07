"""
reformat_time.py

reformat Bonsai timestamps from raw to seconds since midnight
good if time interpolation needs to be done prior to other analysis

Oct. 06, 2020
"""

import argparse
import fnmatch
import numpy as np
import os
import pandas as pd
import dateutil
from datetime import datetime

# glob for subdirectories
def find(pattern, path):
    result = [] # initialize the list as empty
    for root, dirs, files in os.walk(path): # walk though the path directory, and files
        for name in files:  # walk to the file in the directory
            if fnmatch.fnmatch(name,pattern):  # if the file matches the filetype append to list
                result.append(os.path.join(root,name))
    return result # return full list of file of a given type

# read in the timestamps for a camera and adjust to deinterlaced video length if needed
def open_time(path, dlc_len=None, force_shift=False):
    # read in the timestamps if they've come directly from cameras
    read_time = pd.read_csv(path, encoding='utf-8', engine='c', header=0).squeeze()
    startT = (read_time[0])[:-2]
    time_in = []
    for current_time in read_time:
        currentT = current_time[:-2]
        try:
            time_in.append((datetime.strptime(currentT, '%H:%M:%S.%f') - datetime.strptime('00:00:00.000000', '%H:%M:%S.%f')).total_seconds())
        except ValueError:
            time_in.append(np.nan)
    time_in = np.array(time_in)

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

# get user inputs
parser = argparse.ArgumentParser(description='deinterlace videos and adjust timestamps to match')
parser.add_argument('-d', '--data_path')
parser.add_argument('-s', '--save_path')
args = parser.parse_args()

if not args.save_path:
    args.save_path = args.data_path

csv_list = find('*BonsaiTS.csv', args.data_path)

for this_csv in csv_list:
    out_name = this_csv.replace(args.data_path, args.save_path)
    print('running on ' + this_csv)
    if 'EYE' in this_csv:
        csv_out = pd.DataFrame(open_time(this_csv, force_shift=True))
        csv_out.to_csv(out_name, index=False)
    elif 'EYE' not in this_csv:
        csv_out = pd.DataFrame(open_time(this_csv))
        csv_out.to_csv(out_name, index=False)

print('done with ' + str(len(csv_list)) + ' csv items')
