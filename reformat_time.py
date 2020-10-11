"""
reformat_time.py

format Bonsai timestamps from raw to seconds since midnight
good if time interpolation needs to be done prior to other analysis
this doesn't manage videos, only time -- see deinterlace_for_dlc.py

Oct. 09, 2020
"""

import argparse
import fnmatch
import numpy as np
import os
import pandas as pd
import dateutil
from datetime import datetime

from util.read_data import open_time, find

# get user inputs
parser = argparse.ArgumentParser(description='format timestamps and adjust to match deinterlaced videos as needed')
parser.add_argument('-d', '--data_path')
parser.add_argument('-s', '--save_path')
args = parser.parse_args()

if not args.save_path:
    args.save_path = args.data_path

csv_list = find('*BonsaiTS.csv', args.data_path)

for this_csv in csv_list:
    new_name = this_csv.replace(args.data_path, args.save_path)
    out_name = '.'.join([str(''.join(new_name.split('.')[:-1])) + 'formatted', new_name.split('.')[-1]])
    print('running on ' + this_csv)
    try:
        if 'EYE' in this_csv or 'WORLD' in this_csv:
            csv_out = pd.DataFrame(open_time(this_csv, force_shift=True))
            csv_out.to_csv(out_name, index=False, header=None)
        elif 'EYE' not in this_csv and 'WORLD' not in this_csv:
            csv_out = pd.DataFrame(open_time(this_csv))
            csv_out.to_csv(out_name, index=False, header=None)
    except pd.errors.EmptyDataError:
        pass

print('done with ' + str(len(csv_list)) + ' csv items')
