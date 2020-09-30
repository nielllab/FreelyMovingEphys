"""
pickle_to_h5.py

convert any DeepLabcut .pickle outputs to .h5 files

Sept. 29, 2020
"""

# package imports
import deeplabcut
import argparse

# module imports
from util.read_data import find

parser = argparse.ArgumentParser(description='deinterlace videos and adjust timestamps to match')
parser.add_argument('-c', '--dlc_config',help='DeepLabCut config .yaml path')
parser.add_argument('-d', '--data_path', help='parent directory of pickle files')
args = parser.parse_args()

pickle_list = find('*TOP*full.pickle', args.data_path)
for vid in pickle_list:
    print('converting to pickle video at ' + vid)
    deeplabcut.convert_raw_tracks_to_h5(args.dlc_config, vid)
print('done converting ' + len(pickle_list) + ' pickles')