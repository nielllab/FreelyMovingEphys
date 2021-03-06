"""
manual_ephys_analysis.py

launch ephys analysis without GUI

Jan. 20, 2021
"""
# package imports
import subprocess, argparse
from glob import glob
# module imports
from project_analysis.ephys.analyze_ephys import find_files, run_ephys_analysis

def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'False', 'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'True', 'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')

# get user arguments
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, help='recording directory')
    parser.add_argument('--rec_name', type=str, help='recording name')
    parser.add_argument('--unit', type=int, help='ephys unit number (zero-ind) to highlight in figures')
    parser.add_argument('--fm', type=str_to_bool, nargs='?', const=True, default=False, help='bool, is this freely moving?')
    parser.add_argument('--stim_type', type=str, choices=['None','gratings','sparse_noise','white_noise','revchecker'], help='stimulus presented on screen. set as None if freely moving.')
    parser.add_argument('--write_mp4', type=str_to_bool, nargs='?', const=True, default=False, help='bool, want to save mp4 video?')
    args = parser.parse_args()
    return args

def main(args):
    # organize a dictionary of inputs
    file_dict = find_files(args.data_path, args.rec_name, args.fm, args.unit, args.stim_type, args.write_mp4)
    # run the analysis
    run_ephys_analysis(file_dict)

if __name__ == '__main__':
    args = get_args()
    main(args)
