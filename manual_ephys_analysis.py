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

# get user arguemnts
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--rec_name', type=str)
    parser.add_argument('--unit', type=int)
    parser.add_argument('--fm', type=bool)
    parser.add_argument('--stim_type', type=str,choices=['None','gratings','sparse_noise','white_noise'])
    args = parser.parse_args()
    return args

def main(args):
    # organize a dictionary of inputs
    file_dict = find_files(args.data_path, args.rec_name, args.fm, args.unit, args.stim_type)
    # run the analysis
    run_ephys_analysis(file_dict)

if __name__ == '__main__':
    args = get_args()
    main(args)
