"""
___main__.py
"""
import argparse
from utils.config import str_to_bool
from batch_analysis.batch_analysis import main as run_batch_analysis

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, help='path for metadata .csv')
    parser.add_argument('--config', type=str, help='yaml config file')
    parser.add_argument('--log', type=str, help='save path for logger')
    parser.add_argument('--clear_dlc', type=str_to_bool, nargs='?', const=True, default=False, help='delete existing DLC .h5 files?')
    args = parser.parse_args()
    return args

args = get_args()
run_batch_analysis(args.csv, args.config, args.log, args.clear_dlc)