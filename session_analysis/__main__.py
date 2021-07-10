"""
__main__.py
"""
import argparse
import tkinter as tk
from tkinter import filedialog

from utils.config import str_to_bool
from session_analysis.session_analysis import main as run_session_analysis

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--clear_dlc', type=str_to_bool, nargs='?', const=True, default=False)
    args = parser.parse_args()
    return args

args = get_args()
# if no path was given as an argument, open a dialog box
if args.config == None:
    root = tk.Tk()
    root.withdraw()
    config = filedialog.askdirectory(title='Select the animal directory')
else:
    config = args.config
run_session_analysis(config, args.clear_dlc)