"""
__main__.py

launch GUI to run minimal analysis needed to get receptive fields in the worldcam
run on white noise stimulus after spike sorting is complete
"""
from project_analysis.map_receptive_fields.initial_wn_analysis import quick_whitenoise_analysis
import argparse, json, sys, os, subprocess, shutil
import tkinter as tk
from tkinter import filedialog
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ch', type=int, default=64)
    args = parser.parse_args()
    return args

def main(args):
    root = tk.Tk()
    root.withdraw()
    this_subject = filedialog.askdirectory(title='Select the hf whitenoise dir of subject:')
    quick_whitenoise_analysis(this_subject, args.ch)

if __name__ == '__main__':
    args = get_args()
    main(args)