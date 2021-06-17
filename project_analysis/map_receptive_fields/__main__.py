"""
__main__.py

launch GUI to run minimal analysis needed to get receptive fields in the worldcam
run on white noise stimulus after spike sorting is complete
"""
from project_analysis.map_receptive_fields.initial_wn_analysis import quick_whitenoise_analysis
import argparse, json, sys, os, subprocess, shutil
import tkinter as tk
from tkinter import filedialog

root = tk.Tk()
root.withdraw()
this_subject = filedialog.askdirectory(title='Select the hf whitenoise dir of subject:')
quick_whitenoise_analysis(this_subject)
