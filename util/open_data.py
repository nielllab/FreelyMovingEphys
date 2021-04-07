"""
open_data.py

functions for opening files
"""
import pandas as pd
import numpy as np
import xarray as xr
from glob import glob
import os
import fnmatch
import dateutil
import cv2
from tqdm import tqdm
from datetime import datetime
import time
import argparse

def open_h5(path):
    """
    read in .h5 DLC files and manage column names
    INPUTS:
        path -- filepath to .h5 file outputs by DLC
    OUTPUTS:
        pts -- pandas dataframe of points
        pt_loc_names -- list of column names in pts
    """
    try:
        # read the .hf file when there is no key
        pts = pd.read_hdf(path)
    except ValueError:
        # read in .h5 file when there is a key set in corral_files.py
        pts = pd.read_hdf(path, key='data')
    # organize columns
    pts.columns = [' '.join(col[:][1:3]).strip() for col in pts.columns.values]
    pts = pts.rename(columns={pts.columns[n]: pts.columns[n].replace(' ', '_') for n in range(len(pts.columns))})
    pt_loc_names = pts.columns.values

    return pts, pt_loc_names

def open_ma_h5(path):
    """
    open .h5 file of a multianimal DLC project
    INPUTS:
        path -- filepath to .h5 file outputs by DLC
    OUTPUTS:
        pts -- pandas dataframe of points
    """
    pts = pd.read_hdf(path)
    # flatten columns from MultiIndex 
    pts.columns = ['_'.join(col[:][1:]).strip() for col in pts.columns.values]

    return pts
    