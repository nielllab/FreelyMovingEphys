"""
aux_funcs.py

auxiliary functions

Dec. 02, 2020
"""
# package imports
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

# calculates xcorr ignoring NaNs without altering timing
# adapted from /niell-lab-analysis/freely moving/nanxcorr.m
def nanxcorr(x, y, maxlag=25):
    lags = range(-maxlag, maxlag)
    cc = []
    for i in range(0,len(lags)):
        # shift data
        yshift = np.roll(y, lags[i])
        # get index where values are usable in both x and yshift
        use = ~pd.isnull(x + yshift)
        # some restructuring
        x_arr = np.asarray(x, dtype=object); yshift_arr = np.asarray(yshift, dtype=object)
        x_use = x_arr[use]; yshift_use = yshift_arr[use]
        # normalize
        x_use = (x_use - np.mean(x_use)) / (np.std(x_use) * len(x_use))
        yshift_use = (yshift_use - np.mean(yshift_use)) / np.std(yshift_use)
        # get correlation
        cc.append(np.correlate(x_use, yshift_use))
    cc_out = np.hstack(np.stack(cc))
    return cc_out, lags