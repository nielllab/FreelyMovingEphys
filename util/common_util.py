"""
FreelyMovingEphys common data-handling utilities
commmon_util.py

Last modified July 29, 2020
"""

# package imports
import xarray as xr
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import cv2
from scipy import signal
from scipy.optimize import curve_fit
import scipy.stats

# get the mean confidence interval
def find_ci(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h, m-h, m+h

# optimization of parameters of the sigmoid function
# adapted from http://www.mathworks.com/matlabcentral/fileexchange/42641-sigmoid-logistic-curve-fit
def sigm_fit(x, y, fixed_params, init_params):

    auto_init_params = [np.quantile(y, 0.05), np.quantile(y, 0.95), np.nan, 1]
    if sum(y == np.quantile(y, 0.5)) == 0:
        temp = x[y == np.quantile(y[1:], 0.5)]
    else:
        temp = x[y == np.quantile(y, 0.5)]
    auto_init_params[2] = temp[0]

    free_param_count = 0
    bool_vec = [np.nan, np.nan, np.nan, np.nan]
    for i in range(0,4):
        if np.isnan(fixed_params[i]):
            free_param_count = free_param_count + 1
            bool_vec[i] = 1
        else:
            bool_vec[i] = 0
    # non-linear regression
    popt, pcov = curve_fit(x, y, init_params[bool_vec==1])
    # confidence interval of the parameters
    ypred, delta, ypred_lowerci, ypred_upperci = find_ci(popt)
    out = [popt, ypred, delta, ypred_lowerci, ypred_upperci]

    return out

# calculates xcorr ignoring NaNs without altering timing
# adapted from /niell-lab-analysis/freely moving/nanxcorr.m
def nanxcorr(x,y,maxlag=25,normalization='coeff'):
    if normalization == 'zero':
        x = x - np.nanmean(x)
        y = y - np.nanmean(y)
    lags = range(-maxlag, maxlag)
    for i in range(0,len(lags)):
        yshift = np.roll(y, lags[i])
        use = ~np.isnan(x + yshift)
        cc[i,0] = np.correlate(x[use], yshift,'full')

    return cc, lags
