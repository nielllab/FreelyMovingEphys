import os
import argparse
import pandas as pd
import numpy as np
import scipy.interpolate

import fmEphys

def set_timebase(sparse_time, fixedinter_time, speed, seek_win=0.030):
    """Bin movement to constant timebase.
    
    Treadmill data is aquired without a constant sample rate; there is a maximum
    aquisition rate, but stationary periods are not recorded. This function bins
    speed data into a constant timebase, filling zeros for steps in time when a
    period without running left no samples recorded.
    
    Parameters
    ----------
    sparse_time : np.array
        Timestamps (in seconds) for samples do not exist when there was no change in data.
    fixedinter_time : np.array
        Timestamps (in seconds) with a constant step size, where start and end match sparse_time.
        e.g. an array created with `np.arange(t0, t_end, samprate)

    speed : np.array
        Array of values that match the timebase of sparse_time.
    
    Returns
    -------
    data_out : np.array
        data with constant time step size and with zeros filled in where both data and
        sparse_time previously had no values
    """
    data_out = np.zeros(len(fixedinter_time))
    for t in sparse_time:
        ind = np.searchsorted(fixedinter_time, t)
        if ind < len(fixedinter_time):
            data_out[ind] = (speed[ind] if t >= (fixedinter_time[ind]-seek_win) and t <= fixedinter_time[ind] else 0)
    return data_out

def preprocess_treadmill(cfg, csv_path=None):
    """ Track the movement of the ball for headfixed recordings.

    samprate
    # float in seconds, window in which a previous timestamp in
    # sparse_time must fall, otherwise a zero will be filled in
    """

    if csv_path is None:
        csv_path = fmEphys.find('*BALLMOUSE_BonsaiTS_X_Y.csv'.format(cfg['rfname']),
                                     cfg['rpath'], mr=True)

    # read in one csv file with timestamps, x position, and y position in three columns
    csv_data = pd.read_csv(csv_path)

    # from this, we can get the timestamps, as seconds since midnight before the recording
    time = fmEphys.format(csv_data['Timestamp.TimeOfDay'])

    # convert center-subtracted pixels into cm
    x_pos = (csv_data['Value.X'] - cfg['tdml_x']) / cfg['tdml_pxl2cm']
    y_pos = (csv_data['Value.Y'] - cfg['tdml_y']) / cfg['tdml_pxl2cm']

    # set up new time base
    t0 = time[0]
    t_end = time[-1]
    arange_time = np.arange(t0, t_end, cfg['tdml_resamprate'])

    # interpolation of xpos, ypos 
    xinterp = scipy.interpolate.interp1d(time, x_pos, bounds_error=False, kind='nearest')(arange_time)
    yinterp = scipy.interpolate.interp1d(time, y_pos, bounds_error=False, kind='nearest')(arange_time)

    # if no timestamp within 30ms, set interpolated val to 0
    full_x = set_timebase(time, arange_time, xinterp)
    full_y = set_timebase(time, arange_time, yinterp)

    # cm per second
    xpersec = full_x[:-1] / np.diff(arange_time)
    ypersec = full_y[:-1] / np.diff(arange_time)

    # speed
    speed = fmEphys.convfilt(np.sqrt(xpersec**2 + ypersec**2), 10)

    # collect all data
    savedata = {
        'time': time,
        'cm_x': full_x,
        'cm_y': full_y,
        'x_persec': xpersec,
        'y_persec': ypersec,
        'speed_cmpersec': speed
    }

    savepath = os.path.join(cfg['rpath'], '{}_treadmill.h5'.format(cfg['rfname']))
    fmEphys.write_h5(savepath, savedata)

    return savedata

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str, default=None)
    args = parser.parse_args()

    cfg = fmEphys.get_cfg()

    preprocess_treadmill(cfg, csv_path=args.csv_path)