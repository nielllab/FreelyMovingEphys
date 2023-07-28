"""
fmEphys/utils/time.py

Timestamp helper functions.

Functions
---------
fmt_time
    String formatting for timestamps.
interp_time
    Interpolate timestamps for deinterlaced video.
read_time
    Read timestamps from a .csv file.
fmt_now
    Format today's date and time.


Written by DMM, 2021
"""


import sys
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.interpolate import interp1d


def fmt_time(s):
    """ String formatting for timestamps.

    Input must be a pd.Series, so a filepath to a .csv
    should be used with time.read() not time.format()
 
    Parameters
    ----------
    s : pd.Series
        Timestamps as a single column. No header, so that the first
        value is the first timestamp. Expected to be formated as
        hours : minutes : seconds . microsecond
    
    Returns
    -------
    t_out : np.array
        Number of seconds that have passed since the previous midnight,
        with microescond precision, e.g. 700.000000
    """

    t_out = []
    fmt = '%H:%M:%S.%f'

    if s.dtype != np.float64:
        for t_in in s:
            t = str(t_in).strip()

            # If the timestamp has an unexpected precision
            try:
                _ = datetime.strptime(t, fmt)
            except ValueError as v:
                ulr = len(v.args[0].partition('unconverted data remains: ')[2])
                if ulr:
                    t = t[:-ulr]
            
            try:
                t_out.append((datetime.strptime(t, fmt) - datetime.strptime('00:00:00.000000', fmt)).total_seconds())
            except ValueError:
                t_out.append(np.nan)

        t_out = np.array(t_out)
    
    else:
        t_out = s.values

    return t_out


def interp_time(t_in, use_medstep=False):
    """ Interpolate timestamps for deinterlaced video.
    
    Parameters
    --------
    camT : np.array
        Camera timestamps aquired at 30Hz
    use_medstep : bool
        When True, the median diff(camT) will be used as the timestep
        in interpolation. If False, the timestep between each frame will
        be used instead.

    Returns
    --------
    camT_out : np.array
        Timestamps of camera interpolated so that there are twice the number
        of timestamps in the array. Each timestamp in camT will be replaced by
        two, set equal distances from the original.

    """
    t_out = np.zeros(np.size(t_in, 0)*2)

    medstep = np.nanmedian(np.diff(t_in, axis=0))

    # Shift each deinterlaced frame by 0.5 frame periods forward/backwards

    if use_medstep:
        # Use the median timestep
        t_out[::2] = t_in - 0.25 * medstep
        t_out[1::2] = t_in + 0.25 * medstep

    elif not use_medstep:
        # Use the time step for each specific frame (this is preferred)
        steps = np.diff(t_in, axis=0, append=t_in[-1]+medstep)
        t_out[::2] = t_in
        t_out[1::2] = t_in + 0.5 * steps

    return t_out


def read_time(path, dlen=None, shift=False):
    """ Read timestamps from a .csv file.

    Parameters
    --------
    position_data_length : None or int
        Number of timesteps in data from deeplabcut. This is used to
        determine whether or not the number of timestamps is too short
        for the number of video frames.
        Eyecam and Worldcam will have half the number of timestamps as
        the number of frames, since they are aquired as an interlaced
        video and deinterlaced in analysis. To fix this, timestamps need
        to be interpolated.

    """
    # Read in times from the .csv file and format them as an array. Timestamps
    # should be one column. The first cell of the .csv (i.e. header) can either
    # be the first timestamp or the value `0` as a float or int.
    s = pd.read_csv(path, encoding='utf-8', engine='c', header=None).squeeze()

    # Deal with header.
    if s[0] == 0:
        s = s[1:]

    # Format time as string of seconds and convert it into a numpy array.
    t_out = fmt_time(s)

    # Compare timestamp length to its data (e.g. video) to see
    # if there is a 2x difference in length.
    # Or, shift without checking if `shift` is True.
    if ((dlen is not None) and (dlen > len(t_out))) or (shift is True):
        t_out = interp_time(t_out)

    return t_out


def fmt_now(c=False):
    """Format today's date and time.

    Returns
    -------
    str_date : str
        Current date
        e.g. Aug. 30 2022 -> 083022
    str_time : str
        Current hours and minutes
        e.g. 10:15:00 am -> 10h-15m-00s
        Will be 24-hour time

    """
    str_date = datetime.today().strftime('%m%d%y')

    h = datetime.today().strftime('%H')
    m = datetime.today().strftime('%M')
    s = datetime.today().strftime('%S')
    str_time = '{}h-{}m-{}s'.format(h,m,s)

    if c==True:
        out = '{}_{}'.format(str_date, str_time)
        return out

    return str_date, str_time



def time2str(time_array):
    """ Convert datetime to string.

    The datetime values cannot be written into a hdf5
    file, so we convert them to strings before writing.

    Parameters
    ----------
    time_array : np.array, datetime.datetime
        If np.array with the shape (n,) where n is the
        number of samples in the recording. If datetime,
        the value will be converted to a single string.

    Returns
    -------
    out : str, list
        If time_array was a datetime, the returned value
        is a single string. Otherwise, it will be a list
        of strings with the same length as the input array.
        Str timestamps are use the format '%Y-%m-%d-%H-%M-%S-%f'.

    """

    fmt = '%Y-%m-%d-%H-%M-%S-%f'

    if type(time_array) == datetime:
        return time_array.strftime(fmt)


    out = []

    for t in time_array:
        tstr = t.strftime(fmt)
        out.append(tstr)

    return out


def str2time(input_str):
    """ Convert string to datetime.

    Need to convert the strings back to datetime objects
    after they are read back in from the hdf5 file.

    Parameters
    ----------
    input_str : str, byte, list, dict
        If str or byte, the value will be converted to a single
        datetime object. If list or dict, the values will be
        converted to an array of datetime objects. Datetime
        objects are returned with the format '%Y-%m-%d-%H-%M-%S-%f'.

    Returns
    -------
    out : datetime.datetime, np.array
        If input_str was a str or byte, the returned value is a single
        datetime object. Otherwise, it will be a np.array of datetime
        objects with the same length as the list or dict given for
        str_list.
    
    """

    fmt = '%Y-%m-%d-%H-%M-%S-%f'
    out = np.zeros(len(input_str), dtype=datetime)

    if type(input_str)==str:
        out = datetime.strptime(input_str, fmt)

    elif type(input_str)=='byte':
        out = datetime.strptime(input_str.decode('utf-8'), fmt)

    elif type(input_str)==list:

        for i,t in enumerate(input_str):
            out[i] = datetime.strptime(t, fmt)

    elif type(input_str)==dict:

        for k,v in input_str.items():

            out[int(k)] = datetime.strptime(v.decode('utf-8'), fmt)

    return out


def time2float(timearr, rel=None):
    """ Convert datetime to float.

    Parameters
    ----------
    timearr : np.array
        Array of datetime objects.
    rel : datetime.datetime, optional
        If not None, the returned array will be relative
        to this time. The default is None, in which case the
        returned float values will be relative to the first
        time in timearr (i.e. start at 0 sec).
    
    Returns
    -------
    out : np.array
        Array of float values representing the time in seconds.
    
    """
    if rel is None:
        return [t.total_seconds() for t in (timearr - timearr[0])]
    elif rel is not None:
        if type(rel)==list or type(rel)==np.ndarray:
            rel = rel[0]
            rel = datetime(year=rel.year, month=rel.month, day=rel.day)
        return [t.total_seconds() for t in timearr - rel]
    

def interpT(x, xT, toT):
    """ Interpolate timestamps.
    
    Parameters
    ----------
    x : np.array
        Array of values to interpolate.
    xT : np.array
        Array of datetime objects corresponding to x.
    toT : np.array
        Array of datetime objects to interpolate to.

    Returns
    -------
    out : np.array
        Array of interpolated values.

    """

    # Convert timestamps to float values.
    if (type(xT[0]) == datetime) and (type(toT[0]) == datetime):
        xT = time2float(xT)
        toT = time2float(toT)

    out = interp1d(xT, x,
                   bounds_error=False)(toT)
    
    return out


if __name__ == '__main__':
    globals()[sys.argv[1]]()

