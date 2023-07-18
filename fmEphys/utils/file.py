"""
fmEphys/utils/file.py

Functions for reading and writing files.

Functions
---------
read_DLC_data
    Read the .h5 file from DLC and return a pandas dataframe.
write_h5
    Write a dictionary to an .h5 file.
recursively_save_dict_contents_to_group
    Helper function for write_h5.
read_h5
    Read a .h5 file and return a dictionary.
recursively_load_dict_contents_from_group
    Helper function for read_h5.
write_group_h5
    Use pandas .to_hdf() method for multiple recordings.
get_group_h5_keys
    Get the keys of a .h5 file written with write_group_h5.
read_group_h5
    Read a .h5 file written with write_group_h5.


Written by DMM, 2021
"""


import h5py
import datetime
import numpy as np
import pandas as pd

import fmEphys as fme


def read_DLC_data(path, multianimal=False):
    """ Read the .h5 file from DLC and return a pandas dataframe.

    Parameters
    ----------
    path : str
        Path to the .h5 file.
    multianimal : bool
        If True, the columns will be named with the animal name in
        addition to the body part name. This is only needed for 
        videos analyzed using a multi-animal DLC network.

    Returns
    -------
    pts : pandas.DataFrame
        A pandas dataframe with the x, y, and likelihood coordinates
        of the body. The columns are named with the body part name.

    """

    pts = pd.read_hdf(path)

    if multianimal is False:

        # Organize columns
        pts.columns = [' '.join(col[:][1:3]).strip() for col in pts.columns.values]
        pts = pts.rename(columns={pts.columns[n]: pts.columns[n].replace(' ',                                               
                                            '_') for n in range(len(pts.columns))})

    elif multianimal is True:

        pts.columns = ['_'.join(col[:][1:]).strip() for col in pts.columns.values]

    return pts


def write_h5(filename, dic):
    """ Write a dictionary to an .h5 file.

    The dictionary can only contain values that are of the
    following types: dict, list, numpy.ndarray, or basic scalar
    types (int, float, str, bytes). The hierarchy of the dictionary
    is preserved in the .h5 file that is written. The keys of
    the dictionary can only be type str (not int).

    Parameters
    ----------
    filename : str
        Path to the .h5 file.
    dic : dict
        Dictionary to be saved.

    Notes
    -----
    Modified from https://codereview.stackexchange.com/a/121308

    """

    with h5py.File(filename, 'w') as h5file:

        recursively_save_dict_contents_to_group(h5file, '/', dic)


def recursively_save_dict_contents_to_group(h5file, path, dic):

    if isinstance(dic,dict):
        iterator = dic.items()

    elif isinstance(dic,list):
        iterator = enumerate(dic)

    else:
        ValueError('Cannot save %s type' % type(dic))

    for key, item in iterator:

        if isinstance(dic,list):
            key = str(key)
            
        if isinstance(item, (np.ndarray, np.int16, np.int64, np.float64, int, float, str, bytes, np.float32, np.int32)):
            
            try:
                h5file[path + key] = item
            
            except TypeError:
                if isinstance(item, np.ndarray) and (item.dtype == object):
                    recursively_save_dict_contents_to_group(h5file, path + key + '/', item.item())

        elif isinstance(item, dict) or isinstance(item, list):
            recursively_save_dict_contents_to_group(h5file, path + key + '/', item)

        elif isinstance(item, datetime.datetime):
             h5file[path + key] = fme.time2str(item)

        else:
            raise ValueError('Cannot save %s type'%type(item))


def read_h5(filename, ASLIST=False):
    """ Read an .h5 file in as a dictionary.

    Parameters
    ----------
    filename : str
        Path to the .h5 file.
    ASLIST : bool
        If True, the dictionary will be read in as a list (on the first
        layer). Keys must have been convertable to integers when the file
        was written.

    Notes
    -----
    Modified from https://codereview.stackexchange.com/a/121308

    """
    with h5py.File(filename, 'r') as h5file:
        out = recursively_load_dict_contents_from_group(h5file, '/')
        if ASLIST:
            outl = [None for l in range(len(out.keys()))]
            for key, item in out.items():
                outl[int(key)] = item
            out = outl
        return out


def recursively_load_dict_contents_from_group(h5file, path):
    
    ans = {}

    for key, item in h5file[path].items():

        if isinstance(item, h5py._hl.dataset.Dataset):
            ans[key] = item[()]

        elif isinstance(item, h5py._hl.group.Group):
            ans[key] = recursively_load_dict_contents_from_group(h5file,
                                                                 path + key + '/')

    return ans


def write_group_h5(df, savepath):
    """ Use pandas .to_hdf() method for multiple recordings.
    
    This is just a wrapper function to make sure it is
    handled in the same each time. The dataframe will be split
    by the column 'session'. Each unqiue session will be put
    into its own key of the h5 file.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe to be saved.
    savepath : str
        Path to the .h5 file.

    """

    split_key = 'session'

    df = fme.replace_xr_obj(df)

    for _, sname in enumerate(df[split_key].unique()):

        use_skey = '_'.join(sname.split('_')[:2])
        
        df[df[split_key]==sname].to_hdf(savepath, use_skey, mode='a')


def get_group_h5_keys(savepath):
    """ Get the keys of a group h5 file.

    This will list the keys (i.e. the session names) of an h5 file
    written by the function write_group_h5 (above). It does not need
    to read the entire file into memory to check these values.

    Parameters
    ----------
    savepath : str
        Path to the .h5 file.
    
    Returns
    -------
    keys : list
        List of keys (i.e. session names) in the h5 file.

    """

    with pd.HDFStore(savepath) as hdf:
        
        keys = [k.replace('/','') for k in hdf.keys()]

    return keys


def read_group_h5(path, keys=None):
    """ Read a group h5 file.

    This will read in a group h5 file written by the function
    write_group_h5 (above). It will read in all keys and stack
    them into a single dataframe. Alternatively, you can specify
    a list of keys to read in from the keys present, and only those
    recordings will be read into memory and stacked together.
    
    Parameters
    ----------
    path : str
        Path to the .h5 file.
    keys : list or str (optional).
        List of keys (i.e. session names) in the h5 file. If None,
        all keys will be read in.
    
    Returns
    -------
    df : pandas.DataFrame
        Dataframe containing all data from the h5 file.

    """

    if type(keys) == str:

        df = pd.read_hdf(path, keys)

        return df
    
    if keys is None:

        keys = get_group_h5_keys(path)

    dfs = []
    for k in sorted(keys):

        _df = pd.read_hdf(path, k) 
        dfs.append(_df)

    df = pd.concat(dfs)

    return df

