"""
FreelyMovingEphys/src/utils/save.py
Modified from https://codereview.stackexchange.com/a/121308
"""
import os
import PySimpleGUI as sg

import h5py
import numpy as np
import pandas as pd
import xarray as xr
from scipy.io import savemat


def read_DLC_data(path, multianimal=False):

    pts = pd.read_hdf(path)

    if multianimal is False:
        # Organize columns
        pts.columns = [' '.join(col[:][1:3]).strip() for col in pts.columns.values]
        pts = pts.rename(columns={pts.columns[n]: pts.columns[n].replace(' ', '_') for n in range(len(pts.columns))})

    elif multianimal is True:
        pts.columns = ['_'.join(col[:][1:]).strip() for col in pts.columns.values]

    return pts

def write_h5(filename, dic):
    """
    Saves a python dictionary or list, with items that are themselves either
    dictionaries or lists or (in the case of tree-leaves) numpy arrays
    or basic scalar types (int/float/str/bytes) in a recursive
    manner to an hdf5 file, with an intact hierarchy.
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
        if isinstance(item, (np.ndarray, np.int64, np.float64, int, float, str, bytes)):
            h5file[path + key] = item
        elif isinstance(item, dict) or isinstance(item,list):
            recursively_save_dict_contents_to_group(h5file, path + key + '/', item)
        else:
            raise ValueError('Cannot save %s type'%type(item))

def read_h5(filename, ASLIST=False):
    """
    Default: load a hdf5 file (saved with io_dict_to_hdf5.save function above) as a hierarchical
    python dictionary (as described in the doc_string of io_dict_to_hdf5.save).
    if ASLIST is True: then it loads as a list (on in the first layer) and gives error if key's are not convertible
    to integers. Unlike io_dict_to_hdf5.save, a mixed dictionary/list hierarchical version is not implemented currently
    for .load
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
            ans[key] = recursively_load_dict_contents_from_group(h5file, path + key + '/')
    return ans

def nc2mat():
    f = sg.popup_get_file('Choose .nc file.')
    data = xr.open_dataset(f)
    data_dict = dict(zip(list(data.REYE_ellipse_params['ellipse_params'].values), [data.REYE_ellipse_params.sel(ellipse_params=p).values for p in list(data.REYE_ellipse_params['ellipse_params'].values)]))
    save_name = os.path.join(os.path.split(f)[0], os.path.splitext(os.path.split(f)[1])[0])+'.mat'
    print('saving {}'.format(save_name))
    savemat(save_name, data_dict)