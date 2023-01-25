"""
FreelyMovingEphys/src/utils/save.py
"""

import h5py
import numpy as np
import pandas as pd

import fmEphys

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
            ans[key] = recursively_load_dict_contents_from_group(h5file, path + key + '/')
    return ans


def write_group_h5(df, savepath, split_key='session'):
    """
    use pandas .to_hdf() method for multiple recordings
    basiclly just a wrapper function to make sure it's handled in the same each time

    split a full dataframe by the column 'session'. each session will be put into its own key
    """

    df = fmEphys.replace_xr_obj(df)

    for _, sname in enumerate(df[split_key].unique()):
        use_skey = '_'.join(sname.split('_')[:2])
        
        df[df[split_key]==sname].to_hdf(savepath, use_skey, mode='a')

def get_group_h5_keys(savepath):

    with pd.HDFStore(savepath) as hdf:
        keys = [k.replace('/','') for k in hdf.keys()]

    return keys

def read_group_h5(savepath, keys=None):
    """
    if keys is None, it will read in all keys and stack them
    """

    if type(keys) == str:
        df = pd.read_hdf(savepath, k)
        return df
    
    if keys is None:
        keys = get_group_h5_keys(savepath)

    dfs = []
    for k in sorted(keys):
        _df = pd.read_hdf(savepath, k) 
        dfs.append(_df)

    df = pd.concat(dfs)
    return df
