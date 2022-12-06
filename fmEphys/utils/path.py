"""
FreelyMovingEphys/src/utils/path.py
"""
import os
import fnmatch
import numpy as np
from time import time

def choose_most_recent(paths):
    
    deltas = np.zeros(len(paths))
    for i, f in enumerate(paths):
        deltas[i] = time.time() - os.path.getmtime(f)
    ind = np.argmin(deltas)
    use_f = paths[ind]
    return use_f

def find(pattern, path):
    """ Glob for subdirectories.

    Parameters
    --------
    pattern : str
        str with * for missing sections of characters
    path : str
        path to search, including subdirectories
    
    Returns
    --------
    result : list
        list of files matching pattern.
    """
    result = [] # initialize the list as empty
    for root, _, files in os.walk(path): # walk though the path directory, and files
        for name in files:  # walk to the file in the directory
            if fnmatch.fnmatch(name,pattern):  # if the file matches the filetype append to list
                result.append(os.path.join(root,name))
    return result # return full list of file of a given type

def check_subdir(basepath, path):
    """ Check if subdirectory exists, and create it if it does not exist.

    Parameters
    --------
    basepath : str
        Directory in which the directory is expected.
    path : str
        Name of subdirectory expected.

    Returns
    --------
    Name of directory found or created.
    """
    if path in basepath:
        return basepath
    elif not os.path.exists(os.path.join(basepath, path)):
        os.makedirs(os.path.join(basepath, path))
        print('Added Directory:'+ os.path.join(basepath, path))
        return os.path.join(basepath, path)
    else:
        return os.path.join(basepath, path)

def list_subdirs(rootdir, givepath=False):
    """ List subdirectories in a root directory.
    """
    paths = []; names = []
    for item in os.scandir(rootdir):
        if os.path.isdir(item):
            if item.name[0]!='.':
                paths.append(item.path)
                names.append(item.name)
    if givepath:
        return paths
    elif not givepath:
        return names

def auto_recording_name(recording_path):
    """ Parse file names in recording path to build name of the recording.

    Parameters
    --------
    recording_path : str
        Path to the directory of one recording. Must be stimulus-specific.
        e.g. D:/path/to/animal/hf1_wn
    
    Returns
    recording_name : str
        Name of recording from a specific stimulus.
        e.g. 010101_animal_Rig2_control_hf1_wn
    """
    recording_name = '_'.join(os.path.splitext(os.path.split([i for i in find('*.avi', recording_path) if all(bad not in i for bad in ['plot','IR','rep11','betafpv','side_gaze','._'])][0])[1])[0].split('_')[:-1])
    return recording_name