"""
fmEphys/utils/path.py

Filepath helper functions.

Functions
---------
choose_most_recent
    Choose the most recent file in a list.
up_dir
    Step up directories.
find
    Glob for subdirectories.
filter_file_search
    Apply criteria to a list of paths.
check_subdirs
    Create subdirectory if it does not exist.
list_subdirs
    List subdirectories in a root directory.
auto_recording_name
    Parse file path to infer recording name.


Written by DMM, 2021
"""


import os
import time
import fnmatch
import numpy as np


def choose_most_recent(paths):
    """ Choose the most recent file in a list.

    Parameters
    ----------
    paths : list
        List of file paths.
    
    Returns
    -------
    use_f : str
        Path for the file in `paths` which was written
        most recently.
    """
    
    deltas = np.zeros(len(paths))

    for i, f in enumerate(paths):
        deltas[i] = time.time() - os.path.getmtime(f)

    ind = np.argmin(deltas)
    use_f = paths[ind]

    return use_f


def up_dir(f, num=1):
    """ Step up directories.

    Step up multiple directories from an input path
    by splitting the path `num` times and keeping only
    the path head each time.
    
    Parameters
    ----------
    f : str
        Path for a file or directory
    num : int
        Number of directories to step up.

    Returns
    -------
    dir : str
        Directory that exists `num` levels up
        from `f`.
    """

    dir = f
    for n in range(num):
        dir = os.path.split(dir)[0]

    return dir


def find(pattern, path, MR=False):
    """ Glob for subdirectories.

    Parameters
    ----------
    pattern : str
        Pattern to search for within the given `path`. Use
        a '*' for missing sections of characters.
    path : str
        Path to search, including subdirectories.
    MR : bool
        If only the most recent file should be returned. When
        MR is True, `_ret` will be returned as a str and be
        the path for the file matching the pattern which was
        written most recently. Other files which match the
        pattern but are not the most recent will be ignored.
        When MR is False, `_ret` is returned as a list of all
        files that matched the pattern. Default for MR is False.
    
    Returns
    -------
    _ret : list or str
        When MR is False, `_ret` is a list of files matching
        pattern. Otherwise when MR is True, `_ret` is a str
        containing only the path to the file which matched the
        pattern and was written most recently.

    """

    result = []
    
    # Walk though the path directory, and files
    for root, _, files in os.walk(path): 
        
        # Walk to the file in the directory
        for name in files:
            
            # if the file matches the filetype append to list
            if fnmatch.fnmatch(name,pattern):
                result.append(os.path.join(root,name))
    
    if MR is True:
        # Return only the most recent result
        _ret = choose_most_recent(result)
        
    elif MR is False:
        # Or return the fulll list of items matching the pattern
        _ret = result

    return _ret


def filter_file_search(files, keep=[], toss=[], MR=False):
    """ Apply criteria to list of paths.

    Parameters
    ----------
    files : list
        List of paths that will be filtered using criteria.
    keep : list
        Optional. When provided, items in `files` that do
        not contain ALL of the str in `keep` will be removed.
    toss : list
        Optional. When provided, items in `files` that contain
        ANY of the str in `toss` will be removed.
    MR : bool
        If only the most recent file should be returned. When
        MR is True, `_ret` will be returned as a str and be
        the path for the path passing criteria which was written
        most recently. Other files which pass criteria but are
        not the most recent will be ignored. When MR is False,
        `_ret` is returned as a list of all items that passed
        criteria. Default for `MR` is False.

    Returns
    -------
    _ret : list or str
        Either the list of paths or the path for the most recently
        written item which passed criteria (determined by the input
        parameter `MR`).
    
    """

    # Remove files that do not contain `keep` strings
    if keep != []:
        for k in keep:
            files = [f for f in files if k in f]
    
    # Remove files that contain `toss` strings
    if toss != []:
        for t in toss:
            files = [f for f in files if t not in f]

    # Return the remaining files. If the user wants to return the file
    # written to disk most recently, just return that one file. Otherwise,
    # return what remains of the list.
    if MR is True:
        _ret = choose_most_recent(files)

    elif MR is False:
        _ret = files

    return _ret


def check_subdir(basepath, path):
    """ Create subdirectory if it does not exist.

    Parameters
    ----------
    basepath : str
        Directory in which the directory is expected.
    path : str
        Name of expected subdirectory.

    Returns
    -------
    _res : str
        Name of directory found or created.

    """

    if path in basepath:
        _res = basepath
    
    elif not os.path.exists(os.path.join(basepath, path)):
        
        # Make the directory
        os.makedirs(os.path.join(basepath, path))

        print('Added Directory:'+ os.path.join(basepath, path))
        _res = os.path.join(basepath, path)
    
    else:
        _res = os.path.join(basepath, path)
    
    return _res


def list_subdirs(rootdir, givepath=False):
    """ List subdirectories in a root directory.

    Parameters
    ----------
    rootdir : str

    """

    paths = []
    names = []
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
    """ Parse file path to infer recording name.

    Parameters
    ----------
    recording_path : str
        Path to the directory of one recording. Must be
        stimulus-specific. e.g. D:/path/to/animal/hf1_wn
    
    Returns
    -------
    name : str
        Name of recording.
        e.g., 010101_animal_Rig2_control_hf1_wn

    """

    # Ignore files that contain any of these str
    b_items = ['plot','IR','rep11','betafpv','side_gaze','._']

    # Get the list of files
    a = find('*.avi', recording_path)
    c = [i for i in a if all(b not in i for b in b_items)][0]
    d = os.path.splitext(os.path.split(c)[1])[0].split('_')[:-1]
    name = '_'.join(d)

    # Previously, done on a single line...
    # recording_name = '_'.join(os.path.splitext(os.path.split(
    # [i for i in find('*.avi', recording_path) if all(bad not
    # in i for bad in ['plot','IR','rep11','betafpv','side_gaze'
    # '._'])][0])[1])[0].split('_')[:-1])

    return name

