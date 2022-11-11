"""File path tools.
"""

import os
import time
import fnmatch
import numpy as np

def choose_most_recent(paths):
    
    deltas = np.zeros(len(paths))
    for i, f in enumerate(paths):
        deltas[i] = time.time() - os.path.getmtime(f)
    ind = np.argmin(deltas)
    use_f = paths[ind]
    return use_f

def find(pattern, path, mr=False, exclude=[], none_possible=False):
    """ Glob for subdirectories.

    Parameters
    ----------
    pattern : str
        str with * for missing sections of characters
    path : str
        path to search, including subdirectories
    most_recent
    
    Returns
    -------
    result : list
        list of files matching pattern.
    """
    result = []

    # Walk though the path directory and files
    for root, _, files in os.walk(path):

        # Walk to the file in the directory
        for name in files:

            # If the file matches the filetype, append it to the list
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root ,name))

    if (result == []) and (none_possible is True):
        return None
    elif (result == []) and (none_possible is False):
        raise ValueError

    if exclude != []:
        result = [r for r in result if not any(x in r for x in exclude)]

    if not mr:
        # Return list of all matches
        return result

    elif mr:
        # Return a single string, the path of the most recently written
        # file which met criteria.
        return choose_most_recent(result)

def list_subdirs(rootdir):
    """ List subdirectories in a root directory.

    without keep_parent, the subdirectory itself is named
    with keep_parent, the subdirectory will be returned *including* its parent path
    """
    paths = []; names = []
    for item in os.scandir(rootdir):
        if os.path.isdir(item):
            if item.name[0]!='.':
                paths.append(item.path)
                names.append(item.name)

    return paths, names