"""
paths.py
"""
import os
import fnmatch

def find(pattern, path):
    """ Glob for subdirectories

    Parameters:
    pattern (str): str with * for missing sections of characters
    path (str): path to search, including subdirectories
    
    Returns:
    result (list): list of files matching pattern
    """
    result = [] # initialize the list as empty
    for root, _, files in os.walk(path): # walk though the path directory, and files
        for name in files:  # walk to the file in the directory
            if fnmatch.fnmatch(name,pattern):  # if the file matches the filetype append to list
                result.append(os.path.join(root,name))
    return result # return full list of file of a given type

def check_path(basepath, path):
    """ Check if path exists, if not then create directory
    """
    if path in basepath:
        return basepath
    elif not os.path.exists(os.path.join(basepath, path)):
        os.makedirs(os.path.join(basepath, path))
        print('Added Directory:'+ os.path.join(basepath, path))
        return os.path.join(basepath, path)
    else:
        return os.path.join(basepath, path)

def list_subdirs(root_dir):
    """ List subdirectories in a root directory
    """
    dirnames = []
    for _, dirs, _ in os.walk(root_dir):
        for rec_dir in dirs:
            dirnames.append(rec_dir)
    return dirnames