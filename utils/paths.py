"""
paths.py
"""
import os
import fnmatch

def find(pattern, path):
    """
    glob for subdirectories
    INPUTS
        pattern: str (with * for missing sectiosn of characters) like glob function
        path: dict to search, including subdirectories
    OUTPUTS
        result: list of files
    """
    result = [] # initialize the list as empty
    for root, _, files in os.walk(path): # walk though the path directory, and files
        for name in files:  # walk to the file in the directory
            if fnmatch.fnmatch(name,pattern):  # if the file matches the filetype append to list
                result.append(os.path.join(root,name))
    return result # return full list of file of a given type

def check_path(basepath, path):
    """
    check if path exists, if not then create directory
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
    """
    list subdirectories in a root directory
    """
    dirnames = []
    for _, dirs, _ in os.walk(root_dir):
        for rec_dir in dirs:
            dirnames.append(rec_dir)
    return dirnames