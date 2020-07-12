#####################################################################################
"""
find.py

Creates a function to find file of a given type in a path and recursively searches subfolders

last modified: June 19, 2020
"""
#####################################################################################

import os
import fnmatch

def find(pattern, path):
    result = [] # initialize the list as empty
    for root, dirs, files in os.walk(path): # walk though the path directory, and files
        for name in files:  # walk to the file in the directory
            if fnmatch.fnmatch(name,pattern):  # if the file matches the filetype append to list
                result.append(os.path.join(root,name))
    return result # return full list of file of a given type