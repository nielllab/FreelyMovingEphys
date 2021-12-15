"""
FreelyMovingEphys/core/utils/path.py
"""
import os, fnmatch

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

def list_subdirs(root_dir, name_only=False):
    """ List subdirectories in a root directory.

    Parameters
    --------
    root_dir : str
        Root direction that subdirectories are in.
    name_only : bool
        When True, returns the name of subdirectories. When False,
        returns the full path to that directory including the
        name.

    Returns
    --------
    dirnames : list
        List of directories as strings.
    """
    dirnames = []

    if not name_only:
        for _, dirs, _ in os.walk(root_dir):
            for rec_dir in dirs:
                dirnames.append(rec_dir)
    elif name_only:
        for _, _, filenames in os.walk(root_dir):
            for name in filenames:
                dirnames.append(name)
            break
    return dirnames

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