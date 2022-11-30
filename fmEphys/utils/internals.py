

import os
import json

import fmEphys

def fill_rec_details(cfg):

    if ('_rpath' not in cfg.keys()):
        cfg['_rpath'] = None
    if ('_rabrv' not in cfg.keys()):
        cfg['_rabrv'] = None

    if (cfg['_rpath'] is None) and (cfg['_rabrv'] is not None):
        cfg['_rabrv'] = os.path.join(cfg['apath'], cfg['_rabrv'])

    if ('_rname' not in cfg.keys()) or (cfg['_rname'] is None):
        cfg['_rname'] = fmEphys.get_rec_name(cfg['_rpath'])

    return cfg

def get_cfg(cfg_in=None, opts_in=None):
    """
    cfg_in always takes priority over opts_in, if there are any keys already
    defined in cfg_in.
    """

    if type(cfg_in)==str:
        # Read in the dict of default options
        cfg = fmEphys.read_yaml(cfg_in)
    elif type(cfg_in)==dict:
        # Already a dict
        cfg = cfg_in
    elif cfg is None:
        cfg = {}

    # If a path to options.yml was not specified, use the default.
    if opts_in is None:
        head, _ = os.path.split(os.path.abspath(__file__))
        opts_in = os.path.join(head, 'options.yml')
    opts = fmEphys.read_yaml(opts_in)

    # Merge default cfg options with default internals.
    # cfg (listed AFTER opts) will be prioritized
    # so that values in opts that are already defined in cfg will be ignored
    merge_cfg = {**opts, **cfg}

    return merge_cfg

def assign_stim_name(a):
    """
    return in pattern:
        full_stim_description, abbrev
    
    """

    # Free movement
    if all([x in a for x in ['fm','light']]) or ('fm' in a and 'dark' not in a):
        return 'fm_light', 'FmLt'
    if all([x in a for x in ['fm','dark']]):
        return 'fm_dark', 'FmDk'

    # White noise
    if ('wn' in a):
        return 'hf_white_noise', 'Wn'

    # Gratings
    if ('grat' in a and 'static' not in a):
        return 'hf_drift_gratings', 'GtDf'

    if all([x in a for x in ['grat','static','500ms']]) and ('ISI' not in a):
        return 'hf_static_gratings', 'GtSt'

    if all([x in a for x in ['grat','static','500ms','ISI']]):
        return 'hf_staticISI_gratings', 'GtSI'

    # Flashed sparse noise
    if all([x in a for x in ['sp','noiseflash']]) and ('500ms' not in a):
        return 'hf_250ms_sparse_noise'
    if all([x in a for x in ['sp','noiseflash','500ms']]):
        return 'hf_500ms_sparse_noise'
    if all([x in a for x in ['sp','noiseflash','500ms']]) and ('ISI' not in a):
        return 'hf_500ms_sparse_noise'

    # Flashed reversing checkerboard
    if ('revchecker' in a and 'rand' not in a and '500ms' not in a):
        return 'hf_1s_checker'
    if ('revchecker' in a and 'rand' not in a and '500ms' in a):
        return 'hf_500ms_checker'
    if all([x in a for x in ['revchecker','rand']]):
        return 'hf_rand_checker'

    # Replay worldcam while head-fixed
    if all([x in a for x in ['hf','worldcam']]) and ('10s' not in a):
        return 'hf_long_world'
    if all([x in a for x in ['hf','worldcam','10s']]):
        return 'hf_10s_world'

def get_path(cfg, key, pathdict=None):

    if pathdict is None:
        # The expected file path
        json_path = os.path.join(cfg['_rpath'], 'recpaths.json')
        
        if os.path.isfile(json_path):
            # Read it if it already exists
            pathdict = json.load(json_path)
        elif not os.path.isfile(json_path):
            # Make a blank recpaths.json file if one does not already exist
            default_json = os.path.join(os.getcwd(), 'recpaths.json')
            pathdict = json.load(default_json)
            json.dump(pathdict, json_path)
        
    if pathdict['rpath'] is None:
        pathdict['rpath'] = cfg['_rpath']

    if pathdict[key] is not None:
        name = pathdict[key]['name']
        path = pathdict[key]['name']
        name = pathdict[key]['name']
    
    if pathdict[key] is None:

    if pathdict[key] is None:
        print('Key does not exist in ')

    

    '_reye_preprocessing.h5'

    cfg['recpaths'] = ['world_h5']

def set_cfg_paths(cfg):

    session_dict = {}

    subdirs = fmEphys.list_subdirs(cfg['apath'])

    check = ['.phy', 'IRspot', 'transfer', 'test']
    rabrvs = [d for d in subdirs if all(x not in d for x in check)]

    if ('use_recordings' in cfg.keys()) and (cfg['use_recordings'] != []):
        rabrvs = cfg['use_recordings']

    if ('ignore_recordings' in cfg.keys()) and (cfg['ignore_recordings'] != []):
        rabrvs = [x for x in rabrvs if x not in cfg['ignore_recordings']]

    for rabrv in rabrvs:

        _rpath = os.path.join(cfg['apath'], rabrv)

        _rabrv_dict = {
            'rpath': _rpath,
            'stim': assign_stim_name(rabrv),
            'rname': get_rec_name(_rpath)
        }

        session_dict[rabrv] = _rabrv_dict

    # sort dictionary of {name: path} so freely-moving recordings are always handled first
    sorted_keys = sorted(session_dict, key=lambda x:('fm' not in x, x))
    session_dict = dict(zip(sorted_keys, [session_dict[k] for k in sorted_keys]))

    cfg['recs'] = session_dict

    return cfg

def get_probe_sites(name=None):
    """
    Get the positions of sites for an ephys probe. Each model of probe
    has a different mapping of site numbers to physical position along a
    shank, and which shank it is on, if it is a multi-shank probe.

    Return the full dictionary, unless a specific model of probe was
    given as an argument, in which case, return an ordered list.
    """

    # Get path
    probe_file_path = os.path.join(os.getcwd(), 'probes.json')

    # Read yaml files
    site_dict = fmEphys.read_json(probe_file_path)

    if name is not None:
        out = site_dict[name]
    else:
        out = site_dict

    return out

def get_rec_name(path):
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

    ignore = ['plot','IR','rep11','betafpv','side_gaze','._']

    fs = fmEphys.find('*.avi', path)
    filt = [f for f in fs if all(b not in f for b in ignore)][0]

    _, tail = os.path.split(filt)
    name_noext, _ = os.path.splitext(tail)
    split_name = name_noext.split('_')[:-1]
    name = '_'.join(split_name)
    
    return name

def create_file_json():

    