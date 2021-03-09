"""
config.py

config and user input utilities
"""
import os, json

def set_preprocessing_config_defaults(novel_config):
    """
    set default values for config file, if any are missing
    defaults are read in from /FreelyMovingEphys/example_configs/preprocessing_config.json
    changing the default values, or adding new config options should be done in that json in the /example_configs/ directory
    input, novel_config, should be a dictionary with config options, structured largely like the default config .json in /example_configs/
    returns the same config with any missing values filled in
    """
    try:
        # get the path of the default json config file in this repository, relative to util/config.py
        # this assumes windows filepaths
        default_json_path = '/'.join(os.path.abspath(__file__).split('\\')[:-2]) + '/example_configs/preprocessing_config.json'
        # read in the json
        with open(default_json_path, 'r') as fp:
            default_config = json.load(fp)
    except FileNotFoundError: 
        # on linux, the file path needs to be found differently
        default_json_path = '/'.join(os.path.abspath(__file__).split('/')[:-2]) + '/example_configs/preprocessing_config.json'
        with open(default_json_path, 'r') as fp:
            default_config = json.load(fp)
    # iterate through keys in the dictionary loaded in from json
    for default_key in default_config:
        # if a key does not exist, add the value in the default config file
        if default_key not in novel_config:
            novel_config[default_key] = default_config[default_key]
            print('filling default value for config option '+default_key +' -- value will be '+str(default_config[default_key]))
    
    return novel_config

def str_to_bool(value):
    """
    parse strings to read argparse flag entries in as True/False
    """
    if isinstance(value, bool):
        return value
    if value.lower() in {'False', 'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'True', 'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')