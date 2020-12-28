"""
config.py

config and user input utilities

Dec. 19, 2020
"""
# package imports
import os, json

# set default values for config file, if any are missing
# defaults are read in from /FreelyMovingEphys/example_configs/preprocessing_config.json
# changing the default values, or adding new config options should be done in that json in the /example_configs/ directory
def set_preprocessing_config_defaults(novel_config):
    # get the path of the default json config file in this repository, relative to util/config.py
    default_json_path = os.path.dirname(os.path.join(os.path.realpath(__file__),'..','/example_configs/preprocessing_config.json'))
    # read in the json
    with open(default_json_path, 'r') as fp:
        default_config = json.load(fp)
    for default_key, default_val in default_config:
        if default_key not in novel_config:
            novel_config[default_key] = default_val
            print('filling default value for config option '+default_key +' -- value will be '+str(default_key))

    return novel_config