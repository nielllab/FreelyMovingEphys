"""
auto_config.py

automaticlly build a config file from inputs to the GUI

Jan. 15, 2021
"""

import json

def write_config(user_dict):
    config_save_path = user_dict['data_path']

    with open('/home/dmartins/Desktop/user_entries.json','w') as fp:
        json.dump(user_dict, fp)

    # internal_dict = {'data_path':user_dict['data_path'],
    #                 'steps_to_run':{
    #                     'deinter':
    #                 }  
    # }
    