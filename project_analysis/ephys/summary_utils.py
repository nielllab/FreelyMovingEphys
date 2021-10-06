"""
summary_utils.py
"""
import pandas as pd
import os, platform

from utils.ephys import load_ephys
from project_analysis.ephys.population_utils import make_session_summary, make_unit_summary

def independent_summary(config):
    print('setting up metadata')
    split_path = [x for x in (r'%s'%config['animal_dir']).split('/') if x]
    print(split_path)
    csv = pd.Series({
        'run_preprocessing': False,
        'run_ephys_analysis': False,
        'load_for_data_pool': True,
        'experiment_date': split_path[4],
        'animal_name': split_path[5],
        'best_fm_light': config['independent_summary']['best_fm_light'],
        'best_fm_dark': (config['independent_summary']['best_fm_dark'] if config['independent_summary']['best_fm_dark'] != False else ''),
        'unit2highlight': config['ephys_analysis']['unit_to_highlight'],
        'probe_name': config['ephys_analysis']['unit_to_highlight'],
        'animal_dirpath': config['animal_dir'],
        'computer': split_path[0],
        'drive': split_path[1]
    })
    print('loading data from .h5 files')
    df = load_ephys(csv)
    print('writing independent session summary')
    make_session_summary(df, config['animal_dir'])
    print('writing independent unit summary')
    _ = make_unit_summary(df, config['animal_dir'])
    print('done')