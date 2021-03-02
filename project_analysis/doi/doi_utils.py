"""
doi_utils.py

utilities for doi analysis

Feb 23, 2021
"""
import pandas as pd
import numpy as np

def doi_saline_label(row, stim, group1, group2, condition_dict):
    date = str(row['session']).split('_')[0]
    if any(i in stim for i in group1):
        val = None
    elif any(i in stim for i in group2) and date in condition_dict.get('doi_dates'):
        val = 'doi'
    elif any(i in stim for i in group2) and date in condition_dict.get('saline_dates'):
        val = 'saline'
    else:
        val = None
    return val

def pre_post_label(row, stim, group1, group2, condition_dict):
    date = str(row['session']).split('_')[0]
    if any(i in stim for i in group1) and date in condition_dict.get('pre_dates'):
        val = 'pre'
    elif any(i in stim for i in group1) and date in condition_dict.get('post_dates'):
        val = 'post'
    elif any(i in stim for i in group2):
        val = None
    else:
        val = None
    return val

# condition_dict should be a nested dict like {'doi_saline':{'doi_dates':[...],'saline_dates':[...]},'pre_post':{'pre_dates':[...],'post_dates':[...]}}
def label_doi_conditions(ephys_data, condition_dict):
    group1 = ['fm1','hf1','hf2','hf3','hf4']
    group2 = ['fm2','hf5','hf6','hf7','hf8']
    
    doi_saline_condition_dates = condition_dict['doi_saline']
    pre_post_condition_dates = condition_dict['pre_post']

    stim_list = ['_'.join(col.split('_')[:-1]) for col in ephys_data.columns.values if 'trange' in col]
    
    for stim in stim_list:
        ephys_data[stim+'doi/saline'] = ephys_data.apply(doi_saline_label, args=(stim, group1, group2, doi_saline_condition_dates), axis=1)
        ephys_data[stim+'pre/post'] = ephys_data.apply(pre_post_label, args=(stim, group1, group2, pre_post_condition_dates), axis=1)
    
    return ephys_data