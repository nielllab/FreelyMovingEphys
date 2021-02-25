"""
doi_utils.py

utilities for doi analysis

Feb 23, 2021
"""
import pandas as pd
import numpy as np

def doi_saline_label(row, group1, group2, condition_dict):
    date = str(row['session']).split('_')[0]
    if any(i in row['session'] for i in group1):
        val = None
    elif any(i in row['session'] for i in group2) and date in condition_dict.get('doi_dates'):
        val = 'doi'
    elif any(i in row['session'] for i in group2) and date in condition_dict.get('saline_dates'):
        val = 'saline'
    return val

def pre_post_label(row, group1, group2, condition_dict):
    date = str(row['session']).split('_')[0]
    if any(i in row['session'] for i in group1) and date in condition_dict.get('pre_dates'):
        val = 'pre'
    elif any(i in row['session'] for i in group1) and date in condition_dict.get('post_dates'):
        val = 'post'
    elif any(i in row['session'] for i in group2):
        val = None
    return val

# condition_dict should be a nested dict like {'doi_saline':{'doi_dates':[...],'saline_dates':[...]},'pre_post':{'pre_dates':[...],'post_dates':[...]}}
def label_doi_conditions(ephys_data, condition_dict):
    group1 = ['fm1','hf1','hf2','hf3','hf4']
    group2 = ['fm2','hf5','hf6','hf7','hf8']
    
    doi_saline_condition_dates = condition_dict['doi_saline']
    pre_post_condition_dates = condition_dict['pre_post']

    ephys_data['doi/saline'] = ephys_data.apply(doi_saline_label, args=(group1, group2, doi_saline_condition_dates), axis=1)
    ephys_data['pre/post'] = ephys_data.apply(pre_post_label, args=(group1, group2, pre_post_condition_dates), axis=1)
    
    return ephys_data