"""
ephys_utils.py

utilities for using ephys analysis outputs

Feb 23, 2021
"""
import pandas as pd
import numpy as np
import os
from util.paths import find

def load_ephys(datapath, dates):
    sessions = [find('*_ephys_props.h5',os.path.join(datapath, day)) for day in dates]
    all_data = pd.DataFrame([])
    for session in sessions:
        session_data = pd.DataFrame([])
        for recording in session:
            rec_data = pd.read_hdf(recording)
            column_names = list(session_data.columns.values) + list(rec_data.columns.values)
            session_data = pd.concat([session_data, rec_data],axis=1,ignore_index=True) # new columns for same unit within a day
            session_data.columns = column_names
            session_data = session_data.loc[:,~session_data.columns.duplicated()]
        all_data = pd.concat([all_data,session_data],axis=0) # new rows for units from different mice or days
        
    return all_data