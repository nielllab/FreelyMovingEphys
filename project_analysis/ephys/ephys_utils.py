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
    # get the .h5 files from each day
    # this will be a list of lists, where each list inside of the main list has all the data of a single session
    sessions = [find('*_ephys_props.h5',os.path.join(datapath, day)) for day in dates]
    # break apart sessions that include more than one subject, and reformat them into seperate lists in the main list
    for session in sessions:
        subject = os.path.split(session[0])[1].split('_')[1]
        new_session = []
        for recording in session:
            if subject not in recording:
                new_session.append(recording)
                session.remove(recording)
        if new_session != []:
            sessions.append(new_session)
    # read the data in and append them into one shared df
    all_data = pd.DataFrame([])
    for session in sessions:
        session_data = pd.DataFrame([])
        for recording in session:
            rec_data = pd.read_hdf(recording)
            # get name of the current recording (i.e. 'fm' or 'hf1_wn')
            rec_type = '_'.join(([col for col in rec_data.columns.values if 'trange' in col][0]).split('_')[:-1])
            # rename spike time columns so that data is retained for each of the seperate trials
            rec_data = rec_data.rename(columns={'spikeT':rec_type+'_spikeT', 'spikeTraw':rec_type+'_spikeTraw'})
            # get column names
            column_names = list(session_data.columns.values) + list(rec_data.columns.values)
            # new columns for same unit within a session
            session_data = pd.concat([session_data, rec_data],axis=1,ignore_index=True)
            # add the list of column names from all sessions plus the current recording
            session_data.columns = column_names
            # remove duplicate columns (i.e. shared metadata)
            session_data = session_data.loc[:,~session_data.columns.duplicated()]
        # new rows for units from different mice or sessions
        all_data = pd.concat([all_data,session_data],axis=0)
    return all_data