"""
oa_utils.py
"""
import sys
sys.path.insert(0, '/home/niell_lab/Documents/github/FreelyMovingEphys/')
import pandas as pd
import xarray as xr
from utils.format_data import *
from utils.paths import find
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import os
from pathlib import Path

def make_oa_df(directory_path, dates):
    data_dict = {'date': [],
                'animal': [],
                'task': [],             
                'poke1_ts':[],
                'poke2_ts': [],
                'top1_ts': [],
                'poke1_t0':[],
                'poke2_t0': [],
                'top1_t0': []}
    # list of dates for analysis
    data_path = Path(directory_path).expanduser()
    all_paths = []
    # populate dict with metadata and timestamps
    for date in dates:
        for ani in os.listdir(data_path / date): 
            for task in os.listdir(data_path / date/ ani):
                data_paths = list((data_path / date/ ani/ task).rglob('*.csv'))
                if data_paths != []:
                    data_dict['date'].append(data_paths[1].name.split('_')[0])
                    data_dict['animal'].append(data_paths[1].name.split('_')[1])
                    data_dict['task'].append(data_paths[1].name.split('_')[4])
                for ind, csv in enumerate(data_paths):
                    data_dict[data_paths[ind].name.split('_')[5] +'_ts'].append(open_time(csv))
                    data_dict[data_paths[ind].name.split('_')[5] +'_t0'].append(open_time(csv)[0])
    df = pd.DataFrame.from_dict(data_dict)
    return df

def filter_likelihood(da, thresh=0.99):
    x_cols = [i for i in da.columns.values if '_x' in i and 'arena' not in i]
    y_cols = [i for i in da.columns.values if '_y' in i and 'arena' not in i]
    l_cols = [i for i in da.columns.values if '_likelihood' in i and 'arena' not in i]
    for i in range(len(x_cols)):
        x = da.loc[:,x_cols[i]]; y = da.loc[:,y_cols[i]]; l = da.loc[:,l_cols[i]]
        x[l<thresh] = np.nan; y[l<thresh] = np.nan
        da.loc[:,x_cols[i]] = x; da.loc[:,y_cols[i]] = y
    return da

def make_task_df(df, index, dlc_h5):
    row = df.iloc[index]
    num_odd_trails = np.min([len(row['poke1_ts']), len(row['poke2_ts'])])
    df1 = pd.DataFrame([])
    dlc_positions, dlc_labels = open_h5(dlc_h5)
    dlc_positions = filter_likelihood(dlc_positions)
    count = -1
    for c in range(num_odd_trails):
        # odd
        count += 1
        df1.at[count, 'first_poke'] = row['poke1_ts'][c]
        df1.at[count, 'second_poke'] = row['poke2_ts'][c]
        time = row['top1_ts']; time = time[time > df1.loc[count,'first_poke']]; time = time[time < df1.loc[count,'second_poke']]
        df1.at[count, 'trail_timestamps'] = time.astype(object)
        start_stop_inds = (int(np.where([row['top1_ts']==time[0]])[1]), int(np.where([row['top1_ts']==time[-1]])[1]))
        for pos in dlc_positions:
            df1.at[count, pos] = np.array(dlc_positions.loc[start_stop_inds[0]:start_stop_inds[1], pos]).astype(object)
        df1.at[count, 'len'] = start_stop_inds[1] - start_stop_inds[0]
        # even
        count += 1
        if c+1 < len(row['poke1_ts']):
            df1.at[count, 'first_poke'] = row['poke2_ts'][c]
            df1.at[count, 'second_poke'] = row['poke1_ts'][c+1]
            time = row['top1_ts']; time = time[time > df1.loc[count,'first_poke']]; time = time[time < df1.loc[count,'second_poke']]
            df1.at[count, 'trail_timestamps'] = time.astype(object)
            start_stop_inds = (int(np.where([row['top1_ts']==time[0]])[1]), int(np.where([row['top1_ts']==time[-1]])[1]))
            for pos in dlc_positions:
                df1.at[count, pos] = np.array(dlc_positions.loc[start_stop_inds[0]:start_stop_inds[1], pos]).astype(object)
            df1.at[count, 'len'] = start_stop_inds[1] - start_stop_inds[0]
    df1['animal'] = row['animal']; df1['date'] = row['date']; df1['task'] = row['task']
    return df1