"""
create_datasets_alignment.py

create a .npy of post-alignment timestamps matching each frame of the
worldcam video to the ephys aquisition timestamps, and bin spike times
into spike rates in each frame's bin
"""
import numpy as np
import xarray as xr
import pandas as pd
import os, json, sys
from tqdm import tqdm
sys.path.insert(0,'/home/niell_lab/Documents/github/FreelyMovingEphys/')
from util.paths import *

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_csv_path', type=str, default='/home/niell_lab/Documents/github/FreelyMovingEphys/project_analysis/inception_loop/metadata/exp_pool.csv')
    parser.add_argument('--root_dir', type=str, default='/home/niell_lab/data/freely_moving_ephys/inception_loop/inputs/')
    args = parser.parse_args()
    return args

def main(args):
    print('reading metadata')
    root_dir = args.root_dir
    # open the csv file of the pooled experiments
    metadata = pd.read_csv(args.exp_csv_path)
    # build a new df of essential info
    print('organizing metadata and finding files')
    usedata = pd.DataFrame([])
    usedata['path'] = metadata['animal_dirpath']
    usedata['date'] = [''.join([q[0].zfill(2),q[1].zfill(2),q[2].zfill(2)]) for q in [str(j).split('/') for j in list(metadata['experiment_date'])]]
    usedata['animal'] = metadata['animal_name']
    # find the read paths that were used to collect the worldcam frames
    search_paths = [i for i in [row['path'] if row['animal'] in dirname and row['date'] in dirname else None for ind, row in usedata.iterrows() for dirname in list_subdirs(root_dir)] if i is not None]
    # worldcam directory names
    wc_names = list_subdirs(root_dir)
    # get the read path for the initial data source
    wc_read_paths = ['/'.join([split.capitalize() if split != '' and split == x.split('\\')[2] else split for split in x.split('\\')[2:]]) for x in search_paths]
    wc_read_paths = [os.path.join('/home/niell_lab/', x) for x in wc_read_paths]
    # get all the timestamp files, and then filter out the unformatted ones
    wc_timestamps = [find('*'+wc_name+'*.csv', search_path) for wc_name in wc_names for search_path in wc_read_paths]
    timestamp_files = [i for i in [item for items in wc_timestamps for item in items] if i != [] and 'BonsaiTSformatted' in i]
    # open train + validation files
    train_file = pd.read_csv(os.path.join(root_dir, 'WC_Train_Data.csv'))
    val_file = pd.read_csv(os.path.join(root_dir, 'WC_Val_Data.csv'))
    # now, read the worldcam timestamps in for each video
    for timestamp in timestamp_files:
        print('writing file for '+timestamp)
        # find the ephys file, which should be in the same dir as timestamps
        ephys_file = find('*ephys_merge.json', os.path.split(timestamp)[0])[0]
        # then read in the worldcam timestamps
        worldT = np.array(pd.read_csv(timestamp))
        # timing correction to sync with ephys aquisition
        ephys_data = pd.read_json(ephys_file)
        ephysT0 = ephys_data.iloc[0,12]
        worldT = worldT - ephysT0
        if worldT[0]<-600:
            worldT = worldT + 8*60*60
        # get number of spikes between worldcam timestamps
        print('binning spike rates by worldcam frame')
        hist_dt = 1/60
        hist_t = np.arange(0, np.nanmax(worldT),hist_dt)
        goodcells = ephys_data.loc[ephys_data['group']=='good']
        for unit, unit_row in enumerate(goodcells.index):
            rate, bins = np.histogram(ephys_data.at[unit,'spikeT'],hist_t)
            # add the calculated spike rate to the input .csv as a new column for each frame bin
            # get the base name of the current file
            actual_base_name = os.path.split(timestamp)[1].replace('_BonsaiTSformatted.csv','')
            print(actual_base_name)
            print('populating training file with spike rates')
            for ind, file_row in tqdm(train_file.iterrows()):
                frame_num = int(file_row['FileName'].split('.')[0].split('_')[1].strip('0'))
                row_base_name = file_row['BasePath']
                # if the row is for the current file
                if row_base_name == actual_base_name:
                    # find the bin in histogram bins that matches this row's frame_num
                    # add as column to train file
                    try:
                        train_file.loc[ind, 'unit'+str(unit)+'_spikerate'] = rate[frame_num]
                    except IndexError:
                        train_file.loc[ind, 'unit'+str(unit)+'_spikerate'] = 0
            print('populating validation file with spike rates')
            for ind, row in tqdm(val_file.iterrows()):
                frame_num = int(file_row['FileName'].split('.')[0].split('_')[1].strip('0'))
                row_base_name = file_row['BasePath']
                # if the row is for the current file
                if row_base_name == actual_base_name:
                    # find the bin in histogram bins that matches this row's frame_num
                    # add as column to val file
                    try:
                        val_file.loc[ind, 'unit'+str(unit)+'_spikerate'] = rate[frame_num]
                    except IndexError:
                        val_file.loc[ind, 'unit'+str(unit)+'_spikerate'] = 0
    print('writing new csv files')
    train_file.to_csv(os.path.join(root_dir, 'WC_Train_Data_spikes.csv'), index=False)
    val_file.to_csv(os.path.join(root_dir, 'WC_Val_Data_spikes.csv'), index=False)

if __name__ == '__main__':
    args = get_args()
    main(args)