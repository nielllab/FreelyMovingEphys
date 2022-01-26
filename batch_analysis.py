"""
batch_analysis.py

takes in a csv file path, yaml config file, and directory into which log should be saved
might work with json config, but ephys analysis won't be possible, so yaml is best
runs preprocessing and ephys analysis for each of the trials marked to be analyzed in csv file
"""
import argparse
import traceback
import yaml
import os
import platform
import pandas as pd
import logging
import numpy as np
from pathlib import Path

from util.params import extract_params
# from util.dlc import run_DLC_Analysis
# from util.deinterlace import deinterlace_data
# from util.track_world import track_LED
from util.config import set_preprocessing_config_defaults, str_to_bool, open_config
# from util.calibration import get_calibration_params, calibrate_new_world_vids, calibrate_new_top_vids
# from project_analysis.ephys.analyze_ephys import find_files, run_ephys_analysis
from util.log import log
from util.paths import find
from session_analysis import main as analyze_session
from project_analysis.ephys.ephys_utils import population_analysis

# get user arguments


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_filepath', type=str, help='read path for metadata .csv',
                        default='T:/BinocOptoPreyCapture/csv_testing.csv')
    parser.add_argument('--config', type=str, help='yaml config file',
                        default='C:/Users/Niell lab/Documents/GitHub/FreelyMovingEphys/project_analysis/prey_capture/config.yaml')
    parser.add_argument('--log_dir', type=str,
                        help='save path for logger .csv', default='T:/BinocOptoPreyCapture/')
    parser.add_argument('--clear_dlc', type=str_to_bool, nargs='?',
                        const=True, default=False, help='delete existing DLC .h5 files?')
    args = parser.parse_args()

    return args

 
def main(csv_filepath, config_path, log_dir, clear_dlc):
    base_path = Path(csv_filepath).parent

    # initialize logger
    logf = log(os.path.join(log_dir, 'batch_log.txt'), name=['recording'])
    logging.basicConfig(filename=os.path.join(log_dir, 'batch_log.txt'), level=logging.DEBUG)

    # read in the csv batch file
    print('opening csv file')
    csv = pd.read_csv(csv_filepath)

    # if running on linux, make new paths to replace the windows paths in csv
    if platform.system() == 'Linux':
        for ind, row in csv.iterrows():
            drive = [row['drive'] if row['drive'] ==
                     'nlab-nas' else row['drive'].capitalize()][0]
            csv[ind, 'animal_dirpath'] = os.path.expanduser('~/'+('/'.join([row['computer'].title(
            ), drive] + list(filter(None, row['animal_dirpath'].replace('\\', '/').split('/')))[2:])))

    csv['experiment_date'] = pd.to_datetime(
        csv['experiment_date'], infer_datetime_format=True, format='%m%d%Y').dt.strftime('%m%d%y')
    csv = csv.loc[(csv['run_preprocessing'] == True) |
                  (csv['run_ephys_analysis'] == True)]
    csv = csv[csv['experiment_outcome'] == 'good'].reset_index(drop=True)
    # Format Pandas Dataframe to have Trial number and Stimulus condition

    cols = list(csv.keys()[:-4])
    cols.append('Trial')
    cols.append('LaserOn')
    csv2 = pd.DataFrame(columns=cols)
    for ind, row in csv.iterrows():
        for n in range(1, 6):
            if pd.isna(row['excluded_trials'])==True:
                if '*' in row['{:d}'.format(n)]:
                    csv2 = csv2.append(
                        row[:-4].append(pd.Series([n, True], index=['Trial', 'LaserOn'])), ignore_index=True)
                else:
                    csv2 = csv2.append(
                        row[:-4].append(pd.Series([n, False], index=['Trial', 'LaserOn'])), ignore_index=True)
            elif n in np.array(row['excluded_trials']):
                pass
    inds, labels = csv2['Environment'].factorize()
    # delete existing DLC .h5 files so that there will be only one in the directory
    # needed in case a different DLC network is being used
    if clear_dlc is True:
        run_preproc = csv2.loc[csv2['run_preprocessing'] == any('TRUE', True)]
        for ind, row in run_preproc.iterrows():
            del_path = row['animal_dirpath']
            h5_list = find('*DLC_resnet50*.h5', del_path)
            pickle_list = find('*DLC_resnet50*.pickle', del_path)
            file_list = h5_list + pickle_list
            for item in file_list:
                os.remove(item)

    for ind, row in csv.iterrows():
        # if step was switched off for this index in the batch file, overwrite what is in the config file
        # if the csv file has a step switched on, this will leave the config file as it is

        if row['run_preprocessing'] == any(['TRUE', True]) or row['run_ephys_analysis'] == any(['TRUE', True]):
            # read in the generic config for this batch analysis
            with open(config_path, 'r') as infile:
                config = yaml.load(infile, Loader=yaml.FullLoader)
            # get the provided data path
            # update generic config path for the current index of batch file
            if os.path.exists(str(row['animal_dirpath'])):
                data_path = row['animal_dirpath']
            else:
                data_path = os.path.normpath(os.path.join(
                    row['drive']+':/', 'BinocOptoPreyCapture', row['experiment_date'], row['animal_name']))
            # update generic config path for the current index of batch file
            config['animal_dir'] = data_path
            # if step was switched off for this index in the batch file, overwrite what is in the config file
            # if the csv file has a step switched on, this will leave the config file as it is
            if row['run_preprocessing'] != any(['TRUE', True]):
                config['deinterlace']['run_deinter'] = False
                config['img_correction']['run_img_correction'] = False
                config['calibration']['run_cam_calibration'] = False
                config['calibration']['undistort_recordings'] = False
                config['pose_estimation']['run_dlc'] = False
                config['parameters']['run_params'] = False
                config['ir_spot_in_space']['run_is_spot_in_space'] = False
            if row['run_ephys_analysis'] != any(['TRUE', True]):
                config['ephys_analysis']['run_ephys_analysis'] = False
            # run session analysis using the yaml config file
            try:
                if (row['run_ephys_analysis'] == True) & (row['run_preprocessing'] == True):
                    analyze_session(
                        config, clear_dlc=clear_dlc, force_probe_name=row['probe_name'], force_flip_gx_gy=row['flip_gx_gy'], batch=True)
                elif (row['run_ephys_analysis'] == False) & (row['run_preprocessing'] == True):
                    analyze_session(
                        config, clear_dlc=clear_dlc, force_probe_name=None, force_flip_gx_gy=False, batch=True)
            except Exception as e:
                logf.log([row['experiment_date'] + '_' +row['animal_name'], traceback.format_exc()], PRINT=False)
                # logging.error(e,exc_info=True)
        if config['population']['pool_h5_files']:
            population_analysis(config)


if __name__ == '__main__':
    args = get_args()
    main(os.path.normpath(args.csv_filepath), os.path.normpath(
        args.config), os.path.normpath(args.log_dir), args.clear_dlc)
