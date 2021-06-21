"""
batch_analysis.py

takes in a csv file path, yaml config file, and directory into which log should be saved
might work with json config, but ephys analysis won't be possible, so yaml is best
runs preprocessing and ephys analysis for each of the trials marked to be analyzed in csv file
"""
import argparse, traceback, yaml, os
import pandas as pd

from util.params import extract_params
from util.dlc import run_DLC_Analysis
from util.deinterlace import deinterlace_data
from util.track_world import track_LED
from util.config import set_preprocessing_config_defaults, str_to_bool, open_config
from util.calibration import get_calibration_params, calibrate_new_world_vids, calibrate_new_top_vids
from project_analysis.ephys.analyze_ephys import find_files, run_ephys_analysis
from util.log import log
from util.paths import find
from session_analysis import main as analyze_session
from project_analysis.ephys.ephys_utils import population_analysis

# get user arguments
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_filepath', type=str, help='read path for metadata .csv')
    parser.add_argument('--config', type=str, help='yaml config file')
    parser.add_argument('--log_dir', type=str, help='save path for logger .csv')
    parser.add_argument('--clear_dlc', type=str_to_bool, nargs='?', const=True, default=False, help='delete existing DLC .h5 files?')
    args = parser.parse_args()

    return args

def main(csv_filepath, config_path, log_dir, clear_dlc):
    # initialize logger
    logf = log(os.path.join(log_dir,'batch_log.csv'),name=['recording'])

    # read in the csv batch file
    print('opening csv file')
    csv = pd.read_csv(csv_filepath)

    # delete existing DLC .h5 files so that there will be only one in the directory
    # needed in case a different DLC network is being used
    if clear_dlc is True:
        run_preproc = csv.loc[csv['run_preprocessing'] == any('TRUE', True)]
        for ind, row in run_preproc.iterrows():
            del_path = row['animal_dirpath']
            h5_list = find('*DLC_resnet50*.h5',del_path)
            pickle_list = find('*DLC_resnet50*.pickle',del_path)
            file_list = h5_list + pickle_list
            for item in file_list:
                os.remove(item)

    for ind, row in csv.iterrows():
        if row['run_preprocessing'] == any(['TRUE', True]) or row['run_ephys_analysis'] == any(['TRUE', True]):
            # read in the generic config for this batch analysis
            with open(config_path, 'r') as infile:
                config = yaml.load(infile, Loader=yaml.FullLoader)
            # get the provided data path
            # update generic config path for the current index of batch file
            config['animal_dir'] = row['animal_dirpath']
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
                analyze_session(config, clear_dlc=clear_dlc, force_probe_name=row['probe_name'], force_flip_gx_gy=row['flip_gx_gy'],batch=True)
            except Exception as e:
                logf.log([row['experiment_date']+'_'+row['animal_name'], traceback.format_exc()],PRINT=False)
    with open(config_path, 'r') as infile:
        config = yaml.load(infile, Loader=yaml.FullLoader)
    if config['population']['pool_h5_files']:
        population_analysis(config)

if __name__ == '__main__':
    args = get_args()
    main(args.csv_filepath, args.config, args.log_dir, args.clear_dlc)