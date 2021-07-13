"""
batch_analysis.py
"""
import traceback, yaml, os, platform
import pandas as pd

from utils.log import log
from utils.paths import find
from session_analysis.session_analysis import main as analyze_session
from utils.ephys import population_analysis

def main(csv_filepath, config_path, log_dir, clear_dlc):
    # initialize logger
    logf = log(os.path.join(log_dir,'batch_log.csv'),name=['recording'])

    # read in the csv batch file
    print('opening csv file')
    csv = pd.read_csv(csv_filepath)

    # if running on linux, make new paths to replace the windows paths in csv
    if platform.system() == 'Linux':
        for ind, row in csv.iterrows():
            if type(row['drive']) == str and type(row['animal_dirpath']) == str and type(row['computer']) == str:
                drive = [row['drive'] if row['drive'] == 'nlab-nas' else row['drive'].capitalize()][0]
                csv.loc[ind,'animal_dirpath'] = os.path.expanduser('~/'+('/'.join([row['computer'].title(), drive] + list(filter(None, row['animal_dirpath'].replace('\\','/').split('/')))[2:])))
            else:
                csv = csv.drop(index=ind)

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
            except:
                logf.log([row['experiment_date']+'_'+row['animal_name'], traceback.format_exc()],PRINT=False)
    with open(config_path, 'r') as infile:
        config = yaml.load(infile, Loader=yaml.FullLoader)
    if config['population']['pool_h5_files']:
        population_analysis(config)