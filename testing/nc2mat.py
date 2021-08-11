import xarray as xr
from scipy.io import savemat
import numpy as np
import os, sys, yaml, argparse
sys.path.insert(0, '/Users/Angie Michaiel/Documents/GitHub/FreelyMovingEphys')
from utils.paths import find

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='C:/Users/Angie Michaiel/Desktop/config.yaml')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    with open(args.config, 'r') as infile:
        config = yaml.load(infile, Loader=yaml.FullLoader)
    base_path = config['animal_dir']
    eye_files = find('*Reye*.nc', base_path)
    for eye_file in eye_files:
        data = xr.open_dataset(eye_file)
        data_dict = dict(zip(list(data.REYE_ellipse_params['ellipse_params'].values), [data.REYE_ellipse_params.sel(ellipse_params=p).values for p in list(data.REYE_ellipse_params['ellipse_params'].values)]))
        save_name = os.path.join(os.path.split(eye_file)[0], os.path.splitext(os.path.split(eye_file)[1])[0])+'.mat'
        print(save_name)
        savemat(save_name, data_dict)
        print('done')
