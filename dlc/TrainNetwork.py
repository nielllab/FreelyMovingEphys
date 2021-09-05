import deeplabcut 
import glob, os
import numpy as np
import argparse
from tkinter import filedialog
import tkinter as tk
from pathlib import Path
from tqdm import trange, tqdm


# config_path = r'T:\deeplabcut_projects\BinocularOpto-Elliott-2021-08-12\config.yaml'
def pars_args():
    # get user inputs
    parser = argparse.ArgumentParser(description='deinterlace videos and adjust timestamps to match')
    parser.add_argument('--multi_animal', type=bool, default=False)
    parser.add_argument('--add_videos', type=bool, default=False)
    parser.add_argument('--retrain', type=bool, default=False)
    parser.add_argument('--config_path', type=str, default='~/Research/DLC_Networks/BinocularOpto-Elliott-2021-08-12/config.yaml')
    # parser.add_argument('--videofile_path', type=str, default='~\Documents\deeplabcut_projects\EphysEyeCams3-dylan-2021-02-02\videos')
    args = parser.parse_args()
    return args

def main(args):
    
    config_path = os.path.expanduser(args.config_path)
    train_pose_config, _, _ = deeplabcut.return_train_network_path(config_path)

    augs = {
        "gaussian_noise": True,
        "elastic_transform": True,
        "rotation": 180,
        "covering": True,
        "motion_blur": True,
    }
    deeplabcut.auxiliaryfunctions.edit_config(
        train_pose_config,
        augs,
    )

    deeplabcut.train_network(config_path, allow_growth=True, displayiters=100, saveiters=5000)

    deeplabcut.evaluate_network(config_path, plotting=True, trainingsetindex='all')


if __name__ == '__main__':
    args = pars_args()
    main(args)