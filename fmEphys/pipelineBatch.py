"""

"""
import os
import sys
import json
import yaml
import argparse
import warnings
import PySimpleGUI as sg

import fmEphys

warnings.filterwarnings("ignore")

def batch_cycle(cfg, cycle_dict):

    print(' ---> Starting {} {}'.format(cycle_dict['date'], cycle_dict['animal']))

    cfg['probe'] = cycle_dict['probe']
    cfg['animal_directory'] = cycle_dict['path']

    sess = fmEphys.Session(cfg)
    sess.run_main()

def pipelineBatch():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg', type=str)
    parser.add_argument('-j', '--recJson', type=str)
    parser.add_argument('-i', '--iter', type=str)
    parser.add_argument('-l', '--log', type=fmEphys.str_to_bool, default=False)
    args = parser.parse_args()

    if args.log is True:
        sys.stdout = fmEphys.Logger(os.path.split(args.cfg)[0])

    with open(args.recJson, 'r') as fp:
        rec_dict = json.load(fp)

    with open(args.cfg, 'r') as infile:
        cfg = yaml.load(infile, Loader=yaml.FullLoader)

    if args.iter == 'all':
        for key, val in rec_dict.items():
            _use_cfg = cfg.copy()
            batch_cycle(_use_cfg, val)
    else:
        batch_cycle(cfg, rec_dict[args.iter])

if __name__ == '__main__':

    pipelineBatch()
