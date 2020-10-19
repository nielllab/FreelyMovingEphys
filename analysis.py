"""
analysis.py

ephys analysis and figures

Oct. 19, 2020
"""

import argparse, json, sys, os

from util.read_data import pars_args
from util.figures import get_figures

def main(args):
    json_config_path = os.path.expanduser(args.json_config_path)

    # open config file
    with open(json_config_path, 'r') as fp:
        config = json.load(fp)

    get_figures(config)

if __name__ == '__main__':
    args = pars_args()
    main(args)