"""

"""

import os
import argparse
import PySimpleGUI as sg

import fmEphys

def main(cfg_path):

    ### Get paths and names
    cfg = fmEphys.get_cfg(cfg_path)
    cfg = fmEphys.set_cfg_paths(cfg)

    # Iterate through recordings
    # rabrv e.g., 'hf1_wn'
    # rinfo
    #    e.g.,  {
    #       'rpath': '/recording/path',
    #       'stim': 'hf_white_noise',
    #       'rname': '010122_Animal_Rig_Manipulation'
    #     }
    for rabrv, rinfo in cfg['recs'].items():
        cfg['_rpath'] = rinfo['rpath']
        cfg['_rname'] = rinfo['rname']
        cfg['_stim'] = rinfo['stim']

        cfg = fmEphys.fill_rec_details(cfg)
    
        ### Preprocessing
        cfg = fmEphys.preprocess_eyecam(cfg)


def pipeline():

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg', type=str)
    parser.add_argument('--log', type=fmEphys.str_to_bool, default=False)
    args = parser.parse_args()

    sg.theme('Default1')
    if args.cfg is None:
        # if no path was given as an argument, open a dialog box
        cfg_path = sg.popup_get_file('Choose animal ephys_cfg.yaml')
    else:
        cfg_path = args.cfg


    if args.log is True:
        head, _ = os.path.split(cfg_path)

        date_str, time_str = fmEphys.fmt_now()
        log_path = os.path.join(head,
                        'errlog_{}_{}.txt'.format(date_str, time_str))

        logging = fmEphys.Log(log_path)


    main(cfg_path)

if __name__ == '__main__':
    pipeline()
