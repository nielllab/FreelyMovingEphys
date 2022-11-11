"""Preliminary receptive field
"""
import os
import json
import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm
import scipy.interpolate
import PySimpleGUI as sg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import utils

def prelimRF_raw(rdir):

    # Get files
    world_avi = utils.path.find('*WORLD.avi', wn_dir, mr=True)
    world_csv = utils.path.find('*WORLD_BonsaiTS.csv', wn_dir, mr=True)
    ephys_bin = utils.path.find('*Ephys.bin', wn_dir, mr=True)
    ephys_csv = utils.path.find('*Ephys_BonsaiBoardTS.csv', wn_dir, mr=True)

    # Worldcam setup
    worldT = utils.time.read(world_csv)
    worldT = worldT - ephysT0

    stim_arr = utils.video.to_arr(world_avi, ds=0.25)

    # Ephys
    ephysT = utils.time.read(ephys_csv)
    ephysT0 = ephysT[0]

    n_ch, _ = utils.base.probe_to_nCh(probe)
    ephys = utils.ephys.read_ephysbin(ephys_bin, n_ch=n_ch)
    spikeT = utils.ephys.calc_approx_sp(ephys, ephysT0, fixT=True) # values are corrected for drift/offset, too

    # get stimulus
    cam_gamma = 2
    norm_stim = (stim_arr / 255)**cam_gamma

    std_im = np.std(norm_stim, axis=0)
    std_im[std_im < 10/255] = 10/255

    img_norm = (norm_stim - np.mean(norm_stim, axis=0)) / std_im
    img_norm = img_norm * (std_im > 20/255)
    img_norm[img_norm < -2] = -2

    movInterp = scipy.interpolate.interp1d(worldT, img_norm, axis=0, bounds_error=False)

    # Start the pdf file
    str_date, _ = utils.base.str_today()
    pdf_path = os.path.join(rdir, 'quickRF_{}.pdf'.format(str_date))
    pdf = PdfPages(pdf_path)

    # Calculate tuning
    _ = utils.stim.calc_tuning()

    # Calculate STAs
    _ = utils.stim.calc_STA()


def set_window_layout(probe_opts):
    sg.theme('Default1')
    opt_layout =  [[sg.Text('Probe layout')],
                   [sg.Combo(values=(probe_opts), default_value=probe_opts[0],
                             readonly=True, k='k_probe', enable_events=True)],

                   [sg.Text('White noise directory')],
                   [sg.Button('Open directory', k='k_dir')],

                   [sg.Radio('Raw', group_id='code_type', k='k_raw', default=True)],
                   [sg.Radio('Spike-sorted', group_id='code_type', k='k_sorted')],
                
                   [sg.Button('Start', k='k_start')]]

    return sg.Window('FreelyMovingEphys: Preliminary Modules', opt_layout)


def make_window(chmaps_path):

    with open(chmaps_path, 'r') as fp:
            mappings = json.load(fp)
    probe_opts = mappings.keys()

    sg.theme('Default1')
    ready = False
    w = set_window_layout(probe_opts)
    while True:
        event, values = w.read(timeout=100)

        if event == 'k_dir':
            wn_dir = sg.popup_get_folder('Open directory')
            print('Set {} as white noise directory'.format(wn_dir))

        elif event in (None, 'Exit'):
            break

        elif event == 'k_start':
            use_probe = values['k_probe']

            if values['k_raw'] is True:
                spike_sorted = False
            elif values['k_sorted'] is True:
                spike_sorted = True

            ready = True
            break

    w.close()
    if ready:
        return wn_dir, use_probe, spike_sorted
    else:
        return None, None, None

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dir', type=str, default=None)
    parser.add_argument('-p', '--probe', type=str, default=None)
    parser.add_argument('-s', '--sorted', type=utils.base.str_to_bool, nargs='?', const=True, default=False)
    args = parser.parse_args()

    if args.dir is None or args.probe is None:
        src_dir, _ = os.path.split(__file__)
        repo_dir, _ = os.path.split(src_dir)
        chmaps_path = os.path.join(repo_dir, 'config/probe_sites.json')
        
        wn_dir, probe, ssorted = make_window(chmaps_path)
    else:
        wn_dir = args.dir; probe = args.probe; ssorted = args.sorted

    if all(x is not None for x in [wn_dir, probe, ssorted]):
        print('White noise path: {}\n Probe map: {}\n Spike sorted: {}'.format(
            wn_dir, probe, ssorted))

        if ssorted is False:
            prelimRF_raw(wn_dir, probe)

        elif ssorted is True:
            prelimRF_sort(wn_dir, probe)