"""


"""


import os
import json
import argparse
import PySimpleGUI as sg

import fmEphys

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

def make_window(probes_path):

    mappings = json.load(probes_path)
    probe_opts = mappings.keys()

    sg.theme('Default1')

    ready = False
    w = set_window_layout(probe_opts)

    while True:

        event, values = w.read(timeout=100)

        if event in (None, 'Exit'):
            break

        elif event == 'k_dir':
            wn_dir = sg.popup_get_folder('Open directory')
            print('Set {} as white noise directory'.format(wn_dir))

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

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default=None)
    parser.add_argument('--probe', type=str, default=None)
    parser.add_argument('--sorted', type=fmEphys.str_to_bool, default=False)
    args = parser.parse_args()

    if (args.dir is not None) or (args.probe is not None):

        wn_dir = args.dir
        use_probe = args.probe
        spike_sorted = args.sorted

    elif (args.dir is None) or (args.probe is None):
        
        # If terminal arguments are missing information, open a GUI
        # If no arguments were given, default to a GUI
        probes_path = os.path.join(os.path.split(__file__), 'utils/probes.json')
        wn_dir, use_probe, spike_sorted = make_window(probes_path)
    
    if spike_sorted is False:

        fmEphys.prelimRF_raw(wn_dir, use_probe)

    elif spike_sorted is True:

        fmEphys.prelimRF_sort(wn_dir, use_probe)
