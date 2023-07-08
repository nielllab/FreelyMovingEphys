""" Quickly map receptive fields.
fmEphys/quickRF.py

Split recordings into individual ephys files. Files for all stimuli
were merged before spike sorting.

Command Line Arguments
----------------------
--dir
    Path to the white noise directory.
--probe
    Name of the probe layout to use. Must be a key in the
    dictionary from fmEphys/utils/probes.json.
--sorted
    Whether ephys data is spike-sorted. If False, the raw
    ephys binary file will be used, and approximate spikes
    will be measured from the LFP of each channel. If true,
    the spike-sorted data from Kilosort and Phy2 will be
    used instead of the ephys binary file.

Example use
-----------
Running from a terminal:
    $ python -m fmEphys.quickRF --dir T:/Path/to/hf1_wn
        --probe DB_P128_6 --sorted False
Or, choosing the parameters in a popup window:
    $ python -m fmEphys.quickRF


Written by DMM, 2021
"""


import os
import json
import argparse
import PySimpleGUI as sg

import fmEphys as fme


def set_window_layout(probe_opts):
    """ Create the window layout to select recording parameters.

    Parameters
    ----------
    probe_opts: list
        List of probe names to display in the dropdown menu.
        e.g., ['default16', 'NN_H16',... ]

    Returns
    -------
    sg.Window
        PySimpleGUI window object.

    """
    
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
    """ Run the GUI to select recording parameters.

    Parameters
    ----------
    probes_path: str
        Path to the probes.json file.
    
    Returns
    -------
    wn_dir: str
        Path to the white noise directory.
    use_probe: str
        Name of the probe layout to use.
    spike_sorted: bool
        Whether ephys data is spike-sorted. If False, the raw
        ephys binary file will be used, and approximate spikes
        will be measured from the LFP of each channel.

    """

    # Read the probes.json file
    with open(probes_path, 'r') as fp:
        mappings = json.load(fp)
    # Get the names from the dictionary keys
    probe_opts = mappings.keys()

    sg.theme('Default1')

    ready = False
    w = set_window_layout(list(probe_opts))

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

def quickRF():

    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default=None)
    parser.add_argument('--probe', type=str, default=None)
    parser.add_argument('--sorted', type=fme.str_to_bool, default=False)
    args = parser.parse_args()

    if (args.dir is not None) or (args.probe is not None):

        wn_dir = args.dir
        use_probe = args.probe
        spike_sorted = args.sorted

    elif (args.dir is None) or (args.probe is None):
        
        # If terminal arguments are missing information, open a GUI
        # If no arguments were given, default to a GUI
        probes_path = os.path.join(os.path.split(__file__)[0], 'utils/probes.json')
        wn_dir, use_probe, spike_sorted = make_window(probes_path)
    
    if spike_sorted is False:

        fme.prelimRF_raw(wn_dir, use_probe)

    elif spike_sorted is True:

        fme.prelimRF_sort(wn_dir, use_probe)

if __name__ == '__main__':

    quickRF()