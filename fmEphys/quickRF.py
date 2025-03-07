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

import sys
import os
import json
import argparse
import tkinter as tk
from tkinter import filedialog

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
    run_opts: dict
        Dictionary of run options.

    """
    
    run_opts = {
        'probe': None,
        'dir': None,
        'isSorted': None,

    }

    # Function to open directory selection dialog
    def open_directory_dialog():
        directory = filedialog.askdirectory()
        if directory:
            run_opts['dir'] = directory
            print(f"Selected directory: {directory}")

    # Function to handle radio button selection
    def on_radio_button_selected():
        selected_option = var.get()
        run_opts['isSorted'] = selected_option
        print(f"Selected spike-sort status: {selected_option}")

    def get_dropdown_value():
        selected_option = dropdown.get()
        print(f"Selected probe: {selected_option}")
        run_opts['probe'] = selected_option

    def close_win():
        global ret_run_opts
        get_dropdown_value()
        print('Closing window.')
        root.destroy()
        ret_run_opts = run_opts

    # Create the main window
    root = tk.Tk()
    root.title("Quick receptive field mapping")
    root.minsize(width=500, height= 40)
    # Dropdown menu
    dropdown_label = tk.Label(root, text="Select probe:")
    dropdown_label.pack(padx=10, pady=5)
    dropdown = tk.StringVar()
    dropdown.set(probe_opts[0])  # Set default value
    dropdown_menu = tk.OptionMenu(root, dropdown, *probe_opts)
    dropdown_menu.pack(padx=10, pady=5)

    # Button to open directory dialog
    directory_button = tk.Button(root, text="White noise directory", command=open_directory_dialog)
    directory_button.pack(padx=10, pady=5)

    # Radio buttons
    var = tk.StringVar(value="Raw")  # Default selection

    radio_button1 = tk.Radiobutton(root, text="Raw", variable=var, value="raw", command=on_radio_button_selected)
    radio_button1.pack(padx=10, pady=5)

    radio_button2 = tk.Radiobutton(root, text="Spike-sorted", variable=var, value="spike-sorted", command=on_radio_button_selected)
    radio_button2.pack(padx=10, pady=5)

    # Button
    button = tk.Button(root, text="Start", command=lambda: close_win())
    button.pack(padx=10, pady=5)

    # Run the application
    root.mainloop()

    return ret_run_opts


def make_window(probes_path):
    
    # Read the probes.json file
    with open(probes_path, 'r') as fp:
        mappings = json.load(fp)
    # Get the names from the dictionary keys
    probe_opts = mappings.keys()

    run_opts = set_window_layout(list(probe_opts))

    if type(run_opts['probe']) == str:
        use_probe = run_opts['probe']
    else:
        sys.exit('No probe selected.')
    if type(run_opts['dir']) == str:
        wn_dir = run_opts['dir']
    else:
        sys.exit('No white noise directory selected.')
    if run_opts['isSorted'] == 'spike-sorted':
        spike_sorted = True
    elif run_opts['isSorted'] == 'raw':
        spike_sorted = False
    elif run_opts['isSorted'] is None:
        sys.exit('No spike-sorting status given.')

    return wn_dir, use_probe, spike_sorted


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