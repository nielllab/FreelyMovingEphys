"""
__main__.py
"""
from core.prelim import PrelimRF
import PySimpleGUI as sg
import argparse

def make_window(theme):
    sg.theme(theme)
    options_layout =  [[sg.Text('Choose ephys probe.')],
                       [sg.Combo(values=('default16', 'NN_H16', 'default64', 'NN_H64-LP', 'DB_P64-3', 'DB_P64-8', 'DB_P128-6'), default_value='default16', readonly=True, k='probe', enable_events=True)],
                       [sg.Text('Chose head-fixed white noise recording directory.')],
                       [sg.Button('Open hf1_wn directory')],
                       [sg.Text('Is spike sorting complete?')],
                       [sg.Combo(values=('no','yes'), default_value='no', readonly=True, k='stage', enable_events=True)],
                       [sg.Button('Start')]]
    layout = [[sg.Text('Preliminary whitenoise receptive field mapping', size=(38, 1), justification='center', font=("Times", 16), relief=sg.RELIEF_RIDGE, k='-TEXT HEADING-', enable_events=True)]]
    return sg.Window('PrelimRF', layout)

def main():
    window = make_window(sg.theme())
    while True:
        event, values = window.read(timeout=100)
        if event == 'Open hf1_wn directory':
            wn_dir = sg.popup_get_folder('Choose hf1_wn directory')
            print('Whitenoise directory: ' + str(wn_dir))
        elif event in (None, 'Exit'):
            print('Exiting')
            break
        elif event == 'Run module':
            probe = values['-COMBO-']
            print('Probe: ' + str(probe))
            run_prelim_whitenoise(wn_dir, probe)
    window.close()
    exit(0)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wn_dir', type=str, default=None)
    parser.add_argument('--probe', type=str, default=None)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    if args.wn_dir is None or args.probe is None:
        main()
    else:
        run_prelim_whitenoise(args.wn_dir, args.probe)