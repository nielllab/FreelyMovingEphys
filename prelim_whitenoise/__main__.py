"""
__main__.py
"""
from prelim_whitenoise.prelim_whitenoise import main as run_prelim_whitenoise
import PySimpleGUI as sg

def make_window(theme):
    sg.theme(theme)
    options_layout =  [[sg.Text('Select the model of ephys probe used.')],
                       [sg.Combo(values=('default16', 'NN_H16', 'default64', 'NN_H64-LP', 'DB_P64-3', 'DB_P64-8', 'DB_P128-6', 'DB_P128-D' ), default_value='default16', readonly=True, k='-COMBO-', enable_events=True)],
                       [sg.Text('Select the whitenoise recording directory.')],
                       [sg.Button('Open hf1_wn directory')]]
    logging_layout = [[sg.Text('Run this module')],
                      [sg.Button('Run module')]]
    layout = [[sg.Text('Preliminary whitenoise receptive field mapping', size=(38, 1), justification='center', font=("Times", 16), relief=sg.RELIEF_RIDGE, k='-TEXT HEADING-', enable_events=True)]]
    layout +=[[sg.TabGroup([[sg.Tab('Options', options_layout),
               sg.Tab('Run', logging_layout)]], key='-TAB GROUP-')]]
    return sg.Window('Preliminary whitenoise receptive field mapping', layout)

def main():
    window = make_window(sg.theme())
    while True:
        event, values = window.read(timeout=100)
        if event == 'Open hf1_wn directory':
            binary_file = sg.popup_get_folder('Choose hf1_wn directory')
            print('Whitenoise directory: ' + str(binary_file))
        elif event in (None, 'Exit'):
            print('Exiting')
            break
        elif event == 'Run module':
            probe = values['-COMBO-']
            print('Probe: ' + str(probe))
            run_prelim_whitenoise(binary_file, probe)
    window.close()
    exit(0)

if __name__ == '__main__':
    main()