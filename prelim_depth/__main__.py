import PySimpleGUI as sg

from utils.prelim import PrelimDepth

def make_window(theme):
    sg.theme(theme)
    options_layout =  [[sg.Text('Choose a probe.')],
                       [sg.Combo(values=('default16', 'NN_H16', 'default64', 'NN_H64-LP', 'DB_P64-3', 'DB_P64-8', 'DB_P128-6'), default_value='default16', readonly=True, k='-COMBO-', enable_events=True)],
                       [sg.Text('Choose an ephys .bin file.')],
                       [sg.Button('Open .bin file')]]
    logging_layout = [[sg.Text('Run this module')],
                      [sg.Button('Run module')],
                      [sg.Text("Traceback")],
                      [sg.Output(size=(60,15))]]
    layout = [[sg.Text('Estimate laminar depth from MUA power', size=(38, 1), justification='center', font=("Times", 16), relief=sg.RELIEF_RIDGE, k='-TEXT HEADING-', enable_events=True)]]
    layout +=[[sg.TabGroup([[sg.Tab('Options', options_layout),
               sg.Tab('Run', logging_layout)]], key='-TAB GROUP-')]]     
    return sg.Window('Estimate laminar depth from MUA power', layout)
    
def main():
    window = make_window(sg.theme())
    while True:
        event, values = window.read(timeout=100)
        if event == 'Open ephys binary file':
            binary_file = sg.popup_get_file('Choose .bin file')
            print('Binary file: ' + str(binary_file))
        elif event in (None, 'Exit'):
            print('Exiting')
            break
        elif event == 'Run module':
            probe = values['-COMBO-']
            print('Probe: ' + str(probe))
            
            pd = PrelimDepth(binary_file, probe)
            pd.process()

    window.close()
    exit(0)

if __name__ == '__main__':
    main()