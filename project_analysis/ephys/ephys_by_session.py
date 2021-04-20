"""
ephys_by_session.py

analyze ephys for all recordings of a session
"""
import os

from util.paths import list_subdirs
from project_analysis.ephys.analyze_ephys import find_files, run_ephys_analysis

def session_ephys_analysis(config):
    # get options out
    data_path = config['data_path']
    unit = config['unit2highlight']
    probe_name = config['probe']
    mp4 = config['write_ephys_vids']
    # get subdirectories (i.e. name of each recording for this session)
    dirnames = list_subdirs(data_path)
    recording_names = sorted([i for i in dirnames if 'hf' in i or 'fm' in i])
    # iterate through each recording's name
    for recording_name in recording_names:
        try:
            print('starting ephys analysis for '+recording_name)
            if 'fm' in recording_name:
                fm = True
            elif 'fm' not in recording_name:
                fm = False
            this_unit = int(unit)
            if fm == True:
                stim_type = 'None'
            elif 'wn' in recording_name:
                stim_type = 'white_noise'
            elif 'grat' in recording_name:
                stim_type = 'gratings'
            elif 'noise' in recording_name:
                stim_type = 'sparse_noise'
            elif 'revchecker' in recording_name:
                stim_type = 'revchecker'
            recording_path = os.path.join(data_path, recording_name)
            full_recording_name = '_'.join(recording_path.split(os.sep)[-3:-1])+'_control_Rig2_'+recording_path.split(os.sep)[-1]
            if 'spotsub' in full_recording_name:
                full_recording_name = full_recording_name.replace('spotsub', '')
            mp4 = False
            file_dict = find_files(recording_path, full_recording_name, fm, this_unit, stim_type, mp4, probe_name)
            run_ephys_analysis(file_dict)
        except Exception as e:
            print(e)