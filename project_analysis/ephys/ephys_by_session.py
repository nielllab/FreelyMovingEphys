"""
ephys_by_session.py

analyze ephys for all recordings of a session
"""
import os

from uilt.paths import list_subdirs
from project_analysis.ephys.analyze_ephys import find_files, run_ephys_analysis

def session_ephys_analysis(config):
    # get options out
    data_path = config['animal_dir']
    unit = config['ephys_analysis']['unit_to_highlight']
    ch_count = config['ephys_analysis']['channel_count']
    mp4 = config['ephys_analysis']['write_videos']
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
            mp4 = False
            file_dict = find_files(recording_path, full_recording_name, fm, this_unit, stim_type, mp4, ch_count)
            run_ephys_analysis(file_dict)
        except Exception as e:
            print(e)