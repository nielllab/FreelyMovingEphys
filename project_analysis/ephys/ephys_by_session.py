"""
ephys_by_session.py
"""
import os
import traceback

from util.paths import list_subdirs
from project_analysis.ephys.analyze_ephys import find_files, run_ephys_analysis

def session_ephys_analysis(config):
    # get options out
    data_path = config['animal_dir']
    unit = config['ephys_analysis']['unit_to_highlight']
    probe_name = config['ephys_analysis']['probe_type']
    # get subdirectories (i.e. name of each recording for this session)
    dirnames = list_subdirs(data_path)
    recording_names = sorted([i for i in dirnames if 'hf' in i or 'fm' in i])
    if config['ephys_analysis']['recording_list'] != []:
        recording_names = [i for i in recording_names if i in config['ephys_analysis']['recording_list']]
    # iterate through each recording's name
    for recording_name in recording_names:
        try:
            print('starting ephys analysis for',recording_name,'in path',data_path)
            if 'fm' in recording_name:
                fm = True
            elif 'fm' not in recording_name:
                fm = False
            this_unit = int(unit)
            if fm == True and 'light' in recording_name:
                stim_type = 'light_arena'
            elif fm == True and 'dark' in recording_name:
                stim_type = 'dark_arena'
            elif fm == True and 'light' not in recording_name and 'dark' not in recording_name:
                stim_type = 'light'
            elif 'wn' in recording_name:
                stim_type = 'white_noise'
            elif 'grat' in recording_name:
                stim_type = 'gratings'
            elif 'noise' in recording_name:
                stim_type = 'sparse_noise'
            elif 'revchecker' in recording_name:
                stim_type = 'revchecker'
            recording_path = os.path.join(data_path, recording_name)
            norm_recording_path = os.path.normpath(recording_path).replace('\\', '/')
            full_recording_name = '_'.join(norm_recording_path.split('/')[-3:-1])+'_control_Rig2_'+os.path.split(norm_recording_path)[1].split('/')[-1]
            mp4 = config['ephys_analysis']['write_videos']
            file_dict = find_files(recording_path, full_recording_name, fm, this_unit, stim_type, mp4, probe_name)
            run_ephys_analysis(file_dict)
        except Exception as e:
            print(traceback.format_exc())