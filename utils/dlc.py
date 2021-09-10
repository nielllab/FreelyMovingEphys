"""
dlc.py
"""
import os
os.environ["DLClight"] = "True"
import deeplabcut

from utils.paths import find

def runDLCbatch(vid_list, config_path, config):
    """ Given a list of videos, run them through the same DLC network.
    
    Parameters:
    vidlist (list or str): list of video filepaths, otherwise, a single video as a str
    config_path (str): path to DLC config.yaml
    config (dict): options dictionary
    """
    if isinstance(vid_list, str):
        vid_list = [vid_list]
    for vid in vid_list:
        if config['pose_estimation']['crop_for_dlc'] is True:
            deeplabcut.cropimagesandlabels(config_path, size=(400, 400), userfeedback=False)
        deeplabcut.analyze_videos(config_path, [vid])
        if config['pose_estimation']['filter_dlc_predictions'] is True:
            deeplabcut.filterpredictions(config_path, vid)

def run_DLC_analysis(config):
    """ Find files and organize them by which DLC config file they are assocaited with.
    
    Parameters:
    config (dict): options dictionary
    """
    # get each camera type's entry
    for cam in config['pose_estimation']['projects']:
        # there's an entry for the name of the camera to be used
        cam_key = cam
        # and an entry for the config file for that camear type (this will be used by DLC)
        cam_config = config['pose_estimation']['projects'][cam_key]
        if cam_config != '' and cam_config != 'None' and cam_config != None:
            # if it's one of the cameras that needs to needs to be deinterlaced first, make sure and read in the deinterlaced 
            if any(cam_key in s for s in ['REYE','LEYE']):
                # find all the videos in the data directory that are from the current camera and are deinterlaced
                if config['parameters']['follow_strict_naming'] is True:
                    vids_this_cam = find('*'+cam_key+'*deinter.avi', config['animal_dir'])
                elif config['parameters']['follow_strict_naming'] is False:
                    vids_this_cam = find('*'+cam_key+'*.avi', config['animal_dir'])
                # remove unflipped videos generated during jumping analysis
                bad_vids = find('*'+cam_key+'*unflipped*.avi', config['animal_dir'])
                for x in bad_vids:
                    if x in vids_this_cam:
                        vids_this_cam.remove(x)
                ir_vids = find('*IR*.avi', config['animal_dir'])
                for x in ir_vids:
                    if x in vids_this_cam:
                        vids_this_cam.remove(x)
                print('found ' + str(len(vids_this_cam)) + ' deinterlaced videos from cam_key ' + cam_key)
                # warning for user if no videos found
                if len(vids_this_cam) == 0:
                    print('no ' + cam_key + ' videos found -- maybe the videos are not deinterlaced yet?')
            else:
                # find all the videos for camera types that don't neeed to be deinterlaced
                if config['parameters']['follow_strict_naming'] is True:
                    vids_this_cam = find('*'+cam_key+'*.avi', config['animal_dir'])
                elif config['parameters']['follow_strict_naming'] is False:
                    vids_this_cam = find('*'+cam_key+'*.avi', config['animal_dir'])
                print('found ' + str(len(vids_this_cam)) + ' videos from cam_key ' + cam_key)
            # analyze the videos with DeepLabCut
            # this gives the function a list of files that it will iterate over with the same DLC config file
            vids2run = [vid for vid in vids_this_cam if 'plot' not in vid]
            runDLCbatch(vids2run, cam_config, config)