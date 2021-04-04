import argparse, json, sys, os, subprocess, shutil
os.environ["DLClight"] = "True"
import deeplabcut


def pars_args():
    # get user inputs
    parser = argparse.ArgumentParser(description='deinterlace videos and adjust timestamps to match')
    parser.add_argument('--multi_animal', type=bool, default=False)
    parser.add_argument('--add_videos', type=bool, default=False)
    parser.add_argument('--retrain', type=bool, default=False)
    parser.add_argument('--config_path', type=str, default='~/config.yaml')
    parser.add_argument('--data_path', type=str, default='~/videos')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = pars_args()
    path_config_file = os.path.expanduser(args.config_path)
    videofile_path = os.path.expanduser(args.data_path)

    
    deeplabcut.create_training_dataset(path_config_file, augmenter_type='imgaug',windows2linux=True)
    # Set up pose_config.yaml
    cfg = deeplabcut.auxiliaryfunctions.read_config(path_config_file)
    # Add imageaug to pose_config.yaml 
    trainposeconfigfile,testposeconfigfile,snapshotfolder=deeplabcut.return_train_network_path(path_config_file)
    cfg_dlc=deeplabcut.auxiliaryfunctions.read_plainconfig(trainposeconfigfile)
    cfg_dlc['scale_jitter_lo']= 0.5
    cfg_dlc['scale_jitter_up']=1.5
    cfg_dlc['augmentationprobability']=.5
    cfg_dlc['batch_size']=1 #pick that as large as your GPU can handle it
    cfg_dlc['elastic_transform']=True
    cfg_dlc['rotation']=180
    cfg_dlc['covering']=True
    cfg_dlc['motion_blur'] = True
    cfg_dlc['optimizer'] ="adam"
    if args.multi_animal:
        cfg_dlc['dataset_type']='multi-animal-imagaug'
    else:
        cfg_dlc['dataset_type']='imgaug'
        
    # cfg_dlc['multi_step']=[[1e-4, 7500], [5.0e-5, 12000], [1e-5, 50000]]
    deeplabcut.auxiliaryfunctions.write_plainconfig(trainposeconfigfile,cfg_dlc)
    # Train and Evaluate Network
    deeplabcut.train_network(path_config_file, allow_growth=True, displayiters=500, maxiters=100000, saveiters=10000)
    deeplabcut.evaluate_network(path_config_file, plotting=True, trainingsetindex='all')

    scorername = deeplabcut.analyze_videos(path_config_file, [videofile_path] ,save_as_csv=True)
    deeplabcut.filterpredictions(path_config_file, [videofile_path], videotype='avi')
    deeplabcut.create_labeled_video(path_config_file, [videofile_path], filtered = True )

