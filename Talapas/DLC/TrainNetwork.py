import argparse, json, sys, os, subprocess, shutil
os.environ["DLClight"] = "True"
import deeplabcut


def pars_args():
    # get user inputs
    parser = argparse.ArgumentParser(description='deinterlace videos and adjust timestamps to match')
    parser.add_argument('--multi_animal', type=bool, default=False)
    parser.add_argument('--add_videos', type=bool, default=False)
    parser.add_argument('--create_dataset', type=bool, default=True)
    parser.add_argument('--config_path', type=str, default='~/config.yaml')
    parser.add_argument('--data_path', type=str, default='~/videos')
    args = parser.parse_args()
    return args

def get_snapshot_num(cfg,trainingsetindex=0,shuffle=1,modelprefix="", modelfolder=""):
    ind = cfg['snapshotindex']

    # Check which snapshots are available and sort them by # iterations
    try:
        Snapshots = np.array([fn.split(".")[0] for fn in os.listdir(os.path.normpath(os.path.join(modelfolder, "train"))) if "index" in fn])
    except FileNotFoundError:
        raise FileNotFoundError(
            "Snapshots not found! It seems the dataset for shuffle %s has not been trained/does not exist.\n Please train it before using it to analyze videos.\n Use the function 'train_network' to train the network for shuffle %s."
            % (shuffle, shuffle))

    increasing_indices = np.argsort([int(m.split("-")[1]) for m in Snapshots])
    Snapshots = Snapshots[increasing_indices]
    return Snapshots[ind]

def GetModelFolder(trainFraction, shuffle, cfg, modelprefix="", offset=1):
    Task = cfg["Task"]
    date = cfg["date"]
    iterate = "iteration-" + str(cfg["iteration"]-offset)
    return Path(
        modelprefix,
        "dlc-models/"
        + iterate
        + "/"
        + Task
        + date
        + "-trainset"
        + str(int(trainFraction * 100))
        + "shuffle"
        + str(shuffle),

if __name__ == '__main__':
    args = pars_args()
    path_config_file = os.path.expanduser(args.config_path)
    videofile_path = os.path.expanduser(args.data_path)

    if args.create_dataset:
        deeplabcut.create_training_dataset(path_config_file, augmenter_type='imgaug',windows2linux=True)
    # Set up pose_config.yaml
    cfg = deeplabcut.auxiliaryfunctions.read_config(path_config_file)
    # Add imageaug to pose_config.yaml 
    trainposeconfigfile,testposeconfigfile,snapshotfolder=deeplabcut.return_train_network_path(path_config_file)
    cfg_dlc=deeplabcut.auxiliaryfunctions.read_plainconfig(trainposeconfigfile)
    cfg_dlc['scale_jitter_lo']= 0.5
    cfg_dlc['scale_jitter_up']=1.5
    cfg_dlc['augmentationprobability']=.5
    cfg_dlc['batch_size']=3 #pick that as large as your GPU can handle it
    cfg_dlc['elastic_transform']=True
    cfg_dlc['rotation']=180
    cfg_dlc['covering']=True
    cfg_dlc['motion_blur'] = True
    cfg_dlc['optimizer'] ="adam"
    if args.multi_animal:
        cfg_dlc['dataset_type']='multi-animal-imagaug'
    else:
        cfg_dlc['dataset_type']='imgaug'
    
    cfg_dlc['multi_step'] = [[5.0e-4, 10000//cfg_dlc['batch_size']],[1.0e-4, 430000//cfg_dlc['batch_size']],[5.0e-5, 730000//cfg_dlc['batch_size']],[1.0e-5, 1030000//cfg_dlc['batch_size']]]
    # cfg_dlc['multi_step']=[[1e-4, 7500], [5.0e-5, 12000], [1e-5, 50000]]
    if args.retrain:
        modelfolder = os.path.join(cfg_dlc['project_path'],'dcl_models/iteration-{:d}/train/'.format(cfg['iteration']-1))
        snapshot = get_snapshot_num(cfg,trainingsetindex=cfg['iteration']-1)
        cfg_dlc['init_weights'] = os.path.join(modelfolders,'snapshot-{:d}'.format(snapshot))
    deeplabcut.auxiliaryfunctions.write_plainconfig(trainposeconfigfile,cfg_dlc)
    # Train and Evaluate Network
    deeplabcut.train_network(path_config_file, allow_growth=True, displayiters=100, saveiters=10000, maxiters=1030000//cfg_dlc['batch_size'])
    deeplabcut.evaluate_network(path_config_file, plotting=True, trainingsetindex='all')

    scorername = deeplabcut.analyze_videos(path_config_file, [videofile_path] ,save_as_csv=True)
    deeplabcut.filterpredictions(path_config_file, [videofile_path], videotype='avi')
    deeplabcut.create_labeled_video(path_config_file, [videofile_path], filtered = True )

