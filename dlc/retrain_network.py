"""
retrain_network.py

retrain an existing DeepLabCut network with new videos

Nov. 19, 2020
"""
import deeplabcut 
import glob, os
from tqdm import trange, tqdm
from tkinter import filedialog
import argparse
import tkinter as tk

def pars_args():
    # get user inputs
    parser = argparse.ArgumentParser(description='deinterlace videos and adjust timestamps to match')
    parser.add_argument('-ma','--multi_animal', type=bool, default=False)
    parser.add_argument('-n','--get_new_vids', type=bool, default=False)
    parser.add_argument('-r','--retrain', type=bool, default=True)
    parser.add_argument('-c','--path_config_file', type=str, default='~/Documents/deeplabcut_projects/EyeCamTesting-dylan-2020-07-07/config.yaml')
    parser.add_argument('-v','--videofile_path', type=str, default='~/Documents/deeplabcut_projects/EyeCamTesting-dylan-2020-07-07/videos')
    parser.add_argument('-ssn','--snapshotnum', type=int, default=50000)
    args = parser.parse_args()
    return args

def get_snapshot_num(cfg,trainingsetindex=0,shuffle=1,modelprefix=""):
    ind = cfg['snapshotindex']
    cfg = deeplabcut.auxiliaryfunctions.read_config(cfg)
    
    trainFraction = cfg["TrainingFracget_snapshotget_snapshot_numtion"][trainingsetindex]

    # Check which snapshots are available and sort them by # iterations
    try:
        Snapshots = np.array([fn.split(".")[0] for fn in os.listdir(os.path.join(modelfolder, "train")) if "index" in fn])
    except FileNotFoundError:
        raise FileNotFoundError(
            "Snapshots not found! It seems the dataset for shuffle %s has not been trained/does not exist.\n Please train it before using it to analyze videos.\n Use the function 'train_network' to train the network for shuffle %s."
            % (shuffle, shuffle))

    increasing_indices = np.argsort([int(m.split("-")[1]) for m in Snapshots])
    Snapshots = Snapshots[increasing_indices]
    return Snapshots[ind]

def main(args):
    path_config_file = os.path.normpath(os.path.expanduser(args.path_config_file))
    videofile_path = os.path.normpath(os.path.expanduser(args.videofile_path))
    
    # the first time this is run, do this
    if args.get_new_vids:
        root = tk.Tk()
        root.withdraw()

        new_vids = filedialog.askopenfilenames(parent=root,title='Choose a file')
        NewVids = list(new_vids)
        deeplabcut.add_new_videos(path_config_file,NewVids,copy_videos=True)
        deeplabcut.extract_frames(path_config_file, mode='automatic', algo='uniform', userfeedback=False, crop=False)
        deeplabcut.label_frames(path_config_file)

    # then, run the script again and do this
    if args.retrain:
        # deeplabcut.cropimagesandlabels(path_config_file, size=(400, 400), userfeedback=False)

        deeplabcut.merge_datasets(path_config_file) # ADD THIS BACK AT THE END

#     if args.mulit_animal:
#         deeplabcut.create_multianimaltraining_dataset(path_config_file)
#         # Set up pose_config.yaml
#         cfg = deeplabcut.auxiliaryfunctions.read_config(path_config_file)
#         # Add imageaug to pose_config.yaml 
#         trainposeconfigfile,testposeconfigfile,snapshotfolder=deeplabcut.return_train_network_path(path_config_file)
#         cfg_dlc=deeplabcut.auxiliaryfunctions.read_plainconfig(trainposeconfigfile)
#         cfg_dlc['scale_jitter_lo']= 0.5
#         cfg_dlc['scale_jitter_up']=1.5
#         cfg_dlc['augmentationprobability']=.5
#         cfg_dlc['batch_size']=8 #pick that as large as your GPU can handle it
#         cfg_dlc['elastic_transform']=True
#         cfg_dlc['rotation']=180
#         cfg_dlc['covering']=True
#         cfg_dlc['motion_blur'] = True
#         cfg_dlc['optimizer'] ="adam"
#         cfg_dlc['dataset_type']='multi-anmial-imgaug' 
#         cfg_dlc['multi_step']=[[1e-4, 7500], [5.0e-5, 12000], [1e-5, 50000]]
#         if retrain:
#             modelfolder = os.path.join(cfg_dlc['project_path'],'dcl_models/iteration-{:d}/train/'.format(cfg['iteration']-1)
#             snapshot = get_snapshot_num(cfg,trainingsetindex=cfg['iteration']-1)
#             cfg_dlc['init_weights'] = os.path.join(modelfolders,'snapshot-{:d}'.format(snapshot))
#         deeplabcut.auxiliaryfunctions.write_plainconfig(trainposeconfigfile,cfg_dlc)
        
#         # Train and Evaluate Network
#         # deeplabcut.train_network(path_config_file, allow_growth=True, displayiters=500, maxiters=100000, saveiters=10000)
#         # deeplabcut.evaluate_network(path_config_file, plotting=True, trainingsetindex='all')
#         # deeplabcut.extract_save_all_maps(path_config_file, shuffle=1, Indices=[0])

#         # deeplabcut.evaluate_multianimal_crossvalidate(path_config_file, Shuffles=[1], edgewisecondition=True, leastbpts=1, init_points=20, n_iter=50, target='rpck_train')
        
#         # tracktype= 'box' #box, skeleton
#         # deeplabcut.convert_detections2tracklets(path_config_file, videofile_path, videotype='avi', track_method=tracktype, overwrite=True)
#         # pickles = sorted(list(glob.glob(os.path.join(videofile_path[0],'*bx.pickle'))),reverse=True) #sk
#         # print(pickles)

#         # pickles = ['/home/seuss/Desktop/NewPreyCapture/3DVids/082420_J519LT_control_Trial1.1_TOP1DLC_resnet50_preycapSep30shuffle1_50000_bx.pickle',
#         #             '/home/seuss/Desktop/NewPreyCapture/3DVids/082420_J519LT_control_Trial1.1_TOP3DLC_resnet50_preycapSep30shuffle1_50000_bx.pickle',
#         #             '/home/seuss/Desktop/NewPreyCapture/3DVids/082420_J519LT_control_Trial1.2_TOP1DLC_resnet50_preycapSep30shuffle1_50000_bx.pickle',
#         #             '/home/seuss/Desktop/NewPreyCapture/3DVids/082420_J519LT_control_Trial1.2_TOP3DLC_resnet50_preycapSep30shuffle1_50000_bx.pickle']
#         # for n in trange(len(pickles)):
#         #     deeplabcut.convert_raw_tracks_to_h5(path_config_file, pickles[n], min_tracklet_len=0)

#         # deeplabcut.filterpredictions(path_config_file, videofile_path,videotype='avi', track_method=tracktype)
#         # ## deeplabcut.plot_trajectories(path_config_file, videofile_path, filtered = True)
#         # deeplabcut.create_labeled_video(path_config_file, videofile_path, filtered = True, draw_skeleton=True, videotype='avi', track_method=tracktype )

#     else: 

        deeplabcut.create_training_dataset(path_config_file, augmenter_type='imgaug')
        # Set up pose_config.yaml
        cfg = deeplabcut.auxiliaryfunctions.read_config(path_config_file)
        # Add imageaug to pose_config.yaml 
        trainposeconfigfile,testposeconfigfile,snapshotfolder=deeplabcut.return_train_network_path(path_config_file)
        cfg_dlc=deeplabcut.auxiliaryfunctions.read_plainconfig(trainposeconfigfile)
        cfg_dlc['scale_jitter_lo']= 0.5
        cfg_dlc['scale_jitter_up']=1.5
        cfg_dlc['augmentationprobability']=.5
        cfg_dlc['batch_size']=8 #pick that as large as your GPU can handle it
        cfg_dlc['elastic_transform']=True
        cfg_dlc['rotation']=180
        cfg_dlc['covering']=True
        cfg_dlc['motion_blur'] = True
        cfg_dlc['optimizer'] ="adam"
        cfg_dlc['dataset_type']='imgaug' # 'imagaug'multi-animal-
        cfg_dlc['multi_step']=[[1e-4, 7500], [5.0e-5, 12000], [1e-5, 50000]]
        # if args.retrain:
            # modelfolder = os.path.normpath(os.path.join(cfg_dlc['project_path'],'dlc_models/iteration-{:d}/EyeCamTestingJul7-trainset95shuffle1/train/'.format(cfg['iteration']-1)))
            # # snapshot = get_snapshot_num(cfg,trainingsetindex=cfg['iteration']-1)
            # # cfg_dlc['init_weights'] = os.path.join(modelfolders,'snapshot-{:d}'.format(snapshot))
            # cfg_dlc['init_weights'] = os.path.normpath(os.path.join(modelfolder,'/snapshot-{}'.format(args.snapshotnum)))
        cfg_dlc['init_weights'] = 'C:/Users/Niell Lab/Documents/deeplabcut_projects/EyeCamTesting-dylan-2020-07-07/dlc-models/iteration-1/EyeCamTestingJul7-trainset95shuffle1/train/snapshot-50000'
        deeplabcut.auxiliaryfunctions.write_plainconfig(trainposeconfigfile,cfg_dlc)

#         # Train and Evaluate Network
        deeplabcut.train_network(path_config_file, allow_growth=True, displayiters=500, maxiters=100000, saveiters=10000)
        deeplabcut.evaluate_network(path_config_file, plotting=True, trainingsetindex='all')

#         # scorername = deeplabcut.analyze_videos(path_config_file, videofile_path,save_as_csv=True)
#         # deeplabcut.filterpredictions(path_config_file, videofile_path,videotype='avi')
#         # deeplabcut.create_labeled_video(path_config_file, videofile_path, filtered = True )
#         # deeplabcut.create_video_with_all_detections(path_config_file, videofile_path, scorername)


#     #### Need to add 3d parsing

#     # 3D Triangluate
#     # from triangulate_multi2 import triangulate

#     # config_path = "/home/seuss/Desktop/NewPreyCapture/3DPreyCapture-Elliott-2020-09-20-3d/config.yaml"
#     # # h5_files = sorted(list(glob.glob(os.path.join(videofile_path[0],'*bx.h5')))) #sk

#     # # h5_files = [
#     # #     '/home/seuss/Desktop/NewPreyCapture/MathisNetwork2/videos/082620_J519LT_control_Trial1.1_TOP1DLC_resnet50_preycapSep30shuffle1_50000_bx.h5',
#     # #     '/home/seuss/Desktop/NewPreyCapture/MathisNetwork2/videos/082620_J519LT_control_Trial1.1_TOP3DLC_resnet50_preycapSep30shuffle1_50000_bx.h5',
#     # # ]
#     # video_path = ['/home/seuss/Desktop/NewPreyCapture/MathisNetwork2/videos/']#
#     # camera_pair = "TOP1-TOP3"
#     # triangulate(config_path, video_path[0], videotype="avi",save_as_csv=True)

if __name__ == '__main__':
    args = pars_args()
    main(args)