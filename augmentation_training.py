"""
augmentation_training.py

train a DeepLabCut network from labeled data
uses data augmentation

Oct. 11, 2020
"""

import deeplabcut 

path_config_file = '/home/dmartins/projects/FreelyMovingTOP_wGear-dylan-2020-10-08/config.yaml'

deeplabcut.cropimagesandlabels(path_config_file, size=(400, 400), userfeedback=False)
deeplabcut.create_training_dataset(path_config_file, augmenter_type='imgaug')

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
cfg_dlc['dataset_type']='imgaug'
cfg_dlc['multi_step']=[[1e-4, 7500], [5.0e-5, 12000], [1e-5, 50000]]

deeplabcut.auxiliaryfunctions.write_plainconfig(trainposeconfigfile,cfg_dlc)

deeplabcut.train_network(path_config_file, shuffle=1,displayiters=100,maxiters=50000,saveiters=10000)

deeplabcut.evaluate_network(path_config_file, trainingsetindex='all')


# videofile_path = ['/home/seuss/Desktop/ARCricket-Elliott-2020-10-04/videos/082620_J519LT_control_Trial1.1_TOP1.avi']#['/home/seuss/Desktop/ARCricket-Elliott-2020-10-04/videos']

# deeplabcut.analyze_videos(path_config_file, videofile_path,save_as_csv=True)
# deeplabcut.filterpredictions(path_config_file, videofile_path)
# deeplabcut.plot_trajectories(path_config_file, videofile_path, filtered = True)
# deeplabcut.create_labeled_video(path_config_file, videofile_path, filtered = True)