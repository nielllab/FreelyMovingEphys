import os
os.environ["DLClight"] = "True"
import deeplabcut

# use ephys1-gpu environment, not DLC-GPU1

config_path = '/home/niell_lab/Documents/deeplabcut_projects/WidefieldEyeCams-kris-2021-10-12/config.yaml'
batchsize = 14

deeplabcut.create_training_dataset(config_path, augmenter_type='imgaug')

trainposeconfigfile,testposeconfigfile,snapshotfolder=deeplabcut.return_train_network_path(config_path)
cfg_dlc=deeplabcut.auxiliaryfunctions.read_plainconfig(trainposeconfigfile)
cfg_dlc['scale_jitter_lo'] = 0.5
cfg_dlc['scale_jitter_up'] = 1.5
cfg_dlc['augmentationprobability'] = 0.5
cfg_dlc['batch_size'] = batchsize
cfg_dlc['elastic_transform'] = True
cfg_dlc['rotation'] = False
cfg_dlc['covering'] = True
cfg_dlc['motion_blur'] = True
cfg_dlc['optimizer'] = 'adam'
cfg_dlc['dataset_type'] = 'imgaug' # 'imagaug'multi-animal-
cfg_dlc['multi_step'] = [[1e-4, int(75000/batchsize)], [5.0e-5, int(120000/batchsize)], [1e-5, int(500000/batchsize)]]
deeplabcut.auxiliaryfunctions.write_plainconfig(trainposeconfigfile, cfg_dlc)

deeplabcut.train_network(config_path, allow_growth=True, displayiters=500, maxiters=50000, saveiters=5000)

deeplabcut.evaluate_network(config_path, plotting=True, trainingsetindex='all')