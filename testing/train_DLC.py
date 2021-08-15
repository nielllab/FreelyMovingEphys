import os
os.environ["DLClight"] = "True"
import deeplabcut

config_path = '/home/niell_lab/Documents/deeplabcut_projects/EphysEyeCams4-dylan-2021-08-09/config.yaml'

# deeplabcut.create_training_dataset(config_path, augmenter_type='imgaug')

# deeplabcut.train_network(config_path)

# deeplabcut.evaluate_network(config_path)