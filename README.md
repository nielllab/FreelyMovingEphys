# FreelyMovingEphys
Preprocessing and analysis for ephys experiements in freely moving mice.

## Setup
Instillation instructions assume that GPU drivers and CUDA are set up ahead of time.
### Installation of environment (Windows)
Navigate to the DeepLabCut repository installed locally. Run
```
cd conda-environments
conda env create -f DLC-GPU.yaml
```
In the FreelyMovingEphys repository, the requirements.txt file contains a list of packages required for the rest of the analysis not included with DLC. Instillation, assuming the working directory is `/FreelyMovingEphys/`:
```
pip install -r requirements.txt
```
The computer must have ffmpeg installed to flip and deinterlace videos. Instructions [here](https://video.stackexchange.com/questions/20495/how-do-i-set-up-and-use-ffmpeg-in-windows).
### Installation of environment (Ubuntu)
Starting from `/FreelyMovingEphys/` (i.e. the top of this repository):
```
conda create -n DLC-GPU2 python=3.7 tensorflow-gpu=1.13.1
pip install deeplabcut==2.2b8
conda install -c conda-forge wxpython
pip install -r requirements.txt
sudo apt install ffmpeg
```

## Preprocessing usage
### Ephys preprocessing
Open `preprocessEphysData.m` in Matlab. This script is in `/FreelyMovingEphys/matlab/`. Fill out desired settings, run the script using the function `applyCARtoDat_subset` if you want to read a single binary. Select the binary file when the dialog window opens. If you want to select multiple binary files and merge them, run `preprocessEphysData.m` using the function `applyCARtoDat_subset_multi`. Select the .bin files you want to merge in the dialog window, and select the folder to save the merged data.

Run Kilosort on the merged data.

Run Phy2 on the merged data.

In the `DLC-GPU2` environment and at the top of the repository, run `python -m split_ephys_recordings`. A dialog box will open in which the user selects the session .mat file to be split apart into seperate files for each recording.

### Main preprocessing
In the `DLC-GPU2` environment, run `python -m preprocessing` to open the preprocessing pipeline GUI. Fill out the fields to build a config file, write that .json, and then you'll be able to run the pipeline on your dataset. For GUI instructions, see [this](https://github.com/nielllab/FreelyMovingEphys/blob/master/docs/GUI_user_guide.md) user guide.

If you don't want to use the GUI, you can fill out a config file manually and run `python manual_preprocessing.py --config_path T:/path/to/config` in the `DLC-GPU2` environment. Replace the entry for the `--config_path` with the path to the config file Definitions for each of the fields are avalible [here](https://github.com/nielllab/FreelyMovingEphys/blob/master/docs/config_options.md).

## Analysis usage

Analysis scripts, which interpret the preprocessed data, are located in `/FreelyMovingEphys/project_analysis/`. Each project is intended to have a seperate directory within `/project_analysis/`.

### Minimal analysis for mapping of receptive fields
For immidiete mapping of receptive fields, you can run `python -m project_analysis.map_receptive_fields`. You'll be prompted to enter a directory in a dialog box. You should choose the recording directory for **white noise stimulus**. A .png figure of receptive fields will be saved into the directory of each mouse's white noise stimulus recording. This module needs to be run after spike sorting and the splitting of individual recordings, but it includes preprocessing for all necessary data inputs (i.e. do **not** run `python -m preprocessing` before running this).
### Full ephys analysis
To generate figures and videos from freely moving ephys experiments, run `python -m project_analysis.ephys` from the top of this repository. This will launch a GUI with a few options, and a run button that executes the ephys analysis.

To run ephys analysis without a gui, run `python manual_ephys_analysis.py --data_path T:/path/to/recording --rec_name 113020_G6H27P8LT_control_Rig2_fm1 --unit 0 --fm True --stim_type None`, replacing arguments with the information for the recording. Note that this is the **recording** directory (e.g. `hf1_wn`), not the **animal** directory. Mark `--fm` as `False` and choose one of `gratings`, `sparse_noise`, or `white_noise` for `--stim_type` if it is a head-fixed recording.

