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
Open `preprocessEphysData.m` in Matlab.

Fill out desired settings, run the script, select the .bin files you want to merge, select the folder to save the merged data.

Run Kilosort on the merged data.

Run Phy2 on the merged data.

In the `DLC-GPU2` environment and at the top of the repository, run `python -m split_ephys_recordings`. A dialog box will open in which the user selects the session .mat file to be split apart into seperate files for each recording.

### Main preprocessing
In the `DLC-GPU2` environment, run `python -m preprocessing` to open the preprocessing pipeline GUI. Fill out the fields to build a config file, write that .json, and then you'll be able to run the pipeline on your dataset. For GUI instructions, see [this](https://github.com/nielllab/FreelyMovingEphys/blob/master/docs/GUI_user_guide.md) user guide.

If you don't want to use the GUI, you can fill out a config file manually and run `python preprocessing.manual_preprocessing.py` in the `DLC-GPU2` environment. Definitions for each of the fields are avalible [here](https://github.com/nielllab/FreelyMovingEphys/blob/master/docs/config_options.md).

## Analysis usage

Analysis scripts, which interpret the preprocessed data, are located in `/FreelyMovingEphys/project_analysis/`. Each project is intended to have a seperate directory within `/project_analysis/`.

### Ephys
To generate figures and videos from freely moving ephys experiments, run `python -m project_analysis.ephys` from the top of this repository. This will launch a GUI with a few options, and a run button that executes the ephys analysis.

