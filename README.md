# FreelyMovingEphys

## Setup

### installation of environment
Create a DeepLabCut environment:
```
conda create -n DLC-GPU2 python=3.7 tensorflow-gpu=1.13.1
conda activate DLC-GPU2
pip install deeplabcut==2.2b8
pip install -U wxPython
```
In the FreelyMovingEphys repository, the requirements.txt file contains a list of packages required for the rest of the analysis not included with DLC. Instillation, assuming the working directory is `/FreelyMovingEphys/`:
```
cd env
pip install -r requirements.txt
```
The machine must also have ffmpeg installed, instructions [here](https://www.wikihow.com/Install-FFmpeg-on-Windows).
Before running the pipeline, activate the environment with `conda activate DLC-GPU2`.

## Usage

### creating an analysis config file
A .json config file is used to run the analysis.

Parameters included:
`data_path`: path to the parent directory of data
`save_path`: save path for outputs of the analysis pipeline
`steps_to_run`: a dictionary of steps to analysis and whether or not to run those steps as `True`/`False` values
`cams`: a dictionary of camera views in the experiments and the .yaml DeepLabCut config file paths to be used to analyze the videos of each camera type
`flip_eye_during_deinter`: whether or not to flip the eye video verticlly during deinterlacing
`crop_for_dlc`: whether or not to crop the videos down for DLC analysis
`multianimal_TOP`: whether or not the already trained TOP network to be used is a multianimal project (it needs to know what format the files will be in)
`lik_thresh`: threshold to set for pose likelihood
`cricket`: whether or not there is a cricket in the experiments
`tear`: whether or not the outer point and tear duct of the eye was labeled in the eye videos
`pxl_thresh`: the maximum acceptable number of pixels for radius of the pupil
`ell_thresh`: the maximum ratio of ellipse shortaxis to longaxis
`save_vids`: whether or not to save out videos
`save_figs`: whether or not to save out figures
`use_BonsaiTS`: whether to use Bonsai timestamps for Flir timestamps, where `true` would have it use Bonsai timestamps
`range_radius`: the threshold to set for range in radius of the pupil to be used to find pupil rotation
`world_interp_method`: the interpolation method to use for interpolating over eye timestmaps with world timestamps
`num_ellipse_pts_needed`: the number of 'good' eye points required before an ellipse fit will be done on a frame
`dwnsmpl`: factor by which to downsample videos before frames are added to an xarray data structure
`ephys_sample_rate`: sample rate of ephys aquisition
`run_pupil_rotation`: whether or not to analyze eye videos for pupil rotation

An example analysis config file (`/FreelyMovingEphys/example_configs/Example_json.json`) is reproduced below:
```
{
    "data_path": "//new-monster/T/freely_moving_ephys/ephys_recordings/101120/G6H28P6LT/",
    "save_path": "//new-monster/T/freely_moving_ephys/ephys_recordings/101120/G6H28P6LT/",
    "steps_to_run":{
        "deinter": true,
        "dlc": true,
        "params": true
    },
    "cams": {
        "REYE": "C:/Users/Niell Lab/Documents/trained_DLC_projects/EyeCamTesting-dylan-2020-07-07/config.yaml",
        "TOP1": "C:/Users/Niell Lab/Documents/trained_DLC_projects/FreelyMovingTOP_wGear-dylan-2020-10-08/config.yaml",
        "TOP2": "C:/Users/Niell Lab/Documents/trained_DLC_projects/FreelyMovingTOP_wGear-dylan-2020-10-08/config.yaml",
        "TOP3": "C:/Users/Niell Lab/Documents/trained_DLC_projects/FreelyMovingTOP_wGear-dylan-2020-10-08/config.yaml"
    },
    "flip_eye_during_deinter": true,
    "crop_for_dlc": true,
    "multianimal_TOP": false,
    "lik_thresh": 0.99,
    "has_ephys": true,
    "cricket": true,
    "tear": true,
    "pxl_thresh": 50,
    "ell_thresh": 0.9,
    "save_vids": true,
    "save_figs": true,
    "use_BonsaiTS": true,
    "range_radius": 10,
    "world_interp_method": "linear",
    "num_ellipse_pts_needed": 8,
    "dwnsmpl": 0.5,
    "ephys_sample_rate": 30000,
    "run_pupil_rotation": false
}
```

### freely_moving.py
One script runs every step of the analysis, or any subset of steps, `freely_moving.py`. Assuming the default path to the .json config file (`'~/Desktop/FreelyMovingData/Example_json.json'`) is correct, run:
```
conda activate DLC-GPU2
python freely_moving.py
```
If the path to the config file is changed, change this to
```
python freely_moving.py -c \Documents\config.json
```
for wherever the .json is saved.
