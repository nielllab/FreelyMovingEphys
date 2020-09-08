# FreelyMovingEphys

## Setup

# instillation of environments
Install the analysis pipeliene's conda environment from `/FreelyMovingEphys/conda_env/` with the terminal command `conda env create -f /path/to/repository/FreelyMovingEphys/conda_env/environment.yml`. This is used for all deinterlacing before DeepLabCut analyzes new videos, and all analysis after DeepLabCut or Anipose.

Then, install the environment DLC-GPU in order to analyze new videos, instructions [here](https://github.com/DeepLabCut/DeepLabCut/blob/master/conda-environments/README.md) To allow for three-camera configurations of the topdown view, it will be necessary to install Anipose in the DLC-GPU environment, instructions [here](https://github.com/lambdaloop/anipose/blob/master/docs/sphinx/installation.rst).

## Usage

### deinterlacing videos and interpolating timestamps
Before running any analysis, any 30fps eye or world .avi videos must be deinterlaced to bring them to 60fps and corresponding .csv timestamps  must be interpolated over. To accomplish this, run `deinterlace_for_dlc.py` on a parent directory containing all .avi files, .csv files, and .txt files of notes for a given experiment. The `deinterlace_for_dlc.py` script should be given the nested folders that contain videos and timestamps that need not be deinterlaced (frame rates will be checked and files that do not need to be changed will be copied to the output directory to keep experiments together).

Example terminal command: `python deinterlace_for_dlc.py '/path/to/top/of/parent/directory/' '/save/location/'`. Run `python deinterlace_for_dlc.py -h` for information about arguments that can be passed in.

### running DeepLabCut on new videos
To analyze new videos using DeepLabCut and/or Anipose, the script `analyze_new_vids.py` should be run through the terminal.

### analyzing DeepLabCut outputs
To process DeepLabCut outputs, visualize points, and get out calculations of head angle, eye ellipse parameters, pupil rotation, etc., use the jupyter notebook `test_extract_params_from_dlc.ipynb` in which parameters are set by hand in user input cells and each topdown camera, eye camera, and world camera is analyzed separately and independently from one another (with the exception of worldcam analysis which requires the corresponding eye to have been run beforehand). Alternatively, batch analysis can be done with the script `extract_params_from_dlc.py` with a terminal interface. An example terminal command could be `python -W ignore extract_params_from_dlc.py -c '/path/to/pipeline_config.json'` where the path is for a config file looking something like the example .json file below which will be loaded into the pipeline as a dictionary.

```
{
    "data_path": "/path/to/parent/directory/",
    "save_path": "/save/path/",
    "camera_names": [
        "TOP",
        "REYE",
        "RWORLD"
    ],
    "lik_thresh": 0.99,
    "coord_correction": 0,
    "cricket": true,
    "tear": true,
    "pxl_thresh": 50,
    "ell_thresh": 0.9,
    "save_vids": true,
    "save_figs": true,
    "use_BonsaiTS": true,
    "range_radius": 10,
    "world_interp_method": "linear"
}
```

This example would be for a set of experiments that contained top, right eye, and right world cameras and which has DeepLabCut .h5 files for each of these camera views. For `data_path`, subdirectories will be searched so this should only be the parent directory within which each trial has its own folder. For `camera_names`, list only the camera extensions that have .h5 files that came from DeepLabCut. Though there is build-in handling for `'TOP'`, `'LEYE'`, `'REYE'`, `'SIDE'`, `'LWORLD'`, and `'RWORLD'`, any camera name outside of this list can be read in and formatted into a .nc file. The analysis can then be done from the formatted data.

It's important that experiments use the following naming system so that they will be recognized: `013120_mouse1_control_TOP1.h5`. This would be for a topdown view of a mouse named `mouse1`, of experiment group `control`, and the camera `TOP1`.

### opening .nc files and visualizing outputs
Once the analysis is run, there will be plots and videos for each trial and one .nc file for each camera name which contains the data from all trials that had a camear of said name. Using only the .json config file's path, the outputs of the analysis pipeline can be found, viewed, and interacted with in the jupyter notebook `check_pipeline_outputs.ipynb`. There are also examples of way to index and select data from the loaded data structures.
