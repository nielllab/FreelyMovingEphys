# FreelyMovingEphys

## Setup
### System requirements
The pipeline requires a GPU with ~11GB memory to run the pose estimation components. The pipeline makes use of multiprocessing, so a greater number of CPU cores avalible will improve the speed of the pipeline.
### Installation of environment (Windows)
Instillation instructions assume that GPU drivers and CUDA are set up ahead of time.

Navigate to the DeepLabCut repository installed locally. Run
```
cd conda-environments
conda env create -f DLC-GPU.yaml
```
In the FreelyMovingEphys repository, the requirements.txt file contains a list of packages required for the rest of the analysis not included with DLC. Instillation, assuming the working directory is `/FreelyMovingEphys/`:
```
cd env
pip install -r requirements.txt
```
The computer must have ffmpeg installed to flip and deinterlace videos. Instructions [here](https://video.stackexchange.com/questions/20495/how-do-i-set-up-and-use-ffmpeg-in-windows).
### Installation of environment (Ubuntu)
Instillation instructions assume that GPU drivers and CUDA are set up ahead of time.

Starting from `/FreelyMovingEphys/` (i.e. the top of this repository):
```
conda create -n DLC-GPU2 python=3.7 tensorflow-gpu=1.13.1
pip install deeplabcut==2.2b8
conda install -c conda-forge wxpython
cd env
pip install -r requirements.txt
sudo apt install ffmpeg
```

## Preprocessing usage

### Functionality
The preprocessing pipeline begins by deinterlacing worldcam videos and eyecam videos, which are recorded in an interlaced 30fps format. This step also flips the world and eye camera so that they are right-side-up (they're recorded upside-down). Calibration parameters generated from checkerboard videos are created and saved out. Worldcam and topdown cameras are then calibrated using those paramters. Note that getting camera calibration paramters does not need to be run each time a recording is preprocessed; the parameters can be reused between sessions. Next, the pipeline will run DeepLabCut to generate pose estimations for the topdown view and the eye views. Most of the analysis (and most of the time spent running) is in the paramters-generating step in which topdown cameras are thresholded for likelihood, head and body angle are calculated, a short .avi of points plotted on the video is saved out, and a .nc file is saved out storing points, paramters, and a downsampled `int8` copy of the entire video; the eight points around each of the mouse's eyes are read in, threholded by likelihood, an elipse is fit to the points for all frames, the paramters of theta, phi, and omega are calculated (horizonal, vertical, and rotational paramters, respectivly) where omega is an optional parameter (omega is a slow step), and the eyecam video is stored with these points and paramters in an .nc file; worldcam video is read in and formatted into an .nc file; sidecameras are read in, points are threhsolded, head pitch will be calculated, and all side paramters and video will be saved into an .nc file. Lastly, the mouse's running speed on the ball (when head-fixed) will be recorded and converted to cm/s with timestamps, and the IMU will be read in from a binary and converted to -5V to +5V with timestamps. If any of these data inputs are missing, the preprocesing will continue without it. Useful .pdf files of diagnostic figres will be saved out for each of the camera inputs. Each of the listed steps can be run in a modular fashion, so that they can be run in stages, or entire sections can be skipped (e.g. undistortion of recordings is optional (though the `run_with_form_time` config option must be set to `False` in this case)). Additional paramters (in the config file, `addtl_params`) can be run as the last step of the pipeline. Presently, this only includes IR LED tracking and figures to later be used in world-eye correction.

### Data requirements
Directories of data should be structued as: `/date/mouse_and_session_or_depth/recording/` (e.g. `/01012021/G123P4RT/fm1/`).

### Ephys preprocessing
Open `preprocessEphysData.m` in Matlab.

Fill out desired settings, run the script, select the .bin files you want to merge, select the folder to save the merged data.

Run Kilosort on the merged data.

Run Phy2 on the merged data.

In DLC-GPU2 environment and at the top of the repository, run `python -m split_ephys_recordings`. A dialog box will open in which the user selects the session .mat file to be split apart into seperate files for each recording.

### Main preprocessing
A .json config file is used to run the analysis. An example with default values can be found in `/FreelyMovingEphys/example_configs/preprocessing_config.json`. Save a copy of this config file in the directory of the data you want to run the analysis on, and change needed parameters. Any parameters not provided in the trial config file will be copied from this default config file in the repository.

`data_path`: path to the parent directory of data, should be the animal directory

`steps_to_run`: a dictionary of steps to analysis and whether or not to run those steps as `True`/`False` values. `deinter` deinterlaces all interlaced videos in the recording directories. `get_cam_calibration_params` uses the .checkerboard avi video paths in the `calibration` dictionary in the config file to find distortions in the camera. This will save out the paramters as a .npz file to the path named in that same `calibration` dictionary below. This is done for top and world cameras. The user doesn't need to run `get_cam_calibration_params` every time that the pipeline is run, since the paramters can be reused for a camera. `undistort recordings` makes use of the .npz file of distortion paramters and corrects distortions in world and top videos.

`cams`: a dictionary of camera views in the experiments and the .yaml DeepLabCut config file paths to be used to analyze the videos of each camera type; the capitalization of the camera name (i.e. LEYE vs LEye) must match the capitalization of the files exactly. Any combination of capitalization should be fine. Topdown cameras, eye, world, and side cameras are all acceptable.

`calibration`: a dictionary with information about calibrating world and topdown cameras, but also about tracking the IR LEDs for use in calibration. `eye_LED_config` should be the DeepLabCut .yaml config file path to track the reflection of the IR LED on the mouse pupil. `world_LED_config` is the .yaml path for tracking the IR LED in the worldcam. `world_checker_vid` should be the worldcam checkerboard .avi video that will be used to generate the .npz of calibration paramters in the event that `get_cam_calibration_params` is `True` in `steps_to_run`. `world_checker_npz` is treated as the save path for the .npz of worldcam calibration paramters if `get_cam_calibration_params` is `True`, but the path to read the .npz from when `undistort_recording` is True. It will overwrite existing .npz files. `top_checker_vid` and `top_checker_npz` ar ethe equivilent paths for the topdown camera.

`LED_dir_name` is the name of the directory in which the IR LED recordings are saved and should be read from. If the user is not running `addtl_params`, this will be ignored.

`flip_eye_during_deinter`: whether or not to flip the eye video verticlly during deinterlacing (eye video must be right-side-up, or DLC tracking will work poorly)

`flip_world_during_deinter`: whether or not to flip the world video vertically during deinterlacing (world video should be right-side-up)

`crop_for_dlc`: whether or not to crop the videos down for DLC analysis

`multianimal_TOP`: whether or not the already trained TOP network to be used is a multianimal project (there are different functions to read in the TOP network dependeing on how the data are structured wihtin the .h5 output from DLC)

`lik_thresh`: threshold to set for pose likelihood

`lik_thresh_strict`: this is only used for IR LED tracking at the moment. It's a more strict and less inclusive threshold for DLC likleihood values.

`has_ephys`: whether or not to run individual ephys analysis for a recording (ephys should be split out and processed using the script `format_multi_ephys.py`, and not processed alongside videos, so `has_ephys` should always be `False`)

`cricket`: whether or not there is a cricket to track in the experiments

`tear`: whether or not the outer point and tear duct of the eye was labeled in the eye videos\

`pxl_thresh`: the maximum acceptable number of pixels for radius of the pupil

`ell_thresh`: the maximum ratio of ellipse shortaxis to longaxis during ellipse fit of pupil

`eye_dist_thresh_cm`: max. acceptable distance from mean position of an eye point that any frame's position for that point can be (in cm)

`eyecam_pxl_per_cm`: scale factor for camera from pixels to cm on eye

`save_avi_vids`: whether or not to save out videos with parameters plotted on them

`num_save_frames`: number of video frames to write to file with parameters plotted on them

`save_nc_vids`: whether or not to save compressed videos into .nc files along with the data

`save_figs`: whether or not to save out figures

`use_BonsaiTS`: whether to use Bonsai timestamps for Flir timestamps, where `True` would have it use Bonsai timestamps

`range_radius`: the threshold to set for range in radius of the pupil to be used to find pupil rotation

`world_interp_method`: the interpolation method to use for interpolating over eye timestmaps with world timestamps

`num_ellipse_pts_needed`: the number of 'good' eye points required before an ellipse fit will be done on a frame

`dwnsmpl`: factor by which to downsample videos before frames are added to an xarray data structure

`ephys_sample_rate`: sample rate of ephys aquisition

`optical_mouse_screen_center`: dictionary of x and y centers of screen that the optical mouse is reset to

`optical_mouse_pix2cm`: scale factor for optical mouse pixels to cm

`optical_mouse_sample_rate_ms`: optical mouse sample rate in ms

`imu_sample_rate`: sample rate for IMU

`imu_downsample` factor to downsample IMU data by

`run_pupil_rotation`: whether or not to analyze eye videos for pupil rotation

`run_top_angles`: whether or not to get TOP head and body angles; turn off when tracking quality is poor

`run_with_form_time`: this changes how strictly nomenclature must be followed for post-deinterlacing files; if files are *not* going to have 'deinter'/'calib' or 'formatted' in the names of .avi and .csv files, then `run_with_form_time` should be `False`, but most often, if you're running the pipeline start to finish, it should be `True`.

### Running preprocessing
Once these parameters are set, run `python preprocessing.py`. Most often, a dialog box will open in which the user selects the .json config file to be read in. If the dialog box cannot be opened, the user will be prompted to enter the path to the config file in the terminal. User should activate the conda environment with `conda activate DLC-GPU2` before running the python script. Note that on lab computers, the environment is named `DLC-GPU2` not `DLC-GPU`.

## Analysis usage

Analysis scripts, which interpret the preprocessed data, are located in `/FreelyMovingEphys/project_analysis/`. Each project is intended to have a seperate directory within `/project_analysis/`.

### Ephys
To generate figures and videos from freely moving ephys experiments, run `python -m project_analysis.ephys.ephys_analysis` from the top of this repository.


### Jumping
Jumping analysis for eyecam data happens accross a few scripts.

1. Flip eyecams so that they're right-side-up: `python -m project_analysis.jumping.flip_jump_clips`

2. 

3. Get jump metadata and s

X. To analyze the preprocessed data: `python -m project_analysis.jumping.jump_analysis`. A dialog box will open. Select the jumping analysis config file directing the code to the data directory to analyze.

