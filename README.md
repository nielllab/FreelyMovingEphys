# FreelyMovingEphys

## Setup

### installation of environment
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
### installation of ffmpeg
The machine must have ffmpeg installed to flip and deinterlace videos. Instructions [here](https://video.stackexchange.com/questions/20495/how-do-i-set-up-and-use-ffmpeg-in-windows).

## Usage

### data requirements
Directories of data should be structued like: `/date/session_or_depth/recording/`

### ephys preprocessing
Open `preprocessEphysData.m` in Matlab.
Fill out desired settings, run the script, select the .bin files you want to merge, select the folder to save the merged data.
Run Kilosort on the merged data.
Run Phy2 on the merged data.
In DLC-GPU2 environment and at the top of the repository, run `python format_multi_ephys.py` with a config file on the desktop filled out. Only `ephys_sample_rate` matters--the rest will be ignored. A dialog box will open when this is  run. Select the session .mat file.

### video preprocessing
A .json config file is used to run the analysis. An example with default values can be found in `FreelyMovingEphys/example_configs/preprocessing_config/`.
Save a copy of this .json in the directory of the data you want to run the analysis on.
Parameters used:
`data_path`: path to the parent directory of data, should be the date directory
`save_path`: save path for outputs of the analysis pipeline
`steps_to_run`: a dictionary of steps to analysis and whether or not to run those steps as `True`/`False` values
`cams`: a dictionary of camera views in the experiments and the .yaml DeepLabCut config file paths to be used to analyze the videos of each camera type; the capitalization of the camera name (i.e. LEYE vs LEye) must match the capitalization of the file itself
`flip_eye_during_deinter`: whether or not to flip the eye video verticlly during deinterlacing (eye video should be right-side-up)
`flip_world_during_deinter`: whether or not to flip the world video vertically during deinterlacing (world video should be right-side-up)
`crop_for_dlc`: whether or not to crop the videos down for DLC analysis
`multianimal_TOP`: whether or not the already trained TOP network to be used is a multianimal project (there are different functions to read in the TOP network dependeing on how the data are structured wihtin the .h5 output from DLC)
`lik_thresh`: threshold to set for pose likelihood
`has_ephys`: ephys should be split out and processed using the script `format_multi_ephys.py`, and not processed alongside videos; `has_ephys` should be `False`
`cricket`: whether or not there is a cricket in the experiments
`tear`: whether or not the outer point and tear duct of the eye was labeled in the eye videos
`pxl_thresh`: the maximum acceptable number of pixels for radius of the pupil
`ell_thresh`: the maximum ratio of ellipse shortaxis to longaxis
`eye_dist_thresh_cm`: max. acceptable distance from mean position of an eye point that any frame's position for that point can be (in cm)
`eyecam_pxl_per_cm`: scale factor for camera from pixels to cm on eye
`save_avi_vids`: whether or not to save out videos with parameters plotted on them
`num_save_frames`: number of video frames to write to file with paramters plotted on them
`save_nc_vids`: whether or not to save compressed videos into nc files along with the data
`save_figs`: whether or not to save out figures
`use_BonsaiTS`: whether to use Bonsai timestamps for Flir timestamps, where `true` would have it use Bonsai timestamps
`range_radius`: the threshold to set for range in radius of the pupil to be used to find pupil rotation
`world_interp_method`: the interpolation method to use for interpolating over eye timestmaps with world timestamps
`num_ellipse_pts_needed`: the number of 'good' eye points required before an ellipse fit will be done on a frame
`dwnsmpl`: factor by which to downsample videos before frames are added to an xarray data structure
`ephys_sample_rate`: sample rate of ephys aquisition
`optical_mouse_screen_center`: dictionary of x and y centers of screen that the optical mouse is reset to
`optical_mouse_pix2cm`: scale factor for optical mouse pixels to cm
`optical_mouse_sample_rate_ms`: optical mouse sample rate in ms
`run_pupil_rotation`: whether or not to analyze eye videos for pupil rotation
`run_top_angles`: whether or not to get TOP head and body angles; turn off when tracking quality is poor
`run_with_form_time`: this changes how strictly nomenclature must be followed for post-deinterlacing files; if files are *not* going to have 'deinter' or 'formatted' in the names of .avi and .csv files, then `run_with_form_time` should be `False`, but most often, if you're running the pipeline start to finish, it should be `True`.

Then, run:
```
conda activate DLC-GPU2
python preprocessing.py
```
A dialog box will be created. Pick the .json config file that you want to read in.