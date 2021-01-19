# Config Options
Listed below are the defineitions used in the preprocessing config.

## Options

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