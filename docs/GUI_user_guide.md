# FreelyMovingEphys - Preprocessing GUI
User guide for the FreelyMovingEphys preprocessing pipeline's GUI. Draft: DMM, Jan. 18, 2021

## Launching the GUI
In the `DLC-GPU2` conda environment, run `python -m preprocessing`

## Loading an Existing Config File
If a config file already exists and you don't want to change any options prior to running the pipeline, click browse on the 'Welcome' page and select the .json file.

Once the .json file path is selected, all tabs between 'Welcome' and 'Run!' should become unavalible. Enter the 'Run!' tab and click run.

## Starting Without an Existing Config File

### 1. Select Animal Path
Pick the directory ending in the animal's name. Data will be read from the subdirectories startinga at this point. This directory is also where the config file will be written.

### 2. Deinterlace
Check the first box if you want to deinterlace eye and world videos. Once checked, two more boxes will appear. They flip the eye and world videos verticlly when selected. (NOTE: current eye and world cameras are mounted upside-down, so they must be flipped for DLC to track them and paramters to be calculated correctly.)

### 3. Calibrate

**Option A:** There are no existing .npz files.

You'll have to create .npz files using checkerboard videos of the top and world cameras. Check the box to 'get paramters from checkerboard videos.' Then, select the paths to (1) the top camera checkerboard video, (2) the world camera checkerboard video, and (3) the directory into which you'd like to save the calibration parameters. You'll also have to choose names for the top and world .npz files which will be saved in the directory in step (3) above. There are fine default names there already in those fields, though. If you want to check anything, you can print out the default options that will be used if you enter nothing (these are read in from the default config in `/FreelyMovingEphys/example_configs/preprocessing_config.json`), and you can print out what things are currently set to, which will update as browse buttons are clicked. These will be printed in the terminal window that is running the GUI.

Criticlly, you will also need to check the very last checkbox in this tab, 'use parameters saved out during this session,' so that the pipeline knows to use the save paths entered above as the read paths as well. This will allow videos to be undistorted using the paramters right after those parameters are written to file.

**Option B:** There are existing .npz files already written.

In this case, you just need to click 'undistort top and world videos using existing calibration parameters,' and pick (1) the path to the top .npz and (2) the path to teh world .npz. You can print default and current selections out here too, like you might have with Option A.

### 4. Pose Estimation

Now, you can get it set up for DeepLabCut. For each camera, pick the camera name from the dropdown menu, and then browse for the associated .yaml config path. You can choose to crop videos before running DLC, and you can also choose to treat the top camera as a multianimal project. Neither of these last two option are really needed for our current setup.

If you're not running pose estimation, just don't add any cameras and DLC won't be run.

### 5. Parameters

There are a lot of options on this page, and they won't change often. If you want to get .nc files out, start by checking the 'get parameters' box in the top right-hand corner. With the current eye network that we use, you'll need to check the box on the left side labeled 'is the tear duct labeled in this eye network.' All the eye, top, and misc. options on the left side of the window are pretty standard and don't need to be changed (aside from that one already mentioned which manages the tear duct points). Information about what each of the paramters means is described in `README.md`, but the labels in the GUI interface should be pretty clear too.

The right side of that page is more about run options than data-management. There are execution options, save options, and cameras. Eye rotation is quite slow to run, so it's an optional step. Topdown head and body angles aren't going to be useful unless the tracking in the top view is consistant, so it's optional as well. We usually resize videos before storing them in the .nc file by a factor of 0.5, and this can be changed. You should **always** choose to save out diagnostic figures (this is admittedly a silly thing to make optional). Saving out diagnostic .avi video is a really good idea, and you can choose how many frames to save out. Saving 3600 frames is usually enough (that's one minute), but doing more won't take up too much time. Saving videos in .nc files will let you run later analysis, so you should chech this box too.

Lastly, you need to select all the cameras you want to run preprocessing on. This should match pose estimation cameras exactly, if you're running that step too. The order of cameras doesn't matter, just that the same cameras are present in both tabs.

### 6. Extra Parameters

If you want to track an IR LED for eye-world calibration, that happens on this tab. Select the box to track IR LEDs, and then browse for the world and eye IR LED DLC .yaml config files (these are **not** the regular DLC networks, these are specificlly for tracking LEDs in a dark room). You also need to enter the directory name for the LED recording (e.g. `hf3_IRspot`) and the likelihood threshold to use, which should be more strict than the one used for tracking mouse points (we don't need to include all points for this analysis, it's more important to include best points).

### 7. Write Config

Now you can click the button to write the config. The config file that's written will be printed in the terminal as a dictionary. If it doesn't look correct, you can go back and change things without restarting the GUI before clicking 'write' again.

### 8. Run!

Now you can click this to read the .json config that was writen to file and run all of the preprocessing. You need to keep the GUI window open while the code runs. You'll get progress updates in the terminal.