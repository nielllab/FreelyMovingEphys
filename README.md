# FreelyMovingEphys

## Usage

### Deinterlacing videos
In directory deinter where bash script `ffmpeg-batch.sh` is located, run `bash ffmpeg-batch.sh avi avi /source/path/here/ /save/path/here/` where paths are replaced with locations of raw videos and 
Note: if the source path and save path are the same, the videos will be overwritten (i.e. there is no renaming functionality to the script).

### Running DeepLabCut
In directory dlc, open script `run_dlc_on_vids.ipynb` and in the user inputs code cell, enter a path to the DLC config file for the trained data, the path to the videos to analyze formatted for a glob function (e.g. `/path/to/data/*deinter.avi`), the path into which .h5 files should be saved. Run the block below, in which the DLC function analyzes new videos. **Note: This must be run using the `DLC-GPU` conda environment.** For instructions on installing `DLC-GPU` environment, see [this](https://github.com/DeepLabCut/DeepLabCut/blob/master/conda-environments/README.md) GitHub link.

### Analyzing DeepLabCut outputs
To clean up DeepLabCut outputs, visualize them, and get out calcualtions of head theta, eye theta and phi, and pupil rotation, use either the jupyter notebook `modular_dlc_intake.ipynb` in which parameters are set by hand in user input cells and each topdown camera, eye camera, and world camera is analyzed seperately and independently from one another (with the exceptin of worldcam analysis which requires the corresponding eye to have been run beforehand).
Alternatively, batch analysis can be done with the script `dlc_intake.ipynb` with a terminal interface. **Note: This must be run using the `fmephys` conda environment.** The `fmephys` environment can be installed from this repository with the terminal line `conda env create -f <path_to_yaml_file>`. An example terminal use would look like `python3 dlc_intake.py /data/path/ /save/path/` where all default parameters are used. To learn about arguments that can be passed in, run `python3 dlc_intake.py -h` for updated descriptions of each function and to get their defaults.
