# FreelyMovingEphys
Preprocessing and analysis for electrophysiology experiments in freely moving mice.
## Installation
Full instructions can be found in this repository [here](/docs/installation.md).
## Documentation
See the `/docs/` directory of this repository for usage and function details information about the repository.
## Usage
### Data
For information about required files for each session and the expected data formats, see [this markdown overview](/docs/overview.md).
### The .yaml config
Preprocessing and analysis pipelines are run through the .yaml config. An example of this yaml filled in with default values and commented with the meaning of each field can be found [here](/config/config.yaml) in this repository.
### Preliminary laminar depth estimation
For a quick estimation of how deep the ephys probe is, run `python -m prelim_depth`. A dialog box will launch, in which the user will choose the binary ephys file for any recording. It is best to use a longer recording, like a freely moving recording or headfixed whitenoise. When a binary file is selected, a new window will open in which the user will choose which probe model the data was aquired with.

This `prelim_depth` module gets the approximate depth of layer 5 of primary visual cortex by finding the channel with the highest MUA power profie. A figure will be saved out from this module showing the depth of each channel relative to layer 5.

This runs very quickly adn requires only an ephys binary file, so it is possible to run this module before all data is done being aquired.
### Preliminary raw whitenoise receptive field mapping
For a quick mapping of receptive fields, where spikes are found by filtering on each probe channel, run `python -m prelim_raw_whitenoise`. *No spike sorting or preprocessing is needed for this module to run.* This can be run on any headfixed stimulus and runs in 5-10 minutes using only an ephys binary file and the name of the probe used. When run, the module opens a window in which the user chooses a ephys binary and the probe from a drop down menu. It is possible to run this with a freely moving binary, but depending on the length of the recording, but it may be too large a binary to hold in memory.
### Ephys preprocessing
The initial preprocessing of ephys data is done in Matlab using [this](/ephys_preprocessing/preprocessEphysData.m) script to select the .bin files of each ephys recording within a session and merge the data into one combined .bin.

Next, Kilosort and Phy2 are run on the data.

Finally, the python module to split recordings back apart is run with `python -m split_ephys_recordings`. This module will open a dialog box in which the user should select the .mat file specifying the indices at which the .bin need to be split. This is saved out from the initial matlab merge script.
### Preliminary receptive field mapping
For a quick mapping of receptive fields, run `python -m prelim_whitenoise`. This must be after spike sorting is done, and after the `split_ephys_recordings` module has been run. Like `prelim_depth`, a dialog box will open in which the user will choose the whitenoise stimulus directory. Once this directory is selected, a second dialog box will open in which the user will choose the model of probe used to aquire the data.

It is not necessary to run any preprocessing before running this module; the module will deinterlace and undistort the video in addition to the minimal ephys analysis necessary to get receptive fields and a few diagnostic figures.
### Session analysis
To fully analyze one session, run `python -m session_analysis`. This module will run preprocessing through ephys analysis. The user can either add the `--config` option in terminal followed by the path to the .yaml config file (i.e. `python -m session_analysis --config /path/to/config.yaml`), or this flag can be ommitted and a dialog box will be opened in which the user will select the animal directory containing the recording directories to be analyzed. The flag `--clear_dlc True` can be used to delete DeepLabCut .h5 files before running DeepLabCut pose estimation, in the even that DLC is being rerun for a second time. This is false by default, and DLC files will not be removed unless this argument is provided and set to true.
### Batch analysis
To fully analyze more than one session in sequence, run `python -m batch_analysis`. This requires several arguments: `--csv` should be followed by the path to a csv file with metadata about each session to analyze. (see [this documentation](/docs/overview.md) for the required format of this metadata), `--config` should be followed by the path to the .yaml config file, and `--log` should be followed by a directory in which the logger should save a .csv file of all errors that come up during the batch analysis. As with the `session_analysis` module, the flag `--clear_dlc` can be provided optionally.

An example command to run batch analysis might look like:

```
python -m batch_analysis --csv /path/to/completed_experiment_pool.csv --config /path/to/config.yaml --log /path/to/save/
```

This single provided config.yaml file will be used for all sessions, with the field for animal directory in the config file ignored and replaced with the animal directory for each session found in the metadata .csv file.

### Project analysis
Project-specific analysis can be found in the `/project_analysis/` directory.