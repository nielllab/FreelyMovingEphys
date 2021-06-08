# FreelyMovingEphys
Preprocessing and analysis for ephys experiements in freely moving mice.

## Installation
Full instructions can be found in this repository [here](/docs/installation.md).

## Documentation
See the `/docs/` directory of this repository for usage and function details information about the repository.

## Usage
### Data
For information about required files for each session and the expected data formats, see [this markdown overview](/docs/overview.md).

### The .yaml config
Preprocessing and analysis pipelines are run through the .yaml config. An example of this yaml filled in with default values and commented with the meaning of each field can be found [here](/example_configs/config.yaml) in this repository.

### Ephys preprocessing
The initial preprocessing of ephys data is done in Matlab using [this](/matlab/preprocessEphysData.m) script to select the .bin files of each ephys recording within a session and merge the data into one combined .bin.

Next, Kilosort and Phy are run on the data.

Finally, the python module to split recordings back apart is run with `python -m split_ephys_recordings`. This module will open a dialog box in which the user should select the .mat file specifying the indices at which the .bin need to be split. This is saved out from the initial matlab merge script.

### Quick receptive field mapping
For a quick mapping of receptive fields, run `python -m project_analysis.map_receptive_fields`. You'll be prompted to enter a directory in a dialog box. Choose the recording directory for white noise stimulus. This module needs to be run after spike sorting and the splitting of individual recordings, but it includes preprocessing for all necessary data inputs. See the [map_receptive_fields documentation](/docs/project_analysis/map_receptive_fields.md) for more information.

### Preprocessing and ephys analysis
Preprocessing for eye, world, and topdown cameras, as well as IMU and optical mouse data can be run on an all of the recordings of a session at the same time. It can also be run on multiple recordings in batch analysis.

In this first case where one session (containing any number of recordings) is being preprocessed, run `python session_analysis.py --config /path/to/config.yaml`. The .yaml file should follow [this](/example_configs/config.yaml) template. Optionally, ephys analysis can be run on each recording of a session at the same time as preprocesing, if this is indicated in the .yaml config. The flag `--clear_dlc True` can be used to delete DeepLabCut .h5 files before running DeepLabCut pose estimation, in the even that DLC is being rerun for a second time.

For batch analysis of multiple sessions, complete a .csv file following the batch file format described in [this markdown overview](/docs/overview.md).
To start batch analysis, run `python batch_analysis.py --csv_filepath /path/to/completed_experiment_pool.csv --config /path/to/config.yaml --log_dir /save/path/`. Like individual session analysis, you can delete existing DLC .h5 files by adding the `--clear_dlc True` flag.