# FreelyMovingEphys
Preprocessing and analysis for electrophysiology experiments in freely moving mice.

## Setup
Create the "ephys1" conda environment with `conda env create -f environment.yml`. Activate the environment with `conda activate ephys1` and run `pip install -e .` to install the code base in your "ephys1" environment as a package named "fmEphys".

## Use


To analyze a recording, fill out a yaml config file (following the template in "/config/ephys_cfg.yaml") and run

`python -m fmEphys.pipeline -c /path/to/your/cfg.yaml`

to analyze the session, replacing the argument for `-c` with the file path of the config file for your experiment. This command does not need to be run from the path of this repository, as long as the "ephys1" environment is active. The "fmEphys" package will update for any changes to the local or pulled changes to the repository.
