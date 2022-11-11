# FreelyMovingEphys
Preprocessing and analysis for electrophysiology experiments in freely moving mice.

## Setup
Create the "ephys1" conda environment with `conda env create -f environment.yml`. Activate the environment with `conda activate ephys1`, navigate to the path of this respoitory, and run `pip install -e .` to install the code base in your "ephys1" environment as a package named "fmEphys".

## Use


To analyze a recording, fill out a yaml config file (following the template in "/config/ephys_cfg.yaml") and run

`python -m fmEphys.pipeline -c /path/to/your/cfg.yaml`

to analyze the session, replacing the argument for `-c` with the file path of the config file for your experiment. This command does not need to be run from the path of this repository, as long as the "ephys1" environment is active. The "fmEphys" package will update for any changes to the local or pulled changes to the repository.


## Running individual function
in the conda environment where fmEphys is installed
run the function named `time()` in the python file '/FreelyMovingEphys/fmEphys/utils/time.py'
You can pass arguments this way, but they all have to be named with `--cfg /path/to/config/
```
python -m fmEphys.utils.time now
```


## Possible labeles for recording directories
fm1
fm_light
fm_dark
fm1_dark
IRspot
hf1_wn
hf2_sparsenoiseflash
hf3_gratings
hf4_revchecker

hf1_revchecker500ms
hf2_sparsenoiseflash500msISI
hf3_staticgratings500ms
hf4_worldcam
hf5_sparsenoiseflash500ms
hf6_staticgratings500msISI
hf7_worldcam10s

hf1_revchecker500ms
hf5_sparsenoiseflash500ms
hf4_worldcam
fm1