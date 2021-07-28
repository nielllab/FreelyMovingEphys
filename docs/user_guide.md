# FreelyMovingEphys User Guide

## Workflow


## Tutorial

## Immidiate diagnostic modules

### Preliminary laminar depth estimation
Estimating the laminar depth of the probe relative to Layer 5 of primary visual cortex takes less than ten minutes. A single-page pdf will be saved out when this is run with a panel for each shank of the probe. A red star will indicate the estimated position of Layer 5 of Cortex. This can often be more reliable than depth estimatino using the reversing checkerboard stimulus.
```
python -m prelim_depth
```
This module will open a window in which the user selects the probe used and the whitenoise ephys binary.

### Preliminary receptive fields
Mapping receptive fields from the raw worldcam video and ephys binary LFP takes less than ten minutes. No spike sorting or preprocessing is needed before this can be run.
```
python -m prelim_raw_whitenoise --wn_dir /path/to/animal/hf1_wn --probe DB-P128-6
```
This module can also be run without arguments (i.e. `python -m prelim_raw_whitenoise`) and a window will open in which the user can select a probe and the whitenoise directory to use.

## Preliminary analysis

## Ephys preprocessing

## Preprocessing

## Ephys


## 