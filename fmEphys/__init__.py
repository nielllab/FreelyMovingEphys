""" Freely moving electrophysiology.

Package for analyzing electrophysiology data recorded
in freely moving animals.


Written by DMM, 2020-2023
"""


#---------------------------------
### MODULES

# Main pipeline
from .pipeline import pipeline

# Batch pipeline analysis
from .pipelineBatch import (
    pipelineBatch,
    batch_cycle
)

# Split (previously merged) ephys binary files
from .splitRec import splitRec

# Preliminary receptive field mapping
from .quickRF import quickRF

# Write default config file
from .makeCfg import makeCfg

# Calculate image distortion from video
# of moving checkerboard
from .calcWarp import calcWarp


#---------------------------------
### HELPER FUNCTIONS AND CLASSES

# Error-logging class
from .utils.log import Logger

# Timestamps
from .utils.time import (
    fmt_time,
    interp_time,
    read_time,
    fmt_now
)

# Misc helper functions
from .utils.auxiliary import (
    write_dummy_cfg,
    str_to_bool,
    find_index_in_list,
    flatten_series,
    show_xr_objs,
    replace_xr_obj,
    fill_NaNs,
    drop_nan_along,
    z_score,
    stderr,
    blank_col
)

# File paths
from .utils.path import (
    choose_most_recent,
    up_dir,
    find,
    filter_file_search,
    check_subdir,
    list_subdirs,
    auto_recording_name
)

# Filters
from .utils.filter import (
    convfilt,
    sub2ind,
    nanmedfilt,
    butterfilt
)

# Correlation
from .utils.correlation import (
    nanxcorr
)

# Read and write files
from .utils.file import (
    read_DLC_data,
    write_h5,
    read_h5,
    write_group_h5,
    get_group_h5_keys,
    read_group_h5
)

# Videos
from .utils.video import (
    deinterlace,
    rotate_video,
    calc_distortion,
    fix_distortion,
    fix_contrast,
    avi_to_arr
)

# PSTH calculations (as seperate funcs)
from .utils.psth import (
    drop_nearby_events,
    drop_repeat_events,
    calc_PSTH,
    calc_hist_PSTH
)


#---------------------------------
### BASE CLASSES

# Base input (mostly for timestamps)
from .utils.base import BaseInput

# Cameras
from .utils.camera import Camera

# Ephys and stimulus analysis
from .utils.ephys import Ephys

# Module to analyze a full session
from .utils.run import Session

# Raw ephys data
from .utils.prelim import RawEphys

# Head-fixed treadmill
from .utils.treadmill import RunningBall

# Intertial measurement unit
from .utils.imu import Imu

# Head-mounted eye camera
from .utils.eyecam import Eyecam

# Top-down arena camera
from .utils.topcam import Topcam

# Head-mounted world camera
from .utils.worldcam import Worldcam

# Arena side camera
from .utils.sidecam import Sidecam

# TTL timing signals from stimulus computer
from .utils.ttl import TTL


#---------------------------------
### STIMULUS-SPECIFIC CLASSES

from .utils.freely_moving import FreelyMovingLight

from .utils.freely_moving_dark import FreelyMovingDark

from .utils.gratings import HeadFixedGratings

from .utils.white_noise import HeadFixedWhiteNoise

from .utils.rev_checker import HeadFixedReversingCheckboard

from .utils.sparse_noise import HeadFixedSparseNoise


#---------------------------------
### STIM-SPECIFIC PRELIMINARY ANALYSIS

# RF mapping with spike-sorted data
from .utils.prelimRF_sort import prelimRF_sort

# RF mapping using raw ephys binary files
from .utils.prelimRF_raw import prelimRF_raw

