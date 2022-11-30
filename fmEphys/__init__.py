"""


"""
import warnings
warnings.filterwarnings('ignore')

# Main scripts
from .calcWarp import calcWarp
from .splitRec import splitRec
# from .pipeline import pipeline
# from quickRF import quickRF
# from quickPSTH import quickPSTH

# Basic helper functions
from .utils.helper import (
    Log,
    calc_list_index,
    stderr,
    z_score,
    modind,
    nearest_ind,
    str_to_bool,
    probe_to_ch,
    get_all_cap_combs,
    drop_NaNs,
    add_jitter,
    series_to_arr,
    series_to_list,
    merge_uneven_xr,
    split_xyl,
    flatten_dict,
    nest_dict,
    empty_obj_col
)
from .utils.file import (
    write_h5,
    read_h5,
    hdf_to_mat,
    mat_to_hdf,
    read_DLC_data,
    read_yaml,
    write_yaml,
    read_json
)
from .utils.path import (
    choose_most_recent,
    find,
    list_subdirs
)
from .utils.internals import (
    get_cfg,
    get_probe_sites,
    get_rec_name,
    set_cfg_paths,
    fill_rec_details,
    assign_stim_name
)

# Filtering
from .utils.filter import (
    convfilt,
    nanmedfilt,
    butterfilt
)

# Timestamps
from .utils.time import (
    fmt_time,
    interp_time,
    read_time,
    fmt_now
)

# Video preprocessing
from .utils.video import (
    deinterlace,
    rotate_video,
    calc_distortion,
    fix_distortion,
    fix_contrast,
    avi_to_arr
)

# IMU
from .utils.imu import (
    read_IMU_binary,
    preprocess_IMU,
    preprocess_TTL
)
from .utils.head_orientation import (
    Kalman,
    ImuOrientation
)

# Treadmill
from .utils.treadmill import preprocess_treadmill

# Top camera
from .utils.topcam import preprocess_topcam
from .utils.body import (
    track_body
)

# Eye camera
from .utils.eyecam import preprocess_eyecam
from .utils.pupil import (
    fit_ellipse,
    track_pupil
)

# World camera
from .utils.worldcam import preprocess_worldcam

# Full recording analysis
from .utils.freely_moving import freely_moving
from .utils.head_fixed import head_fixed
from .utils.response_props import (
    calc_kde_PSTH,
    calc_all_PSTH,
    calc_all_tuning,
    calc_all_STA,
    calc_all_STV,
    drop_nearby_events,
    drop_repeat_events,
    calc_saccade_times
)

# Plotting functions
from .utils.plot import (
    jitter,
    plot_all_tuning,
    plot_all_STA,
    plot_all_STV
)

from .utils.ephys import (
    do_data_split
)