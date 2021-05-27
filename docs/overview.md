# FreelyMovingEphys -- overview

## Typical Sequence (leading up to specific project analysis)
1. Preprocess ephys data and merge recordings
2. Kilosort
3. Phy
4. Split ephys recordings
5. Write yaml config or batch csv
6. Run either session or batch analysis
7. Run additional proejct-specific analysis

## Required Data, Naming, File Structures
### Required Data and Formatting
For a given recording, the following inputs will be expected:
* worldcam .avi video, interlaced at 30fps
* worldcam .csv timestamps, saved out of Bonsai as the time of day to 0.1 microsecond percision (should run with greater percision than this, but that would be mostly meaningless)
* topdown .avi video, 60fps
* eyecam .avi, interlaced at 30fps (ephys analysis will only make use of REYE, but preprocesing is tested with LEYE included)
* IMU binary .bin file (only for freely moving recordings)
* ephys .bin file
* ephys .csv timestamp file
* running ball otical mouse .csv file (only for headfixed recordings)

Expected timestamp .csv format from Bonsai has no header (i.e. index 0 is the first timestamp):

| 15:00:19.0987520 |
|---|
| 15:00:19.1308160 |
| 15:00:19.1619712 |
| 15:00:19.1937664 |
| 15:00:19.2337280 |
| 15:00:19.2656896 |

Expected running ball optical mouse .csv file format should include headers before data:
| Timestamp.TimeOfDay | Value.X | Value.Y |
| --- | --- | --- |
| 14:22:28.7856768 | 960 | 540 |
| 14:23:13.8701184 | 960 | 540 |
| 14:23:13.8780672 | 957 | 543 |
| 14:23:14.2781696 | 960 | 540 |

### File Naming
All files of the same session should have the same base name, with an added recording type for the setup (i.e. freely moving or headfixed) and a data type (e.g. REYE, World, etc.):
```
date_subject_manupulation_rig_recType_dataType.ext
```

## Config Files
A .yaml file like [this one](/example_configs/config.yaml) can be used to run all of preprocessing and ephys analysis. The default .yaml file is commented to describe what each value in the file changes.

## Batch Files
Write a csv file with each index representing a single day+mouse session with the following required columns shown in the table below:

| experiment_date | animal_name | experiment_outcome | run_preproccessing | run_ephys_analysis | load_for_data_pool | best_fm_rec | unit2highlight | current_status | animal_dirpath | computer | drive | probe_name |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
|date|name|good|TRUE|TRUE|TRUE|fm1|0|ephys analysis complete|/path/to/animal/directory/|computer_name|drive_name|probe model name|