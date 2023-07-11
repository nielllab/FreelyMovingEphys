# Aligning data in Matlab

There are two functions: one for head-fixed recordings, `alignHeadFixed`, and another for freely moving recordings, `alignFreelyMoving`. The functions that actually read in the .NC files are in the "utils" subdirectory.
To run these, you just need the path to a recording (it has to end in the directory for a specific stimulus/condition, *not* the overall animal directory).
```
alignFreelyMoving("D:/freely_moving_ephys/010101/AnimalName/fm1")
```
The functions do not return anything. They will write a .mat file containing one struct for each data input (for head-fixed: treadmill, ephys, and eyecam; for freely moving: IMU, ephys, eyecam, and topcam). This file will be written into the recording directory (the path given as the argument to that function).

The timing drift of Ephys and the IMU are corrected before everything is aligned.

All inputs will be aligned relative to a shared T0 (which is the start of ephys data). So, if the camera started slightly before ephys started, it will have a negative timestamp for any of the frames acquired before ephys started. That is to say, this does not drop any data -- it only shifts it to line it up w/ the start of Ephys.

In the `ephysData` struct, spike times will be in in the group `spikeT`. Unit numbers will not increase monotonicly, because I only corrected drift and offset for the units that were labeled as 'good' in Phy2. Any units included in `spikeT` are good. The spike times (w/out timing correction) and for all units (including noise or MUA) are still saved in `spikeT_raw`. Spike times are in units of seconds. The shank and channel numbers are correct in this, but don't rely on the values for depth. The column `t0` will have an identical value for every unit. This is the start time of ephys data in absolute time. For absolute time, this is the time of day that the data were recorded, specificlly, the number of seconds since midnight.

When you load that .mat and look at the groups for each input, there'll be the timestamps `eyeT`, the video `video`, (as a uint8 array, and downsampled from the original resolution), and the behavioral parameters. For example, the eyecam will have an array of doubles for theta (horizontal pupil orientation in radians), phi (vertical pupil orientation, also radians), longaxis (pupil radius in pxls), etc. The original timestamps (no timing alignment) are in here also and have "_raw" at the end of the name.


DMM, July 11 2023