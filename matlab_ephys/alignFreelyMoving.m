function alignFreelyMoving(rpath)
% alignFreelyMoving
% Align data for a head-fixed recording
%
% rpath is the path for a particular recording (meaning the directory
% specific to that stimulus, e.g., "010101/AnimalName/hf1_wn". Run the
% preprocessing pipeline through the "parameters" step so that the Ephys
% .JSON already exists and the .NC files for the other inputs are also
% already written in that recoridng direcory.
%
% Didn't write the function to read & align worldcam data yet, so this
% ignores that input.
%
% This returns nothing, but writes a single .mat file, named
% "aligned_data.mat" that will contain the data for all of the aligned
% data. When reading this back in, make sure to use the videos as the type
% uint8, and to not use the "raw" timestamps (e.g., "eyeT_raw" which are
% not aligned to ephys timestamps (and in the case of the IMU and Ephys
% files, not corrected for the offset and drift).
%
% Written by DMM, July 2023
%


%%% Gather files.

% ephys
ephys_path = getPath(rpath, "*ephys_merge.json");

% pupil camera
eye_path = getPath(rpath, "*REYE.nc");

% topdown (for freely-moving only)
top_path = getPath(rpath, "*TOP1.nc");

% imu
imu_path = getPath(rpath, "*imu.nc");

%%% Load structs and correct timing drifts

display(sprintf("Reading ephys."));
ephysData = readEphys(ephys_path);

display(sprintf("Reading eyecam."));
eyeData = readEyecam(eye_path);

display(sprintf("Reading IMU."));
imuData = readIMU(imu_path);

display(sprintf("Reading topcam."));
topData = readTopcam(top_path);


%%% Align timing

% Start time for ephys

display(sprintf("Aligning to ephys T0."));
T0 = ephysData.t0.x1;

topData.topT = topData.topT_raw - T0;
if (topData.topT(1) < -600)
    topData.topT = topData.topT + 8*60*60;
end

eyeData.eyeT = eyeData.eyeT_raw - T0;
if (eyeData.eyeT(1) < -600)
    % 8 hr offset in old datasets
    eyeData.eyeT = eyeData.eyeT + 8*60*60;
end

% IMU
imuData.imuT = imuData.imuT_raw - T0;


%%% Save as a .mat

savepath = rpath + "/aligned_data.mat";

display(sprintf("Writing data to %s", savepath));

save(savepath, "eyeData", "topData", "ephysData", "imuData");

