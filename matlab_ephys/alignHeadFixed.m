function alignHeadFixed(rpath, skipImu)
% alignHeadFixed
% Align data for a head-fixed recording
%
% skipImu can be changed if you want to ignore the IMU data (some HF
% recordings use it for stimulus TTL signals, in which case skipImu
% should == 0). By default, skipImu == 1 which ignores that file.
%
% rpath is the path for a particular recording (meaning the directory
% specific to that stimulus, e.g., "010101/AnimalName/hf1_wn". Run the
% preprocessing pipeline through the "parameters" step so that the Ephys
% .JSON already exists and the .NC files for the other inputs are also
% already written in that recoridng direcory.
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


% Skip IMU by default 
if ~exist('skipImu', 'var') || isempty(skipImu)
    skipImu = 1;
end

% Skip worldcam by default
%if ~exist('skipWorld', 'var') || isempty(skipWorld)
%    skipWorld = 1;
%end


%%% Gather files.

% ephys
ephys_path = getPath(rpath, "*ephys_merge.json");

% pupil camera
eye_path = getPath(rpath, "*REYE.nc");

% running ball (for head-fixed only)
ball_path = getPath(rpath, "*speed.nc");

% imu
if (skipImu == 0)
    imu_path = getPath(rpath, "*imu.nc");
end

%%% Load as structs and correct timing drifts

sprintf("Reading ephys.")
ephysData = readEphys(ephys_path);

sprintf("Reading eyecam.")
eyeData = readEyecam(eye_path);

sprintf("Reading treadmill.")
ballData = readTreadmill(ball_path);

if (skipImu == 0)
    sprintf("Reading IMU.")
    imuData = readIMU(imu_path);
end

%%% Align timing

% Start time for ephys
T0 = ephysData.t0.x1;

eyeData.eyeT = eyeData.eyeT_raw - T0;
if (eyeData.eyeT(1) < -600)
    % 8 hr offset in old datasets
    eyeData.eyeT = eyeData.eyeT + 8*60*60;
end

% IMU
if (skipImu == 0)
    imuData.imuT = imuData.imuT_raw - T0;
end

% treadmill
ballData.ballT = ballData.ballT_raw - T0;


%%% Save as a .mat

savepath = rpath + "/aligned_data.mat";

sprintf("Writing data to %s", savepath)

if (skipImu == 0)
    save(savepath, 'eyeData', 'ballData', 'ephysData');

elseif (skipImu == 1)
    save(savepath, "eyeData", "ballData", "ephysData", "imuData");

end

