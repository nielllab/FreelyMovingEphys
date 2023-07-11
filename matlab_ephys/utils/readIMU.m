function imuData = readIMU(NC_path)
% readIMU
% Read IMU .NC file and correct timing.
%
% Written by DMM, July 2023
%

drift_rate = -0.000114; % sec/sample
offset_val = 0.1; % sec

imuData = struct;

% 
imu_channels = ncread(NC_path,'__xarray_dataarray_variable__');
imu_ch_names = string(ncread(NC_path,'channel'));

for i = 1:14
    n = imu_ch_names(i, 1);
    imuData.(n) = imu_channels(i,:)';
end

imuData.imuT = ncread(imu_path,'sample');

imuData.imuT_raw = imuData.imuT;

% apply timing correction
imuData.imuT = imuData.imuT - (offset_val + imuData.imuT * drift_rate);


end