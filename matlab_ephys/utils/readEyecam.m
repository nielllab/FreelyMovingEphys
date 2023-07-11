function eyeData = readEyecam(NC_path)
% readEyecam
% Read eye camera .NC file.
%
% Written by DMM, July 2023
%

% create eye data struct
eyeData = struct;

% timestamps
eyeData.eyeT_raw = ncread(NC_path, "timestamps");

% add ellipse params
ellipse_fit_params = ncread(NC_path, "REYE_ellipse_params");
ellipse_param_names = string(ncread(NC_path, "ellipse_params"));

for i = 1:size(ellipse_param_names, 1)
    param_name = ellipse_params_names(i);
    eyeData.(param_name) = ellipse_fit_params(i);
end

% pts tracked in DLC
eye_pts = nc_read(NC_path, "REYE_pts");
eye_pts_names = string(ncread(NC_path, "point_loc"));

for i = 1:size(eye_pts_names, 1)
    param_name = eye_pts_names(i);
    eyeData.(param_name) = eye_pts(i);
end

% load eyecam video
eyeData.video = typecastarray(ncread(NC_path, "REYE_video"), "uint8");
eyeData.video = permute(eyeData.video, [2, 1, 3]);

end