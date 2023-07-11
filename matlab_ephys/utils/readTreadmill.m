function ballData = readTreadmill(NC_path)
% readTreadmill
% Read treadmill .NC file.
%
% Written by DMM, July 2023
%


ballData = struct;

ball_props = ncread(NC_path, "__xarray_dataarray_variable__");
ball_prop_names = string(ncread(NC_path, "move_params"));

for i = 1:size(ball_prop_names, 1)

    param_name = ball_prop_names(i);

    if (param_name == "timestamps")
        param_name = "ballT_raw";
    end

    ballData.(param_name) = ball_props(i,:);

end

end