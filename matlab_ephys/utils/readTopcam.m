function topData = readTopcam(NC_path)
% readTopcam
% Read topdown camera .NC file.
%
% Written by DMM, July 2023
%


topData = struct;

topData.topT_raw = ncread(NC_path, "timestamps");

top_pts = ncread(NC_path, "TOP1_pts");
top_pt_names = string(ncread(NC_path, "pts"));
for i = 1:size(top_pt_names, 1)
    param_name = top_pt_names(i);
    topData.(param_name) = top_pts(i,:);
end

top_props = ncread(NC_path, "TOP1_props");
top_prop_names = string(ncread(NC_path, "prop"));
for i = 1:size(top_prop_names, 1)
    param_name = top_prop_names(i);
    topData.(param_name) = top_props(i,:);
end

topData.video = typecastMulti(ncread(NC_path, "TOP1_video"), "uint8");
topData.video = permute(topData.video, [2, 1, 3]);

end