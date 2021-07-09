%% generate kilosort maps

%% DB P128-6

%%% x coordinates for each site
[zeros(1,32) ones(1,32)*150 ones(1,32)*300 ones(1,32)*450];

%%% y coordinates for each site
[775:-25:0 775:-25:0 775:-25:0 775:-25:0];

%%% shank index
[ones(1,32) ones(1,32)*2 ones(1,32)*3 ones(1,32)*4];

%%% channel map
1:128;

%% DB P64-3

%%% x coordinates for each site
[zeros(1,32) ones(1,32)*250];

%%% y coordinates for each site
[775:-25:0 775:-25:0];

%%% shank index
[ones(1,32) ones(1,32)*2]

%%% channel map
1:64

%% DB P64-8

%%% x coordinates for each site
[repmat([21,0],1,16) repmat([271,250],1,16)]

%%% y coordinates for each site
[387.5:-12.5:0 387.5:-12.5:0]

%%% shank index
[ones(1,32) ones(1,32)*2]

%%% channel map
1:64


%% NN H64-LP

%%% x coordinates for each site
[zeros(1,32) ones(1,32)*250];

%%% y coordinates for each site
[775:-25:0 775:-25:0];

%%% shank index
[ones(1,32) ones(1,32)*2]

%%% channel map
1:64


%% NN 16ch

%%% x coordinates for each site
zeros(1,16)

%%% y coordinates for each site
375:-25:0

%%% shank index
ones(1,16)

%%% channel map
1:16


