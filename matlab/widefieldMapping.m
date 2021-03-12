close all
clear all
dbstop if error

%% folder where the data live:
foldername = 'T:\freely_moving_ephys\widefield\';
pathname = foldername;
datapathname = foldername;  
outpathname = foldername;

%start the counter
n = 0;

%% analyzed 020921 PRLP
% n=n+1;
% files(n).subj = 'EE11P11LT';
% files(n).expt = '020821';
% files(n).topox =  '020821_EE11P11LT_RIG2_MAP\020821_EE11P11LT_RIG2_MAP_TOPOX\020821_EE11P11LT_RIG2_MAP_TOPOXmaps.mat';
% files(n).topoxdata = '020821_EE11P11LT_RIG2_MAP\020821_EE11P11LT_RIG2_MAP_TOPOX\020821_EE11P11LT_RIG2_MAP_TOPOX';
% files(n).topoy =  '020821_EE11P11LT_RIG2_MAP\020821_EE11P11LT_RIG2_MAP_TOPOY\020821_EE11P11LT_RIG2_MAP_TOPOYmaps.mat';
% files(n).topoydata = '020821_EE11P11LT_RIG2_MAP\020821_EE11P11LT_RIG2_MAP_TOPOY\020821_EE11P11LT_RIG2_MAP_TOPOY';
% files(n).rignum = 'rig2'; %%% or 'rig1'
% files(n).monitor = 'land'; %%% for topox and y
% files(n).label = 'camk2 gc6';
% files(n).notes = 'good imaging session';

%% analyzed 020921 PRLP
% n=n+1;
% files(n).subj = 'EE11P11RT';
% files(n).expt = '020821';
% files(n).topox =  '020821_EE11P11RT_RIG2_MAP\020821_EE11P11RT_RIG2_MAP_TOPOX\020821_EE11P11RT_RIG2_MAP_TOPOXmaps.mat';
% files(n).topoxdata = '020821_EE11P11RT_RIG2_MAP\020821_EE11P11RT_RIG2_MAP_TOPOX\020821_EE11P11RT_RIG2_MAP_TOPOX';
% files(n).topoy =  '020821_EE11P11RT_RIG2_MAP\020821_EE11P11RT_RIG2_MAP_TOPOY\020821_EE11P11RT_RIG2_MAP_TOPOYmaps.mat';
% files(n).topoydata = '020821_EE11P11RT_RIG2_MAP\020821_EE11P11RT_RIG2_MAP_TOPOY\020821_EE11P11RT_RIG2_MAP_TOPOY';
% files(n).rignum = 'rig2'; %%% or 'rig1'
% files(n).monitor = 'land'; %%% for topox and y
% files(n).label = 'camk2 gc6';
% files(n).notes = 'good imaging session';


%% analyzed 021421 PRLP
% n=n+1;
% files(n).subj = 'EE11P11TT';
% files(n).expt = '020821';
% files(n).topox =  '020821_EE11P11TT_RIG2_MAP\020821_EE11P11TT_RIG2_MAP_TOPOX\020821_EE11P11TT_RIG2_MAP_TOPOXmaps.mat';
% files(n).topoxdata = '020821_EE11P11TT_RIG2_MAP\020821_EE11P11TT_RIG2_MAP_TOPOX\020821_EE11P11TT_RIG2_MAP_TOPOX';
% files(n).topoy =  '020821_EE11P11TT_RIG2_MAP\020821_EE11P11TT_RIG2_MAP_TOPOY\020821_EE11P11TT_RIG2_MAP_TOPOYmaps.mat';
% files(n).topoydata = '020821_EE11P11TT_RIG2_MAP\020821_EE11P11TT_RIG2_MAP_TOPOY\020821_EE11P11TT_RIG2_MAP_TOPOY';
% files(n).rignum = 'rig2'; %%% or 'rig1'
% files(n).monitor = 'land'; %%% for topox and y
% files(n).label = 'camk2 gc6';
% files(n).notes = 'good imaging session';


%% analyzed 021521 NC NAME INCORRECT IN FOLDER
% n=n+1;
% files(n).subj = 'EE12P1LN';
% files(n).expt = '021521';
% files(n).topox =  '021521_EE13P2LN_RIG2_MAP\021521_EE13P2LN_RIG2_MAP_TOPOX\021521_EE13P2LN_RIG2_MAP_TOPOXmaps.mat';
% files(n).topoxdata = '021521_EE13P2LN_RIG2_MAP\021521_EE13P2LN_RIG2_MAP_TOPOX\021521_EE13P2LN_RIG2_MAP_TOPOX';
% files(n).topoy =  '021521_EE13P2LN_RIG2_MAP\021521_EE13P2LN_RIG2_MAP_TOPOY\021521_EE13P2LN_RIG2_MAP_TOPOYmaps.mat';
% files(n).topoydata = '021521_EE13P2LN_RIG2_MAP\021521_EE13P2LN_RIG2_MAP_TOPOY\021521_EE13P2LN_RIG2_MAP_TOPOY';
% files(n).rignum = 'rig2'; %%% or 'rig1'
% files(n).monitor = 'land'; %%% for topox and y
% files(n).label = 'camk2 gc6';
% files(n).notes = 'good imaging session';


%% analyzed 021521 NC
% n=n+1;
% files(n).subj = 'EE13P2LT';
% files(n).expt = '021521';
% files(n).topox =  '021521_EE13P2LT_RIG2_MAP\021521_EE13P2LT_RIG2_MAP_TOPOX\021521_EE13P2LT_RIG2_MAP_TOPOXmaps.mat';
% files(n).topoxdata = '021521_EE13P2LT_RIG2_MAP\021521_EE13P2LT_RIG2_MAP_TOPOX\021521_EE13P2LT_RIG2_MAP_TOPOX';
% files(n).topoy =  '021521_EE13P2LT_RIG2_MAP\021521_EE13P2LT_RIG2_MAP_TOPOY\021521_EE13P2LT_RIG2_MAP_TOPOYmaps.mat';
% files(n).topoydata = '021521_EE13P2LT_RIG2_MAP\021521_EE13P2LT_RIG2_MAP_TOPOY\021521_EE13P2LT_RIG2_MAP_TOPOY';
% files(n).rignum = 'rig2'; %%% or 'rig1'
% files(n).monitor = 'land'; %%% for topox and y
% files(n).label = 'camk2 gc6';
% files(n).notes = 'good imaging session';

%% analyzed 022221 NC
% n=n+1;
% files(n).subj = 'EE11P12LT';
% files(n).expt = '022221';
% files(n).topox =  '022221_EE11P12LT_RIG2_MAP\022221_EE11P12LT_RIG2_MAP_TOPOX\022221_EE11P12LT_RIG2_MAP_TOPOXmaps.mat';
% files(n).topoxdata = '022221_EE11P12LT_RIG2_MAP\022221_EE11P12LT_RIG2_MAP_TOPOX\022221_EE11P12LT_RIG2_MAP_TOPOX';
% files(n).topoy =  '022221_EE11P12LT_RIG2_MAP\022221_EE11P12LT_RIG2_MAP_TOPOY\022221_EE11P12LT_RIG2_MAP_TOPOYmaps.mat';
% files(n).topoydata = '022221_EE11P12LT_RIG2_MAP\022221_EE11P12LT_RIG2_MAP_TOPOY\022221_EE11P12LT_RIG2_MAP_TOPOY';
% files(n).rignum = 'rig2'; %%% or 'rig1'
% files(n).monitor = 'land'; %%% for topox and y
% files(n).label = 'camk2 gc6';
% files(n).notes = 'good imaging session';

%% analyzed 022221 NC
% n=n+1;
% files(n).subj = 'EE12P3LT';
% files(n).expt = '022221';
% files(n).topox =  '022221_EE12P3LT_RIG2_MAP\022221_EE12P3LT_RIG2_MAP_TOPOX\022221_EE12P3LT_RIG2_MAP_TOPOXmaps.mat';
% files(n).topoxdata = '022221_EE12P3LT_RIG2_MAP\022221_EE12P3LT_RIG2_MAP_TOPOX\022221_EE12P3LT_RIG2_MAP_TOPOX';
% files(n).topoy =  '022221_EE12P3LT_RIG2_MAP\022221_EE12P3LT_RIG2_MAP_TOPOY\022221_EE12P3LT_RIG2_MAP_TOPOYmaps.mat';
% files(n).topoydata = '022221_EE12P3LT_RIG2_MAP\022221_EE12P3LT_RIG2_MAP_TOPOY\022221_EE12P3LT_RIG2_MAP_TOPOY';
% files(n).rignum = 'rig2'; %%% or 'rig1'
% files(n).monitor = 'land'; %%% for topox and y
% files(n).label = 'camk2 gc6';
% files(n).notes = 'good imaging session';


%% analyzed 030121 NC
% n=n+1;
% files(n).subj = 'EE8P7LT';
% files(n).expt = '030121';
% files(n).topox =  '030121_EE8P7LT_RIG2_MAP\030121_EE8P7LT_RIG2_MAP_TOPOX\030121_EE8P7LT_RIG2_MAP_TOPOXmaps.mat';
% files(n).topoxdata = '030121_EE8P7LT_RIG2_MAP\030121_EE8P7LT_RIG2_MAP_TOPOX\030121_EE8P7LT_RIG2_MAP_TOPOX';
% files(n).topoy =  '030121_EE8P7LT_RIG2_MAP\030121_EE8P7LT_RIG2_MAP_TOPOY\030121_EE8P7LT_RIG2_MAP_TOPOYmaps.mat';
% files(n).topoydata = '030121_EE8P7LT_RIG2_MAP\030121_EE8P7LT_RIG2_MAP_TOPOY\030121_EE8P7LT_RIG2_MAP_TOPOY';
% files(n).rignum = 'rig2'; %%% or 'rig1'
% files(n).monitor = 'land'; %%% for topox and y
% files(n).label = 'camk2 gc6';
% files(n).notes = 'good imaging session';



%% analyzed 030121 NC
% n=n+1;
% files(n).subj = 'EE8P7RT';
% files(n).expt = '030121';
% files(n).topox =  '030121_EE8P7RT_RIG2_MAP\030121_EE8P7RT_RIG2_MAP_TOPOX\030121_EE8P7RT_RIG2_MAP_TOPOXmaps.mat';
% files(n).topoxdata = '030121_EE8P7RT_RIG2_MAP\030121_EE8P7RT_RIG2_MAP_TOPOX\030121_EE8P7RT_RIG2_MAP_TOPOX';
% files(n).topoy =  '030121_EE8P7RT_RIG2_MAP\030121_EE8P7RT_RIG2_MAP_TOPOY\030121_EE8P7RT_RIG2_MAP_TOPOYmaps.mat';
% files(n).topoydata = '030121_EE8P7RT_RIG2_MAP\030121_EE8P7RT_RIG2_MAP_TOPOY\030121_EE8P7RT_RIG2_MAP_TOPOY';
% files(n).rignum = 'rig2'; %%% or 'rig1'
% files(n).monitor = 'land'; %%% for topox and y
% files(n).label = 'camk2 gc6';
% files(n).notes = 'good imaging session';


%% analyzed 030821 NC 
n=n+1;
files(n).subj = 'EE8P8LT';
files(n).expt = '030821';
files(n).topox =  '030821_EE8P8LT_RIG2_MAP\030821_EE8P8LT_RIG2_MAP_TOPOX\030821_EE8P8LT_RIG2_MAP_TOPOXmaps.mat';
files(n).topoxdata = '030821_EE8P8LT_RIG2_MAP\030821_EE8P8LT_RIG2_MAP_TOPOX\030821_EE8P8LT_RIG2_MAP_TOPOX';
files(n).topoy =  '030821_EE8P8LT_RIG2_MAP\030821_EE8P8LT_RIG2_MAP_TOPOY\030821_EE8P8LT_RIG2_MAP_TOPOYmaps.mat';
files(n).topoydata = '030821_EE8P8LT_RIG2_MAP\030821_EE8P8LT_RIG2_MAP_TOPOY\030821_EE8P8LT_RIG2_MAP_TOPOY';
files(n).rignum = 'rig2'; %%% or 'rig1'
files(n).monitor = 'land'; %%% for topox and y
files(n).label = 'camk2 gc6';
files(n).notes = 'good imaging session';


%% analyzed 030821 NC 
n=n+1;
files(n).subj = 'EE8P8RT';
files(n).expt = '030821';
files(n).topox =  '030821_EE8P8RT_RIG2_MAP\030821_EE8P8RT_RIG2_MAP_TOPOX\030821_EE8P8RT_RIG2_MAP_TOPOXmaps.mat';
files(n).topoxdata = '030821_EE8P8RT_RIG2_MAP\030821_EE8P8RT_RIG2_MAP_TOPOX\030821_EE8P8RT_RIG2_MAP_TOPOX';
files(n).topoy =  '030821_EE8P8RT_RIG2_MAP\030821_EE8P8RT_RIG2_MAP_TOPOY\030821_EE8P8RT_RIG2_MAP_TOPOYmaps.mat';
files(n).topoydata = '030821_EE8P8RT_RIG2_MAP\030821_EE8P8RT_RIG2_MAP_TOPOY\030821_EE8P8RT_RIG2_MAP_TOPOY';
files(n).rignum = 'rig2'; %%% or 'rig1'
files(n).monitor = 'land'; %%% for topox and y
files(n).label = 'camk2 gc6';
files(n).notes = 'good imaging session';







%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% run the batch analysis (keep this at the bottom of the script
batchDfofMovie


