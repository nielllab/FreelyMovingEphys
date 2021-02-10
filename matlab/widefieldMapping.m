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

%% analyzed 020921
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

%% analyzed 020921
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


%%

%%% call batchDfofMovie
batchDfofMovie