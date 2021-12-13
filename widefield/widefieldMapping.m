%% DOCUMENT TITLE
% INTRODUCTORY TEXT
%%
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
% n=n+1;
% files(n).subj = 'EE8P8LT';
% files(n).expt = '030821';
% files(n).topox =  '030821_EE8P8LT_RIG2_MAP\030821_EE8P8LT_RIG2_MAP_TOPOX\030821_EE8P8LT_RIG2_MAP_TOPOXmaps.mat';
% files(n).topoxdata = '030821_EE8P8LT_RIG2_MAP\030821_EE8P8LT_RIG2_MAP_TOPOX\030821_EE8P8LT_RIG2_MAP_TOPOX';
% files(n).topoy =  '030821_EE8P8LT_RIG2_MAP\030821_EE8P8LT_RIG2_MAP_TOPOY\030821_EE8P8LT_RIG2_MAP_TOPOYmaps.mat';
% files(n).topoydata = '030821_EE8P8LT_RIG2_MAP\030821_EE8P8LT_RIG2_MAP_TOPOY\030821_EE8P8LT_RIG2_MAP_TOPOY';
% files(n).rignum = 'rig2'; %%% or 'rig1'
% files(n).monitor = 'land'; %%% for topox and y
% files(n).label = 'camk2 gc6';
% files(n).notes = 'good imaging session';


%% analyzed 030821 NC 
% % n=n+1;
% % files(n).subj = 'EE8P8RT';
% % files(n).expt = '030821';
% % files(n).topox =  '030821_EE8P8RT_RIG2_MAP\030821_EE8P8RT_RIG2_MAP_TOPOX\030821_EE8P8RT_RIG2_MAP_TOPOXmaps.mat';
% % files(n).topoxdata = '030821_EE8P8RT_RIG2_MAP\030821_EE8P8RT_RIG2_MAP_TOPOX\030821_EE8P8RT_RIG2_MAP_TOPOX';
% % files(n).topoy =  '030821_EE8P8RT_RIG2_MAP\030821_EE8P8RT_RIG2_MAP_TOPOY\030821_EE8P8RT_RIG2_MAP_TOPOYmaps.mat';
% % files(n).topoydata = '030821_EE8P8RT_RIG2_MAP\030821_EE8P8RT_RIG2_MAP_TOPOY\030821_EE8P8RT_RIG2_MAP_TOPOY';
% % files(n).rignum = 'rig2'; %%% or 'rig1'
% % files(n).monitor = 'land'; %%% for topox and y
% % files(n).label = 'camk2 gc6';
% % files(n).notes = 'good imaging session';
% 
% 
% % % analyzed 032221 NC 
% n=n+1;
% files(n).subj = 'EE14P1RT';
% files(n).expt = '032221';
% files(n).topox =  '032221_EE14P1RT_RIG2_MAP\032221_EE14P1RT_RIG2_MAP_TOPOX\032221_EE14P1RT_RIG2_MAP_TOPOXmaps.mat';
% files(n).topoxdata = '032221_EE14P1RT_RIG2_MAP\032221_EE14P1RT_RIG2_MAP_TOPOX\032221_EE14P1RT_RIG2_MAP_TOPOX';
% files(n).topoy =  '032221_EE14P1RT_RIG2_MAP\032221_EE14P1RT_RIG2_MAP_TOPOY\032221_EE14P1RT_RIG2_MAP_TOPOYmaps.mat';
% files(n).topoydata = '032221_EE14P1RT_RIG2_MAP\032221_EE14P1RT_RIG2_MAP_TOPOY\032221_EE14P1RT_RIG2_MAP_TOPOY';
% files(n).rignum = 'rig2'; %%% or 'rig1'
% files(n).monitor = 'land'; %%% for topox and y
% files(n).label = 'camk2 gc6';
% files(n).notes = 'good imaging session';
% 
% 
% % % analyzed 032221 NC 
% n=n+1;
% files(n).subj = 'EE14P1LN';
% files(n).expt = '032221';
% files(n).topox =  '032221_EE14P1LN_RIG2_MAP\032221_EE14P1LN_RIG2_MAP_TOPOX\032221_EE14P1LN_RIG2_MAP_TOPOXmaps.mat';
% files(n).topoxdata = '032221_EE14P1LN_RIG2_MAP\032221_EE14P1LN_RIG2_MAP_TOPOX\032221_EE14P1LN_RIG2_MAP_TOPOX';
% files(n).topoy =  '032221_EE14P1LN_RIG2_MAP\032221_EE14P1LN_RIG2_MAP_TOPOY\032221_EE14P1LN_RIG2_MAP_TOPOYmaps.mat';
% files(n).topoydata = '032221_EE14P1LN_RIG2_MAP\032221_EE14P1LN_RIG2_MAP_TOPOY\032221_EE14P1LN_RIG2_MAP_TOPOY';
% files(n).rignum = 'rig2'; %%% or 'rig1'
% files(n).monitor = 'land'; %%% for topox and y
% files(n).label = 'camk2 gc6';
% files(n).notes = 'good imaging session';
% 
% 
% % % analyzed 032221 NC 
% n=n+1;
% files(n).subj = 'EE14P1RN';
% files(n).expt = '032221';
% files(n).topox =  '032221_EE14P1RN_RIG2_MAP\032221_EE14P1RN_RIG2_MAP_TOPOX\032221_EE14P1RN_RIG2_MAP_TOPOXmaps.mat';
% files(n).topoxdata = '032221_EE14P1RN_RIG2_MAP\032221_EE14P1RN_RIG2_MAP_TOPOX\032221_EE14P1RN_RIG2_MAP_TOPOX';
% files(n).topoy =  '032221_EE14P1RN_RIG2_MAP\032221_EE14P1RN_RIG2_MAP_TOPOY\032221_EE14P1RN_RIG2_MAP_TOPOYmaps.mat';
% files(n).topoydata = '032221_EE14P1RN_RIG2_MAP\032221_EE14P1RN_RIG2_MAP_TOPOY\032221_EE14P1RN_RIG2_MAP_TOPOY';
% files(n).rignum = 'rig2'; %%% or 'rig1'
% files(n).monitor = 'land'; %%% for topox and y
% files(n).label = 'camk2 gc6';
% files(n).notes = 'good imaging session';

% % % analyzed 032221 NC 
% n=n+1;
% files(n).subj = 'EE14P1TT';
% files(n).expt = '032221';
% files(n).topox =  '032221_EE14P1TT_RIG2_MAP\032221_EE14P1TT_RIG2_MAP_TOPOX\032221_EE14P1TT_RIG2_MAP_TOPOXmaps.mat';
% files(n).topoxdata = '032221_EE14P1TT_RIG2_MAP\032221_EE14P1TT_RIG2_MAP_TOPOX\032221_EE14P1TT_RIG2_MAP_TOPOX';
% files(n).topoy =  '032221_EE14P1TT_RIG2_MAP\032221_EE14P1TT_RIG2_MAP_TOPOY\032221_EE14P1TT_RIG2_MAP_TOPOYmaps.mat';
% files(n).topoydata = '032221_EE14P1TT_RIG2_MAP\032221_EE14P1TT_RIG2_MAP_TOPOY\032221_EE14P1TT_RIG2_MAP_TOPOY';
% files(n).rignum = 'rig2'; %%% or 'rig1'
% files(n).monitor = 'land'; %%% for topox and y
% files(n).label = 'camk2 gc6';
% files(n).notes = 'good imaging session';


% % analyzed 032921 NC 
% n=n+1;
% files(n).subj = 'EE14P1LT';
% files(n).expt = '032921';
% files(n).topox =  '032921_EE14P1LT_RIG2_MAP\032921_EE14P1LT_RIG2_MAP_TOPOX\032921_EE14P1LT_RIG2_MAP_TOPOXmaps.mat';
% files(n).topoxdata = '032921_EE14P1LT_RIG2_MAP\032921_EE14P1LT_RIG2_MAP_TOPOX\032921_EE14P1LT_RIG2_MAP_TOPOX';
% files(n).topoy =  '032921_EE14P1LT_RIG2_MAP\032921_EE14P1LT_RIG2_MAP_TOPOY\032921_EE14P1LT_RIG2_MAP_TOPOYmaps.mat';
% files(n).topoydata = '032921_EE14P1LT_RIG2_MAP\032921_EE14P1LT_RIG2_MAP_TOPOY\032921_EE14P1LT_RIG2_MAP_TOPOY';
% files(n).rignum = 'rig2'; %%% or 'rig1'
% files(n).monitor = 'land'; %%% for topox and y
% files(n).label = 'camk2 gc6';
% files(n).notes = 'good imaging session';
% 
% % % analyzed 032921 NC 
% n=n+1;
% files(n).subj = 'EE14P2RT';
% files(n).expt = '032921';
% files(n).topox =  '032921_EE14P2RT_RIG2_MAP\032921_EE14P2RT_RIG2_MAP_TOPOX\032921_EE14P2RT_RIG2_MAP_TOPOXmaps.mat';
% files(n).topoxdata = '032921_EE14P2RT_RIG2_MAP\032921_EE14P2RT_RIG2_MAP_TOPOX\032921_EE14P2RT_RIG2_MAP_TOPOX';
% files(n).topoy =  '032921_EE14P2RT_RIG2_MAP\032921_EE14P2RT_RIG2_MAP_TOPOY\032921_EE14P2RT_RIG2_MAP_TOPOYmaps.mat';
% files(n).topoydata = '032921_EE14P2RT_RIG2_MAP\032921_EE14P2RT_RIG2_MAP_TOPOY\032921_EE14P2RT_RIG2_MAP_TOPOY';
% files(n).rignum = 'rig2'; %%% or 'rig1'
% files(n).monitor = 'land'; %%% for topox and y
% files(n).label = 'camk2 gc6';
% files(n).notes = 'good imaging session';


% % analyzed 032921 NC 
% n=n+1;
% files(n).subj = 'EE14P1NN';
% files(n).expt = '032921';
% files(n).topox =  '032921_EE14P1NN_RIG2_MAP\032921_EE14P1NN_RIG2_MAP_TOPOX\032921_EE14P1NN_RIG2_MAP_TOPOXmaps.mat';
% files(n).topoxdata = '032921_EE14P1NN_RIG2_MAP\032921_EE14P1NN_RIG2_MAP_TOPOX\032921_EE14P1NN_RIG2_MAP_TOPOX';
% files(n).topoy =  '032921_EE14P1NN_RIG2_MAP\032921_EE14P1NN_RIG2_MAP_TOPOY\032921_EE14P1NN_RIG2_MAP_TOPOYmaps.mat';
% files(n).topoydata = '032921_EE14P1NN_RIG2_MAP\032921_EE14P1NN_RIG2_MAP_TOPOY\032921_EE14P1NN_RIG2_MAP_TOPOY';
% files(n).rignum = 'rig2'; %%% or 'rig1'
% files(n).monitor = 'land'; %%% for topox and y
% files(n).label = 'camk2 gc6';
% files(n).notes = 'good imaging session';

% % analyzed 032921 EL 
% n=n+1;
% files(n).subj = 'EE14P1TT';
% files(n).expt = '032921';
% files(n).topox =  '032921_EE14P1TT_RIG2_2PMAP\032921_EE14P1TT_RIG2_2PMAP_TOPOX\032921_EE14P1TT_RIG2_2PMAP_TOPOX2PMAPs.mat';
% files(n).topoxdata = '032921_EE14P1TT_RIG2_2PMAP\032921_EE14P1TT_RIG2_2PMAP_TOPOX\032921_EE14P1TT_RIG2_2PMAP_TOPOX';
% files(n).topoy =  '032921_EE14P1TT_RIG2_2PMAP\032921_EE14P1TT_RIG2_2PMAP_TOPOY\032921_EE14P1TT_RIG2_2PMAP_TOPOY2PMAPs.mat';
% files(n).topoydata = '032921_EE14P1TT_RIG2_2PMAP\032921_EE14P1TT_RIG2_2PMAP_TOPOY\032921_EE14P1TT_RIG2_2PMAP_TOPOY';
% files(n).rignum = 'rig2'; %%% or 'rig1'
% files(n).monitor = 'land'; %%% for topox and y
% files(n).label = 'camk2 gc6';
% files(n).notes = 'good imaging session';

% % analyzed 040621 NC 
% n=n+1;
% files(n).subj = 'J538TT';
% files(n).expt = '040521';
% files(n).topox =  '040521_J538TT_RIG2_MAP\040521_J538TT_RIG2_MAP_TOPOX\040521_J538TT_RIG2_MAP_TOPOXmaps.mat';
% files(n).topoxdata = '040521_J538TT_RIG2_MAP\040521_J538TT_RIG2_MAP_TOPOX\040521_J538TT_RIG2_MAP_TOPOX';
% files(n).topoy =  '040521_J538TT_RIG2_MAP\040521_J538TT_RIG2_MAP_TOPOY\040521_J538TT_RIG2_MAP_TOPOYmaps.mat';
% files(n).topoydata = '040521_J538TT_RIG2_MAP\040521_J538TT_RIG2_MAP_TOPOY\040521_J538TT_RIG2_MAP_TOPOY';
% files(n).rignum = 'rig2'; %%% or 'rig1'
% files(n).monitor = 'land'; %%% for topox and y
% files(n).label = 'camk2 gc6';
% files(n).notes = 'good imaging session';
% 
% 
% % analyzed 040621 NC 
% n=n+1;
% files(n).subj = 'J538LT';
% files(n).expt = '040521';
% files(n).topox =  '040521_J538LT_RIG2_MAP\040521_J538LT_RIG2_MAP_TOPOX\040521_J538LT_RIG2_MAP_TOPOXmaps.mat';
% files(n).topoxdata = '040521_J538LT_RIG2_MAP\040521_J538LT_RIG2_MAP_TOPOX\040521_J538LT_RIG2_MAP_TOPOX';
% files(n).topoy =  '040521_J538LT_RIG2_MAP\040521_J538LT_RIG2_MAP_TOPOY\040521_J538LT_RIG2_MAP_TOPOYmaps.mat';
% files(n).topoydata = '040521_J538LT_RIG2_MAP\040521_J538LT_RIG2_MAP_TOPOY\040521_J538LT_RIG2_MAP_TOPOY';
% files(n).rignum = 'rig2'; %%% or 'rig1'
% files(n).monitor = 'land'; %%% for topox and y
% files(n).label = 'camk2 gc6';
% files(n).notes = 'good imaging session';
% 
% 
% % analyzed 040621 NC 
% n=n+1;
% files(n).subj = 'J538RT';
% files(n).expt = '040521';
% files(n).topox =  '040521_J538RT_RIG2_MAP\040521_J538RT_RIG2_MAP_TOPOX\040521_J538RT_RIG2_MAP_TOPOXmaps.mat';
% files(n).topoxdata = '040521_J538RT_RIG2_MAP\040521_J538RT_RIG2_MAP_TOPOX\040521_J538RT_RIG2_MAP_TOPOX';
% files(n).topoy =  '040521_J538RT_RIG2_MAP\040521_J538RT_RIG2_MAP_TOPOY\040521_J538RT_RIG2_MAP_TOPOYmaps.mat';
% files(n).topoydata = '040521_J538RT_RIG2_MAP\040521_J538RT_RIG2_MAP_TOPOY\040521_J538RT_RIG2_MAP_TOPOY';
% files(n).rignum = 'rig2'; %%% or 'rig1'
% files(n).monitor = 'land'; %%% for topox and y
% files(n).label = 'camk2 gc6';
% files(n).notes = 'good imaging session';

% % analyzed 041321 NC 
% n=n+1;
% files(n).subj = 'J539LT';
% files(n).expt = '041321';
% files(n).topox =  '041321_J539LT_RIG2_MAP\041321_J539LT_RIG2_MAP_TOPOX\041321_J539LT_RIG2_MAP_TOPOXmaps.mat';
% files(n).topoxdata = '041321_J539LT_RIG2_MAP\041321_J539LT_RIG2_MAP_TOPOX\041321_J539LT_RIG2_MAP_TOPOX';
% files(n).topoy =  '041321_J539LT_RIG2_MAP\041321_J539LT_RIG2_MAP_TOPOY\041321_J539LT_RIG2_MAP_TOPOYmaps.mat';
% files(n).topoydata = '041321_J539LT_RIG2_MAP\041321_J539LT_RIG2_MAP_TOPOY\041321_J539LT_RIG2_MAP_TOPOY';
% files(n).rignum = 'rig2'; %%% or 'rig1'
% files(n).monitor = 'land'; %%% for topox and y
% files(n).label = 'camk2 gc6';
% files(n).notes = 'good imaging session';
% 
% 
% % analyzed 041321 NC 
% n=n+1;
% files(n).subj = 'J539LN';
% files(n).expt = '041321';
% files(n).topox =  '041321_J539LN_RIG2_MAP\041321_J539LN_RIG2_MAP_TOPOX\041321_J539LN_RIG2_MAP_TOPOXmaps.mat';
% files(n).topoxdata = '041321_J539LN_RIG2_MAP\041321_J539LN_RIG2_MAP_TOPOX\041321_J539LN_RIG2_MAP_TOPOX';
% files(n).topoy =  '041321_J539LN_RIG2_MAP\041321_J539LN_RIG2_MAP_TOPOY\041321_J539LN_RIG2_MAP_TOPOYmaps.mat';
% files(n).topoydata = '041321_J539LN_RIG2_MAP\041321_J539LN_RIG2_MAP_TOPOY\041321_J539LN_RIG2_MAP_TOPOY';
% files(n).rignum = 'rig2'; %%% or 'rig1'
% files(n).monitor = 'land'; %%% for topox and y
% files(n).label = 'camk2 gc6';
% files(n).notes = 'good imaging session';
% analyzed 042821 EL
% n=n+1;
% files(n).subj = 'J550LT';
% files(n).expt = '042821';
% files(n).topox =  '042821_J550LT_RIG2_MAP\042821_J550LT_RIG2_MAP_TOPOX\042821_J550LT_RIG2_MAP_TOPOXmaps.mat';
% files(n).topoxdata = '042821_J550LT_RIG2_MAP\042821_J550LT_RIG2_MAP_TOPOX\042821_J550LT_RIG2_MAP_TOPOX';
% files(n).topoy =  '042821_J550LT_RIG2_MAP\042821_J550LT_RIG2_MAP_TOPOY\042821_J550LT_RIG2_MAP_TOPOYmaps.mat';
% files(n).topoydata = '042821_J550LT_RIG2_MAP\042821_J550LT_RIG2_MAP_TOPOY\042821_J550LT_RIG2_MAP_TOPOY';
% files(n).rignum = 'rig2'; %%% or 'rig1'
% files(n).monitor = 'land'; %%% for topox and y
% files(n).label = 'camk2 gc6';
% files(n).notes = 'good imaging session';
% analyzed 042821 EL
% n=n+1;
% files(n).subj = 'J550RT';
% files(n).expt = '042821';
% files(n).topox =  '042821_J550RT_RIG2_MAP\042821_J550RT_RIG2_MAP_TOPOX\042821_J550RT_RIG2_MAP_TOPOXmaps.mat';
% files(n).topoxdata = '042821_J550RT_RIG2_MAP\042821_J550RT_RIG2_MAP_TOPOX\042821_J550RT_RIG2_MAP_TOPOX';
% files(n).topoy =  '042821_J550RT_RIG2_MAP\042821_J550RT_RIG2_MAP_TOPOY\042821_J550RT_RIG2_MAP_TOPOYmaps.mat';
% files(n).topoydata = '042821_J550RT_RIG2_MAP\042821_J550RT_RIG2_MAP_TOPOY\042821_J550RT_RIG2_MAP_TOPOY';
% files(n).rignum = 'rig2'; %%% or 'rig1'
% files(n).monitor = 'land'; %%% for topox and y
% files(n).label = 'camk2 gc6';
% files(n).notes = 'good imaging session';

% analyzed 050621 NC
% n=n+1;
% files(n).subj = 'J545LT';
% files(n).expt = '050621';
% files(n).topox =  '050621_J545LT_RIG2_MAP\050621_J545LT_RIG2_MAP_TOPOX\050621_J545LT_RIG2_MAP_TOPOXmaps.mat';
% files(n).topoxdata = '050621_J545LT_RIG2_MAP\050621_J545LT_RIG2_MAP_TOPOX\050621_J545LT_RIG2_MAP_TOPOX';
% files(n).topoy =  '050621_J545LT_RIG2_MAP\050621_J545LT_RIG2_MAP_TOPOY\050621_J545LT_RIG2_MAP_TOPOYmaps.mat';
% files(n).topoydata = '050621_J545LT_RIG2_MAP\050621_J545LT_RIG2_MAP_TOPOY\050621_J545LT_RIG2_MAP_TOPOY';
% files(n).rignum = 'rig2'; %%% or 'rig1'
% files(n).monitor = 'land'; %%% for topox and y
% files(n).label = 'camk2 gc6';
% files(n).notes = 'good imaging session';
% 
% % analyzed 050621 NC
% n=n+1;
% files(n).subj = 'J545RT';
% files(n).expt = '050621';
% files(n).topox =  '050621_J545RT_RIG2_MAP\050621_J545RT_RIG2_MAP_TOPOX\050621_J545RT_RIG2_MAP_TOPOXmaps.mat';
% files(n).topoxdata = '050621_J545RT_RIG2_MAP\050621_J545RT_RIG2_MAP_TOPOX\050621_J545RT_RIG2_MAP_TOPOX';
% files(n).topoy =  '050621_J545RT_RIG2_MAP\050621_J545RT_RIG2_MAP_TOPOY\050621_J545RT_RIG2_MAP_TOPOYmaps.mat';
% files(n).topoydata = '050621_J545RT_RIG2_MAP\050621_J545RT_RIG2_MAP_TOPOY\050621_J545RT_RIG2_MAP_TOPOY';
% files(n).rignum = 'rig2'; %%% or 'rig1'
% files(n).monitor = 'land'; %%% for topox and y
% files(n).label = 'camk2 gc6';
% files(n).notes = 'good imaging session';

% analyzed 051021 NC
% n=n+1;
% files(n).subj = 'J539LN';
% files(n).expt = '051021';
% files(n).topox =  '051021_J539LN_RIG2_MAP\051021_J539LN_RIG2_MAP_TOPOX\051021_J539LN_RIG2_MAP_TOPOXmaps.mat';
% files(n).topoxdata = '051021_J539LN_RIG2_MAP\051021_J539LN_RIG2_MAP_TOPOX\051021_J539LN_RIG2_MAP_TOPOX';
% files(n).topoy =  '051021_J539LN_RIG2_MAP\051021_J539LN_RIG2_MAP_TOPOY\051021_J539LN_RIG2_MAP_TOPOYmaps.mat';
% files(n).topoydata = '051021_J539LN_RIG2_MAP\051021_J539LN_RIG2_MAP_TOPOY\051021_J539LN_RIG2_MAP_TOPOY';
% files(n).rignum = 'rig2'; %%% or 'rig1'
% files(n).monitor = 'land'; %%% for topox and y
% files(n).label = 'camk2 gc6';
% files(n).notes = 'good imaging session';

% % analyzed 051021 NC
% n=n+1;
% files(n).subj = 'J552NC';
% files(n).expt = '051721';
% files(n).topox =  '051721_J552NC_RIG2_MAP\051721_J552NC_RIG2_MAP_TOPOX\051721_J552NC_RIG2_MAP_TOPOXmaps.mat';
% files(n).topoxdata = '051721_J552NC_RIG2_MAP\051721_J552NC_RIG2_MAP_TOPOX\051721_J552NC_RIG2_MAP_TOPOX';
% files(n).topoy =  '051721_J552NC_RIG2_MAP\051721_J552NC_RIG2_MAP_TOPOY\051721_J552NC_RIG2_MAP_TOPOYmaps.mat';
% files(n).topoydata = '051721_J552NC_RIG2_MAP\051721_J552NC_RIG2_MAP_TOPOY\051721_J552NC_RIG2_MAP_TOPOY';
% files(n).rignum = 'rig2'; %%% or 'rig1'
% files(n).monitor = 'land'; %%% for topox and y
% files(n).label = 'camk2 gc6';
% files(n).notes = 'good imaging session';
% 
% % analyzed 051721 NC
% n=n+1;
% files(n).subj = 'J552RT';
% files(n).expt = '051721';
% files(n).topox =  '051721_J552RT_RIG2_MAP\051721_J552RT_RIG2_MAP_TOPOX\051721_J552RT_RIG2_MAP_TOPOXmaps.mat';
% files(n).topoxdata = '051721_J552RT_RIG2_MAP\051721_J552RT_RIG2_MAP_TOPOX\051721_J552RT_RIG2_MAP_TOPOX';
% files(n).topoy =  '051721_J552RT_RIG2_MAP\051721_J552RT_RIG2_MAP_TOPOY\051721_J552RT_RIG2_MAP_TOPOYmaps.mat';
% files(n).topoydata = '051721_J552RT_RIG2_MAP\051721_J552RT_RIG2_MAP_TOPOY\051721_J552RT_RIG2_MAP_TOPOY';
% files(n).rignum = 'rig2'; %%% or 'rig1'
% files(n).monitor = 'land'; %%% for topox and y
% files(n).label = 'camk2 gc6';
% files(n).notes = 'good imaging session';

% analyzed 053021 EL
% n=n+1;
% files(n).subj = 'J549NC';
% files(n).expt = '052421';
% files(n).topox =  '052421_J549NC_RIG2_MAP\052421_J549NC_RIG2_MAP_TOPOX\052421_J549NC_RIG2_MAP_TOPOXmaps.mat';
% files(n).topoxdata = '052421_J549NC_RIG2_MAP\052421_J549NC_RIG2_MAP_TOPOX\052421_J549NC_RIG2_MAP_TOPOX';
% files(n).topoy =  '052421_J549NC_RIG2_MAP\052421_J549NC_RIG2_MAP_TOPOY\052421_J549NC_RIG2_MAP_TOPOYmaps.mat';
% files(n).topoydata = '052421_J549NC_RIG2_MAP\052421_J549NC_RIG2_MAP_TOPOY\052421_J549NC_RIG2_MAP_TOPOY';
% files(n).rignum = 'rig2'; %%% or 'rig1'
% files(n).monitor = 'land'; %%% for topox and y
% files(n).label = 'camk2 gc6';
% files(n).notes = 'good imaging session';
% 
% n=n+1;
% files(n).subj = 'J549RT';
% files(n).expt = '052421';
% files(n).topox =  '052421_J549RT_RIG2_MAP\052421_J549RT_RIG2_MAP_TOPOX\052421_J549RT_RIG2_MAP_TOPOXmaps.mat';
% files(n).topoxdata = '052421_J549RT_RIG2_MAP\052421_J549RT_RIG2_MAP_TOPOX\052421_J549RT_RIG2_MAP_TOPOX';
% files(n).topoy =  '052421_J549RT_RIG2_MAP\052421_J549RT_RIG2_MAP_TOPOY\052421_J549RT_RIG2_MAP_TOPOYmaps.mat';
% files(n).topoydata = '052421_J549RT_RIG2_MAP\052421_J549RT_RIG2_MAP_TOPOY\052421_J549RT_RIG2_MAP_TOPOY';
% files(n).rignum = 'rig2'; %%% or 'rig1'
% files(n).monitor = 'land'; %%% for topox and y
% files(n).label = 'camk2 gc6';
% files(n).notes = 'good imaging session';

% analyzed 062121 EL
% n=n+1;
% files(n).subj = 'EE12P7RN';
% files(n).expt = '062121';
% files(n).topox =  '062121_EE12P7RN_RIG2_MAP\062121_EE12P7RN_RIG2_MAP_TOPOX\062121_EE12P7RN_RIG2_MAP_TOPOXmaps.mat';
% files(n).topoxdata = '062121_EE12P7RN_RIG2_MAP\062121_EE12P7RN_RIG2_MAP_TOPOX\062121_EE12P7RN_RIG2_MAP_TOPOX';
% files(n).topoy =  '062121_EE12P7RN_RIG2_MAP\062121_EE12P7RN_RIG2_MAP_TOPOY\062121_EE12P7RN_RIG2_MAP_TOPOYmaps.mat';
% files(n).topoydata = '062121_EE12P7RN_RIG2_MAP\062121_EE12P7RN_RIG2_MAP_TOPOY\062121_EE12P7RN_RIG2_MAP_TOPOY';
% analyzed 053121 EL
% n=n+1;
% files(n).subj = 'J546RT';
% files(n).expt = '053121';
% files(n).topox =  '053121_J546RT_RIG2_MAP\053121_J546RT_RIG2_MAP_TOPOX\053121_J546RT_RIG2_MAP_TOPOXmaps.mat';
% files(n).topoxdata = '053121_J546RT_RIG2_MAP\053121_J546RT_RIG2_MAP_TOPOX\053121_J546RT_RIG2_MAP_TOPOX';
% files(n).topoy =  '053121_J546RT_RIG2_MAP\053121_J546RT_RIG2_MAP_TOPOY\053121_J546RT_RIG2_MAP_TOPOYmaps.mat';
% files(n).topoydata = '053121_J546RT_RIG2_MAP\053121_J546RT_RIG2_MAP_TOPOY\053121_J546RT_RIG2_MAP_TOPOY';
% files(n).rignum = 'rig2'; %%% or 'rig1'
% files(n).monitor = 'land'; %%% for topox and y
% files(n).label = 'camk2 gc6';
% files(n).notes = 'good imaging session';
% 
% n=n+1;
% files(n).subj = 'J546LT';
% files(n).expt = '062121';
% files(n).topox =  '062121_J546LT_RIG2_MAP\062121_J546LT_RIG2_MAP_TOPOX\062121_J546LT_RIG2_MAP_TOPOXmaps.mat';
% files(n).topoxdata = '062121_J546LT_RIG2_MAP\062121_J546LT_RIG2_MAP_TOPOX\062121_J546LT_RIG2_MAP_TOPOX';
% files(n).topoy =  '062121_J546LT_RIG2_MAP\062121_J546LT_RIG2_MAP_TOPOY\062121_J546LT_RIG2_MAP_TOPOYmaps.mat';
% files(n).topoydata = '062121_J546LT_RIG2_MAP\062121_J546LT_RIG2_MAP_TOPOY\062121_J546LT_RIG2_MAP_TOPOY';
% files(n).rignum = 'rig2'; %%% or 'rig1'
% files(n).monitor = 'land'; %%% for topox and y
% files(n).label = 'camk2 gc6';
% files(n).notes = 'good imaging session';

% % analyzed 062121 EL
% n=n+1;
% files(n).subj = 'EE12P7LT';
% files(n).expt = '062121';
% files(n).topox =  '062121_EE12P7LT_RIG2_MAP\062121_EE12P7LT_RIG2_MAP_TOPOX\062121_EE12P7LT_RIG2_MAP_TOPOXmaps.mat';
% files(n).topoxdata = '062121_EE12P7LT_RIG2_MAP\062121_EE12P7LT_RIG2_MAP_TOPOX\062121_EE12P7LT_RIG2_MAP_TOPOX';
% files(n).topoy =  '062121_EE12P7LT_RIG2_MAP\062121_EE12P7LT_RIG2_MAP_TOPOY\062121_EE12P7LT_RIG2_MAP_TOPOYmaps.mat';
% files(n).topoydata = '062121_EE12P7LT_RIG2_MAP\062121_EE12P7LT_RIG2_MAP_TOPOY\062121_EE12P7RLT_RIG2_MAP_TOPOY';
% files(n).expt = '053121';
% files(n).topox =  '053121_J546LT_RIG2_MAP\053121_J546LT_RIG2_MAP_TOPOX\053121_J546LT_RIG2_MAP_TOPOXmaps.mat';
% files(n).topoxdata = '053121_J546LT_RIG2_MAP\053121_J546LT_RIG2_MAP_TOPOX\053121_J546LT_RIG2_MAP_TOPOX';
% files(n).topoy =  '053121_J546LT_RIG2_MAP\053121_J546LT_RIG2_MAP_TOPOY\053121_J546LT_RIG2_MAP_TOPOYmaps.mat';
% files(n).topoydata = '053121_J546LT_RIG2_MAP\053121_J546LT_RIG2_MAP_TOPOY\053121_J546LT_RIG2_MAP_TOPOY';
% files(n).rignum = 'rig2'; %%% or 'rig1'
% files(n).monitor = 'land'; %%% for topox and y
% files(n).label = 'camk2 gc6';
% files(n).notes = 'good imaging session';

% analyzed 062121 EL
% n=n+1;
% files(n).subj = 'EE12P7RN';
% files(n).expt = '062121';
% files(n).topox =  '062121_EE12P7RN_RIG2_MAP\062121_EE12P7RN_RIG2_MAP_TOPOX\062121_EE12P7RN_RIG2_MAP_TOPOXmaps.mat';
% files(n).topoxdata = '062121_EE12P7RN_RIG2_MAP\062121_EE12P7RN_RIG2_MAP_TOPOX\062121_EE12P7RN_RIG2_MAP_TOPOX';
% files(n).topoy =  '062121_EE12P7RN_RIG2_MAP\062121_EE12P7RN_RIG2_MAP_TOPOY\062121_EE12P7RN_RIG2_MAP_TOPOYmaps.mat';
% files(n).topoydata = '062121_EE12P7RN_RIG2_MAP\062121_EE12P7RN_RIG2_MAP_TOPOY\062121_EE12P7RN_RIG2_MAP_TOPOY';
% % analyzed 061121 EL/SS
% n=n+1;
% files(n).subj = 'G6H28P16LN';
% files(n).expt = '061121';
% files(n).topox =  '061121_G6H28P16LN_RIG2_MAP\061121_G6H28P16LN_RIG2_MAP_TOPOX\061121_G6H28P16LN_RIG2_MAP_TOPOXmaps.mat';
% files(n).topoxdata = '061121_G6H28P16LN_RIG2_MAP\061121_G6H28P16LN_RIG2_MAP_TOPOX\061121_G6H28P16LN_RIG2_MAP_TOPOX';
% files(n).topoy =  '061121_G6H28P16LN_RIG2_MAP\061121_G6H28P16LN_RIG2_MAP_TOPOY\061121_G6H28P16LN_RIG2_MAP_TOPOYmaps.mat';
% files(n).topoydata = '061121_G6H28P16LN_RIG2_MAP\061121_G6H28P16LN_RIG2_MAP_TOPOY\061121_G6H28P16LN_RIG2_MAP_TOPOY';
% files(n).rignum = 'rig2'; %%% or 'rig1'
% files(n).monitor = 'land'; %%% for topox and y
% files(n).label = 'camk2 gc6';
% files(n).notes = 'good imaging session';

% analyzed 061121 EL/SS
% n=n+1;
% files(n).subj = 'G6H28P16TT';
% files(n).expt = '061421';
% files(n).topox =  '061421_G6H28P16TT_RIG2_MAP\061421_G6H28P16TT_RIG2_MAP_TOPOX\061421_G6H28P16TT_RIG2_MAP_TOPOXmaps.mat';
% files(n).topoxdata = '061421_G6H28P16TT_RIG2_MAP\061421_G6H28P16TT_RIG2_MAP_TOPOX\061421_G6H28P16TT_RIG2_MAP_TOPOX';
% files(n).topoy =  '061421_G6H28P16TT_RIG2_MAP\061421_G6H28P16TT_RIG2_MAP_TOPOY\061421_G6H28P16TT_RIG2_MAP_TOPOYmaps.mat';
% files(n).topoydata = '061421_G6H28P16TT_RIG2_MAP\061421_G6H28P16TT_RIG2_MAP_TOPOY\061421_G6H28P16TT_RIG2_MAP_TOPOY';
% files(n).rignum = 'rig2'; %%% or 'rig1'
% files(n).monitor = 'land'; %%% for topox and y
% files(n).label = 'camk2 gc6';
% files(n).notes = 'good imaging session';

% analyzed 061621 SS/MS
% n=n+1;
% files(n).subj = 'G6H31P2LN';
% files(n).expt = '061521';
% files(n).topox =  '061521_G6H31P2LN_RIG2_MAP\061521_G6H31P2LN_RIG2_MAP_TOPOX\061521_G6H31P2LN_RIG2_MAP_TOPOXmaps.mat';
% files(n).topoxdata = '061521_G6H31P2LN_RIG2_MAP\061521_G6H31P2LN_RIG2_MAP_TOPOX\061521_G6H31P2LN_RIG2_MAP_TOPOX';
% files(n).topoy =  '061521_G6H31P2LN_RIG2_MAP\061521_G6H31P2LN_RIG2_MAP_TOPOY\061521_G6H31P2LN_RIG2_MAP_TOPOYmaps.mat';
% files(n).topoydata = '061521_G6H31P2LN_RIG2_MAP\061521_G6H31P2LN_RIG2_MAP_TOPOY\061521_G6H31P2LN_RIG2_MAP_TOPOY';
% files(n).rignum = 'rig2'; %%% or 'rig1'
% files(n).monitor = 'land'; %%% for topox and y
% files(n).label = 'camk2 gc6';
% files(n).notes = 'good imaging session';
% 
% n=n+1;
% files(n).subj = 'G6H31P2LNRT';
% files(n).expt = '061521';
% files(n).topox =  '061521_G6H31P2LN_RIG2_MAP\061521_G6H31P2LN_RIG2_MAP_TOPOX\061521_G6H31P2LN_RIG2_MAP_TOPOXmaps.mat';
% files(n).topoxdata = '061521_G6H31P2LN_RIG2_MAP\061521_G6H31P2LN_RIG2_MAP_TOPOX\061521_G6H31P2LN_RIG2_MAP_TOPOX';
% files(n).topoy =  '061521_G6H31P2LN_RIG2_MAP\061521_G6H31P2LN_RIG2_MAP_TOPOY\061521_G6H31P2LN_RIG2_MAP_TOPOYmaps.mat';
% files(n).topoydata = '061521_G6H31P2LN_RIG2_MAP\061521_G6H31P2LN_RIG2_MAP_TOPOY\061521_G6H31P2LN_RIG2_MAP_TOPOY';
% files(n).rignum = 'rig2'; %%% or 'rig1'
% files(n).monitor = 'land'; %%% for topox and y
% files(n).label = 'camk2 gc6';
% files(n).notes = 'good imaging session';

% n=n+1;
% files(n).subj = 'G6H31P2RN';
% files(n).expt = '061521';
% files(n).topox =  '061521_G6H31P2RN_RIG2_MAP\061521_G6H31P2RN_RIG2_MAP_TOPOX\061521_G6H31P2RN_RIG2_MAP_TOPOXmaps.mat';
% files(n).topoxdata = '061521_G6H31P2RN_RIG2_MAP\061521_G6H31P2RN_RIG2_MAP_TOPOX\061521_G6H31P2RN_RIG2_MAP_TOPOX';
% files(n).topoy =  '061521_G6H31P2RN_RIG2_MAP\061521_G6H31P2RN_RIG2_MAP_TOPOY\061521_G6H31P2RN_RIG2_MAP_TOPOYmaps.mat';
% files(n).topoydata = '061521_G6H31P2RN_RIG2_MAP\061521_G6H31P2RN_RIG2_MAP_TOPOY\061521_G6H31P2RN_RIG2_MAP_TOPOY';
% files(n).rignum = 'rig2'; %%% or 'rig1'
% files(n).monitor = 'land'; %%% for topox and y
% files(n).label = 'camk2 gc6';
% files(n).notes = 'good imaging session'

% n=n+1;
% files(n).subj = 'J539LT';
% files(n).expt = '061621';
% files(n).topox =  '061621_J539LT_RIG2_MAP\061621_J539LT_RIG2_MAP_TOPOX\061621_J539LT_RIG2_MAP_TOPOXmaps.mat';
% files(n).topoxdata = '061621_J539LT_RIG2_MAP\061621_J539LT_RIG2_MAP_TOPOX\061621_J539LT_RIG2_MAP_TOPOX';
% files(n).topoy =  '061621_J539LT_RIG2_MAP\061621_J539LT_RIG2_MAP_TOPOY\061621_J539LT_RIG2_MAP_TOPOYmaps.mat';
% files(n).topoydata = '061621_J539LT_RIG2_MAP\061621_J539LT_RIG2_MAP_TOPOY\061621_J539LT_RIG2_MAP_TOPOY';
% files(n).rignum = 'rig2'; %%% or 'rig1'
% files(n).monitor = 'land'; %%% for topox and y
% files(n).label = 'camk2 gc6';
% files(n).notes = 'good imaging session'

% n=n+1;
% files(n).subj = 'J539NC';
% files(n).expt = '061621';
% files(n).topox =  '061621_J539NC_RIG2_MAP\061621_J539NC_RIG2_MAP_TOPOX\061621_J539NC_RIG2_MAP_TOPOXmaps.mat';
% files(n).topoxdata = '061621_J539NC_RIG2_MAP\061621_J539NC_RIG2_MAP_TOPOX\061621_J539NC_RIG2_MAP_TOPOX';
% files(n).topoy =  '061621_J539NC_RIG2_MAP\061621_J539NC_RIG2_MAP_TOPOY\061621_J539NC_RIG2_MAP_TOPOYmaps.mat';
% files(n).topoydata = '061621_J539NC_RIG2_MAP\061621_J539NC_RIG2_MAP_TOPOY\061621_J539NC_RIG2_MAP_TOPOY';
% files(n).rignum = 'rig2'; %%% or 'rig1'
% files(n).monitor = 'land'; %%% for topox and y
% files(n).label = 'camk2 gc6';
% files(n).notes = 'good imaging session'

% n=n+1;
% files(n).subj = 'G6H31P2RT';
% files(n).expt = '061521';
% files(n).topox =  '061521_G6H31P2RT_RIG2_MAP\061521_G6H31P2RT_RIG2_MAP_TOPOX\061521_G6H31P2RT_RIG2_MAP_TOPOXmaps.mat';
% files(n).topoxdata = '061521_G6H31P2RT_RIG2_MAP\061521_G6H31P2RT_RIG2_MAP_TOPOX\061521_G6H31P2RT_RIG2_MAP_TOPOX';
% files(n).topoy =  '061521_G6H31P2RT_RIG2_MAP\061521_G6H31P2RT_RIG2_MAP_TOPOY\061521_G6H31P2RT_RIG2_MAP_TOPOYmaps.mat';
% files(n).topoydata = '061521_G6H31P2RT_RIG2_MAP\061521_G6H31P2RT_RIG2_MAP_TOPOY\061521_G6H31P2RT_RIG2_MAP_TOPOY';
% files(n).rignum = 'rig2'; %%% or 'rig1'
% files(n).monitor = 'land'; %%% for topox and y
% files(n).label = 'camk2 gc6';
% files(n).notes = 'good imaging session';


% n=n+1;
% files(n).subj = 'G6CK1ARN';
% files(n).expt = '062121';
% files(n).topox =  '062121_G6CK1ARN_RIG2_MAP\062121_G6CK1ARN_RIG2_MAP_TOPOX\062121_G6CK1ARN_RIG2_MAP_TOPOXmaps.mat';
% files(n).topoxdata = '062121_G6CK1ARN_RIG2_MAP\062121_G6CK1ARN_RIG2_MAP_TOPOX\062121_G6CK1ARN_RIG2_MAP_TOPOX';
% files(n).topoy =  '062121_G6CK1ARN_RIG2_MAP\062121_G6CK1ARN_RIG2_MAP_TOPOY\062121_G6CK1ARN_RIG2_MAP_TOPOYmaps.mat';
% files(n).topoydata = '062121_G6CK1ARN_RIG2_MAP\062121_G6CK1ARN_RIG2_MAP_TOPOY\062121_G6CK1ARN_RIG2_MAP_TOPOY';
% files(n).rignum = 'rig2'; %%% or 'rig1'
% files(n).monitor = 'land'; %%% for topox and y
% files(n).label = 'camk2 gc6';
% files(n).notes = 'good imaging session';
% 
% n=n+1;
% files(n).subj = 'G6CK1ALNRT';
% files(n).expt = '062121';
% files(n).topox =  '062121_G6CK1ALNRT_RIG2_MAP\062121_G6CK1ALNRT_RIG2_MAP_TOPOX\062121_G6CK1ALNRT_RIG2_MAP_TOPOXmaps.mat';
% files(n).topoxdata = '062121_G6CK1ALNRT_RIG2_MAP\062121_G6CK1ALNRT_RIG2_MAP_TOPOX\062121_G6CK1ALNRT_RIG2_MAP_TOPOX';
% files(n).topoy =  '062121_G6CK1ALNRT_RIG2_MAP\062121_G6CK1ALNRT_RIG2_MAP_TOPOY\062121_G6CK1ALNRT_RIG2_MAP_TOPOYmaps.mat';
% files(n).topoydata = '062121_G6CK1ALNRT_RIG2_MAP\062121_G6CK1ALNRT_RIG2_MAP_TOPOY\062121_G6CK1ALNRT_RIG2_MAP_TOPOY';
% files(n).rignum = 'rig2'; %%% or 'rig1'
% files(n).monitor = 'land'; %%% for topox and y
% files(n).label = 'camk2 gc6';
% files(n).notes = 'good imaging session';


% % % analyzed 062421 NC 
% n=n+1;
% files(n).subj = 'EE12P7LT';
% files(n).expt = '062121';
% files(n).topox =  '062121_EE12P7LT_RIG2_MAP\062121_EE12P7LT_RIG2_MAP_TOPOX\062121_EE12P7LT_RIG2_MAP_TOPOXmaps.mat';
% files(n).topoxdata = '062121_EE12P7LT_RIG2_MAP\062121_EE12P7LT_RIG2_MAP_TOPOX\062121_EE12P7LT_RIG2_MAP_TOPOX';
% files(n).topoy =  '062121_EE12P7LT_RIG2_MAP\062121_EE12P7LT_RIG2_MAP_TOPOY\062121_EE12P7LT_RIG2_MAP_TOPOYmaps.mat';
% files(n).topoydata = '062121_EE12P7LT_RIG2_MAP\062121_EE12P7LT_RIG2_MAP_TOPOY\062121_EE12P7LT_RIG2_MAP_TOPOY';
% files(n).rignum = 'rig2'; %%% or 'rig1'
% files(n).monitor = 'land'; %%% for topox and y
% files(n).label = 'camk2 gc6';
% files(n).notes = 'good imaging session';
% 
% n=n+1;
% files(n).subj = 'EE12P7RN';
% files(n).expt = '062121';
% files(n).topox =  '062121_EE12P7RN_RIG2_MAP\062121_EE12P7RN_RIG2_MAP_TOPOX\062121_EE12P7RN_RIG2_MAP_TOPOXmaps.mat';
% files(n).topoxdata = '062121_EE12P7RN_RIG2_MAP\062121_EE12P7RN_RIG2_MAP_TOPOX\062121_EE12P7RN_RIG2_MAP_TOPOX';
% files(n).topoy =  '062121_EE12P7RN_RIG2_MAP\062121_EE12P7RN_RIG2_MAP_TOPOY\062121_EE12P7RN_RIG2_MAP_TOPOYmaps.mat';
% files(n).topoydata = '062121_EE12P7RN_RIG2_MAP\062121_EE12P7RN_RIG2_MAP_TOPOY\062121_EE12P7RN_RIG2_MAP_TOPOY';
% files(n).rignum = 'rig2'; %%% or 'rig1'
% files(n).monitor = 'land'; %%% for topox and y
% files(n).label = 'camk2 gc6';
% files(n).notes = 'good imaging session';

% % analyzed 062821 MS 

% n=n+1;
% files(n).subj = 'J553RT';
% files(n).expt = '062821';
% files(n).topox =  '062821_J553RT_RIG2_MAP\062821_J553RT_RIG2_MAP_TOPOX\062821_J553RT_RIG2_MAP_TOPOXmaps.mat';
% files(n).topoxdata = '062821_J553RT_RIG2_MAP\062821_J553RT_RIG2_MAP_TOPOX\062821_J553RT_RIG2_MAP_TOPOX';
% files(n).topoy =  '062821_J553RT_RIG2_MAP\062821_J553RT_RIG2_MAP_TOPOY\062821_J553RT_RIG2_MAP_TOPOYmaps.mat';
% files(n).topoydata = '062821_J553RT_RIG2_MAP\062821_J553RT_RIG2_MAP_TOPOY\062821_J553RT_RIG2_MAP_TOPOY';
% files(n).rignum = 'rig2'; %%% or 'rig1'
% files(n).monitor = 'land'; %%% for topox and y
% files(n).label = 'camk2 gc6';
% files(n).notes = 'good imaging session';
% 
% n=n+1;
% files(n).subj = 'J553LT';
% files(n).expt = '062821';
% files(n).topox =  '062821_J553LT_RIG2_MAP\062821_J553LT_RIG2_MAP_TOPOX\062821_J553LT_RIG2_MAP_TOPOXmaps.mat';
% files(n).topoxdata = '062821_J553LT_RIG2_MAP\062821_J553LT_RIG2_MAP_TOPOX\062821_J553LT_RIG2_MAP_TOPOX';
% files(n).topoy =  '062821_J553LT_RIG2_MAP\062821_J553LT_RIG2_MAP_TOPOY\062821_J553LT_RIG2_MAP_TOPOYmaps.mat';
% files(n).topoydata = '062821_J553LT_RIG2_MAP\062821_J553LT_RIG2_MAP_TOPOY\062821_J553LT_RIG2_MAP_TOPOY';
% files(n).rignum = 'rig2'; %%% or 'rig1'
% files(n).monitor = 'land'; %%% for topox and y
% files(n).label = 'camk2 gc6';
% files(n).notes = 'good imaging session';
% 
% n=n+1;
% files(n).subj = 'J553LN';
% files(n).expt = '062821';
% files(n).topox =  '062821_J553LN_RIG2_MAP\062821_J553LN_RIG2_MAP_TOPOX\062821_J553LN_RIG2_MAP_TOPOXmaps.mat';
% files(n).topoxdata = '062821_J553LN_RIG2_MAP\062821_J553LN_RIG2_MAP_TOPOX\062821_J553LN_RIG2_MAP_TOPOX';
% files(n).topoy =  '062821_J553LN_RIG2_MAP\062821_J553LN_RIG2_MAP_TOPOY\062821_J553LN_RIG2_MAP_TOPOYmaps.mat';
% files(n).topoydata = '062821_J553LN_RIG2_MAP\062821_J553LN_RIG2_MAP_TOPOY\062821_J553LN_RIG2_MAP_TOPOY';
% files(n).rignum = 'rig2'; %%% or 'rig1'
% files(n).monitor = 'land'; %%% for topox and y
% files(n).label = 'camk2 gc6';
% files(n).notes = 'good imaging session';


% % analyzed 07/05/21 MS 
% n=n+1;
% files(n).subj = 'J537LT';
% files(n).expt = '070521';
% files(n).topox =  '070521_J537LT_RIG2_MAP\070521_J537LT_RIG2_MAP_TOPOX\070521_J537LT_RIG2_MAP_TOPOXmaps.mat';
% files(n).topoxdata = '070521_J537LT_RIG2_MAP\070521_J537LT_RIG2_MAP_TOPOX\070521_J537LT_RIG2_MAP_TOPOX';
% files(n).topoy =  '070521_J537LT_RIG2_MAP\070521_J537LT_RIG2_MAP_TOPOY\070521_J537LT_RIG2_MAP_TOPOYmaps.mat';
% files(n).topoydata = '070521_J537LT_RIG2_MAP\070521_J537LT_RIG2_MAP_TOPOY\070521_J537LT_RIG2_MAP_TOPOY';
% files(n).rignum = 'rig2'; %%% or 'rig1'
% files(n).monitor = 'land'; %%% for topox and y
% files(n).label = 'camk2 gc6';
% files(n).notes = 'good imaging session';
% 
% n=n+1;
% files(n).subj = 'J537RT';
% files(n).expt = '070521';
% files(n).topox =  '070521_J537RT_RIG2_MAP\070521_J537RT_RIG2_MAP_TOPOX\070521_J537RT_RIG2_MAP_TOPOXmaps.mat';
% files(n).topoxdata = '070521_J537RT_RIG2_MAP\070521_J537RT_RIG2_MAP_TOPOX\070521_J537RT_RIG2_MAP_TOPOX';
% files(n).topoy =  '070521_J537RT_RIG2_MAP\070521_J537RT_RIG2_MAP_TOPOY\070521_J537RT_RIG2_MAP_TOPOYmaps.mat';
% files(n).topoydata = '070521_J537RT_RIG2_MAP\070521_J537RT_RIG2_MAP_TOPOY\070521_J537RT_RIG2_MAP_TOPOY';
% files(n).rignum = 'rig2'; %%% or 'rig1'
% files(n).monitor = 'land'; %%% for topox and y
% files(n).label = 'camk2 gc6';
% files(n).notes = 'good imaging session';
% 
% n=n+1;
% files(n).subj = 'J538LN';
% files(n).expt = '070521';
% files(n).topox =  '070521_J538LN_RIG2_MAP\070521_J538LN_RIG2_MAP_TOPOX\070521_J538LN_RIG2_MAP_TOPOXmaps.mat';
% files(n).topoxdata = '070521_J538LN_RIG2_MAP\070521_J538LN_RIG2_MAP_TOPOX\070521_J538LN_RIG2_MAP_TOPOX';
% files(n).topoy =  '070521_J538LN_RIG2_MAP\070521_J538LN_RIG2_MAP_TOPOY\070521_J538LN_RIG2_MAP_TOPOYmaps.mat';
% files(n).topoydata = '070521_J538LN_RIG2_MAP\070521_J538LN_RIG2_MAP_TOPOY\070521_J538LN_RIG2_MAP_TOPOY';
% files(n).rignum = 'rig2'; %%% or 'rig1'
% files(n).monitor = 'land'; %%% for topox and y
% files(n).label = 'camk2 gc6';
% files(n).notes = 'good imaging session';
% 
% n=n+1;
% files(n).subj = 'J538RN';
% files(n).expt = '070521';
% files(n).topox =  '070521_J538RN_RIG2_MAP\070521_J538RN_RIG2_MAP_TOPOX\070521_J538RN_RIG2_MAP_TOPOXmaps.mat';
% files(n).topoxdata = '070521_J538RN_RIG2_MAP\070521_J538RN_RIG2_MAP_TOPOX\070521_J538RN_RIG2_MAP_TOPOX';
% files(n).topoy =  '070521_J538RN_RIG2_MAP\070521_J538RN_RIG2_MAP_TOPOY\070521_J538RN_RIG2_MAP_TOPOYmaps.mat';
% files(n).topoydata = '070521_J538RN_RIG2_MAP\070521_J538RN_RIG2_MAP_TOPOY\070521_J538RN_RIG2_MAP_TOPOY';
% files(n).rignum = 'rig2'; %%% or 'rig1'
% files(n).monitor = 'land'; %%% for topox and y
% files(n).label = 'camk2 gc6';
% files(n).notes = 'good imaging session';

% n=n+1;
% files(n).subj = 'G6H31P1TT';
% files(n).expt = '062921';
% files(n).topox =  '062921_G6H31P1TT_RIG2_MAP\062921_G6H31P1TT_RIG2_MAP_TOPOX\062921_G6H31P1TT_RIG2_MAP_TOPOXmaps.mat';
% files(n).topoxdata = '062921_G6H31P1TT_RIG2_MAP\062921_G6H31P1TT_RIG2_MAP_TOPOX\062921_G6H31P1TT_RIG2_MAP_TOPOX';
% files(n).topoy =  '062921_G6H31P1TT_RIG2_MAP\062921_G6H31P1TT_RIG2_MAP_TOPOY\062921_G6H31P1TT_RIG2_MAP_TOPOYmaps.mat';
% files(n).topoydata = '062921_G6H31P1TT_RIG2_MAP\062921_G6H31P1TT_RIG2_MAP_TOPOY\062921_G6H31P1TT_RIG2_MAP_TOPOY';
% files(n).rignum = 'rig2'; %%% or 'rig1'
% files(n).monitor = 'land'; %%% for topox and y
% files(n).label = 'camk2 gc6';
% files(n).notes = 'good imaging session';

% n=n+1;
% files(n).subj = 'J563LT';
% files(n).expt = '070521';
% files(n).topox =  '070521_J563LT_RIG2_MAP\070521_J563LT_RIG2_MAP_TOPOX\070521_J563LT_RIG2_MAP_TOPOXmaps.mat';
% files(n).topoxdata = '070521_J563LT_RIG2_MAP\070521_J563LT_RIG2_MAP_TOPOX\070521_J563LT_RIG2_MAP_TOPOX';
% files(n).topoy =  '070521_J563LT_RIG2_MAP\070521_J563LT_RIG2_MAP_TOPOY\070521_J563LT_RIG2_MAP_TOPOYmaps.mat';
% files(n).topoydata = '070521_J563LT_RIG2_MAP\070521_J563LT_RIG2_MAP_TOPOY\070521_J563LT_RIG2_MAP_TOPOY';
% files(n).rignum = 'rig2'; %%% or 'rig1'
% files(n).monitor = 'land'; %%% for topox and y
% files(n).label = 'camk2 gc6';
% files(n).notes = 'good imaging session';
% 
% n=n+1;
% files(n).subj = 'J563NC';
% files(n).expt = '070521';
% files(n).topox =  '070521_J563NC_RIG2_MAP\070521_J563NC_RIG2_MAP_TOPOX\070521_J563NC_RIG2_MAP_TOPOXmaps.mat';
% files(n).topoxdata = '070521_J563NC_RIG2_MAP\070521_J563NC_RIG2_MAP_TOPOX\070521_J563NC_RIG2_MAP_TOPOX';
% files(n).topoy =  '070521_J563NC_RIG2_MAP\070521_J563NC_RIG2_MAP_TOPOY\070521_J563NC_RIG2_MAP_TOPOYmaps.mat';
% files(n).topoydata = '070521_J563NC_RIG2_MAP\070521_J563NC_RIG2_MAP_TOPOY\070521_J563NC_RIG2_MAP_TOPOY';
% files(n).rignum = 'rig2'; %%% or 'rig1'
% files(n).monitor = 'land'; %%% for topox and y
% files(n).label = 'camk2 gc6';
% files(n).notes = 'good imaging session';
% 
% n=n+1;
% files(n).subj = 'J563RT';
% files(n).expt = '070521';
% files(n).topox =  '070521_J563RT_RIG2_MAP\070521_J563RT_RIG2_MAP_TOPOX\070521_J563RT_RIG2_MAP_TOPOXmaps.mat';
% files(n).topoxdata = '070521_J563RT_RIG2_MAP\070521_J563RT_RIG2_MAP_TOPOX\070521_J563RT_RIG2_MAP_TOPOX';
% files(n).topoy =  '070521_J563RT_RIG2_MAP\070521_J563RT_RIG2_MAP_TOPOY\070521_J563RT_RIG2_MAP_TOPOYmaps.mat';
% files(n).topoydata = '070521_J563RT_RIG2_MAP\070521_J563RT_RIG2_MAP_TOPOY\070521_J563RT_RIG2_MAP_TOPOY';
% files(n).rignum = 'rig2'; %%% or 'rig1'
% files(n).monitor = 'land'; %%% for topox and y
% files(n).label = 'camk2 gc6';
% files(n).notes = 'good imaging session';
% 
% n=n+1;
% files(n).subj = 'J563TT';
% files(n).expt = '070521';
% files(n).topox =  '070521_J563TT_RIG2_MAP\070521_J563TT_RIG2_MAP_TOPOX\070521_J563TT_RIG2_MAP_TOPOXmaps.mat';
% files(n).topoxdata = '070521_J563TT_RIG2_MAP\070521_J563TT_RIG2_MAP_TOPOX\070521_J563TT_RIG2_MAP_TOPOX';
% files(n).topoy =  '070521_J563TT_RIG2_MAP\070521_J563TT_RIG2_MAP_TOPOY\070521_J563TT_RIG2_MAP_TOPOYmaps.mat';
% files(n).topoydata = '070521_J563TT_RIG2_MAP\070521_J563TT_RIG2_MAP_TOPOY\070521_J563TT_RIG2_MAP_TOPOY';
% files(n).rignum = 'rig2'; %%% or 'rig1'
% files(n).monitor = 'land'; %%% for topox and y
% files(n).label = 'camk2 gc6';
% files(n).notes = 'good imaging session';

% analyzed 07/19/21 MS 
% n=n+1;
% files(n).subj = 'J563TT';
% files(n).expt = '071921';
% files(n).topox =  '071921_J564TT_RIG2_MAP\071921_J564TT_RIG2_MAP_TOPOX\071921_J564TT_RIG2_MAP_TOPOXmaps.mat';
% files(n).topoxdata = '071921_J564TT_RIG2_MAP\071921_J564TT_RIG2_MAP_TOPOX\071921_J564TT_RIG2_MAP_TOPOX';
% files(n).topoy =  '071921_J564TT_RIG2_MAP\071921_J564TT_RIG2_MAP_TOPOY\071921_J564TT_RIG2_MAP_TOPOYmaps.mat';
% files(n).topoydata = '071921_J564TT_RIG2_MAP\071921_J564TT_RIG2_MAP_TOPOY\071921_J564TT_RIG2_MAP_TOPOY';
% files(n).rignum = 'rig2'; %%% or 'rig1'
% files(n).monitor = 'land'; %%% for topox and y
% files(n).label = 'camk2 gc6';
% files(n).notes = 'good imaging session';
% 
% n=n+1;
% files(n).subj = 'J564RT';
% files(n).expt = '071921';
% files(n).topox =  '071921_J564RT_RIG2_MAP\071921_J564RT_RIG2_MAP_TOPOX\071921_J564RT_RIG2_MAP_TOPOXmaps.mat';
% files(n).topoxdata = '071921_J564RT_RIG2_MAP\071921_J564RT_RIG2_MAP_TOPOX\071921_J564RT_RIG2_MAP_TOPOX';
% files(n).topoy =  '071921_J564RT_RIG2_MAP\071921_J564RT_RIG2_MAP_TOPOY\071921_J564RT_RIG2_MAP_TOPOYmaps.mat';
% files(n).topoydata = '071921_J564RT_RIG2_MAP\071921_J564RT_RIG2_MAP_TOPOY\071921_J564RT_RIG2_MAP_TOPOY';
% files(n).rignum = 'rig2'; %%% or 'rig1'
% files(n).monitor = 'land'; %%% for topox and y
% files(n).label = 'camk2 gc6';
% files(n).notes = 'good imaging session';
% 
% n=n+1;
% files(n).subj = 'J564LT';
% files(n).expt = '071921';
% files(n).topox =  '071921_J564LT_RIG2_MAP\071921_J564LT_RIG2_MAP_TOPOX\071921_J564LT_RIG2_MAP_TOPOXmaps.mat';
% files(n).topoxdata = '071921_J564LT_RIG2_MAP\071921_J564LT_RIG2_MAP_TOPOX\071921_J564LT_RIG2_MAP_TOPOX';
% files(n).topoy =  '071921_J564LT_RIG2_MAP\071921_J564LT_RIG2_MAP_TOPOY\071921_J564LT_RIG2_MAP_TOPOYmaps.mat';
% files(n).topoydata = '071921_J564LT_RIG2_MAP\071921_J564LT_RIG2_MAP_TOPOY\071921_J564LT_RIG2_MAP_TOPOY';
% files(n).rignum = 'rig2'; %%% or 'rig1'
% files(n).monitor = 'land'; %%% for topox and y
% files(n).label = 'camk2 gc6';
% files(n).notes = 'good imaging session';

% % analyzed 072621 NC 
% n=n+1;
% files(n).subj = 'J560LT';
% files(n).expt = '072621';
% files(n).topox =  '072621_J560LT_RIG2_MAP\072621_J560LT_RIG2_MAP_TOPOX\072621_J560LT_RIG2_MAP_TOPOXmaps.mat';
% files(n).topoxdata = '072621_J560LT_RIG2_MAP\072621_J560LT_RIG2_MAP_TOPOX\072621_J560LT_RIG2_MAP_TOPOX';
% files(n).topoy =  '072621_J560LT_RIG2_MAP\072621_J560LT_RIG2_MAP_TOPOY\072621_J560LT_RIG2_MAP_TOPOYmaps.mat';
% files(n).topoydata = '072621_J560LT_RIG2_MAP\072621_J560LT_RIG2_MAP_TOPOY\072621_J560LT_RIG2_MAP_TOPOY';
% files(n).rignum = 'rig2'; %%% or 'rig1'
% files(n).monitor = 'land'; %%% for topox and y
% files(n).label = 'camk2 gc6';
% files(n).notes = 'good imaging session';
% 
% n=n+1;
% files(n).subj = 'J560RN';
% files(n).expt = '072621';
% files(n).topox =  '072621_J560RN_RIG2_MAP\072621_J560RN_RIG2_MAP_TOPOX\072621_J560RN_RIG2_MAP_TOPOXmaps.mat';
% files(n).topoxdata = '072621_J560RN_RIG2_MAP\072621_J560RN_RIG2_MAP_TOPOX\072621_J560RN_RIG2_MAP_TOPOX';
% files(n).topoy =  '072621_J560RN_RIG2_MAP\072621_J560RN_RIG2_MAP_TOPOY\072621_J560RN_RIG2_MAP_TOPOYmaps.mat';
% files(n).topoydata = '072621_J560RN_RIG2_MAP\072621_J560RN_RIG2_MAP_TOPOY\072621_J560RN_RIG2_MAP';
% files(n).rignum = 'rig2'; %%% or 'rig1'
% files(n).monitor = 'land'; %%% for topox and y
% files(n).label = 'camk2 gc6';
% files(n).notes = 'good imaging session';
% % 
% n=n+1;
% files(n).subj = 'J560TT';
% files(n).expt = '072621';
% files(n).topox =  '072621_J560TT_RIG2_MAP\072621_J560TT_RIG2_MAP_TOPOX\072621_J560TT_RIG2_MAP_TOPOXmaps.mat';
% files(n).topoxdata = '072621_J560TT_RIG2_MAP\072621_J560TT_RIG2_MAP_TOPOX\072621_J560TT_RIG2_MAP_TOPOX';
% files(n).topoy =  '072621_J560TT_RIG2_MAP\072621_J560TT_RIG2_MAP_TOPOY\072621_J560TT_RIG2_MAP_TOPOYmaps.mat';
% files(n).topoydata = '072621_J560TT_RIG2_MAP\072621_J560TT_RIG2_MAP_TOPOY\072621_J560TT_RIG2_MAP_TOPOY';
% files(n).rignum = 'rig2'; %%% or 'rig1'
% files(n).monitor = 'land'; %%% for topox and y
% files(n).label = 'camk2 gc6';
% files(n).notes = 'good imaging session';
% 
% n=n+1;
% files(n).subj = 'J560NC';
% files(n).expt = '072621';
% files(n).topox =  '072621_J560NC_RIG2_MAP\072621_J560NC_RIG2_MAP_TOPOX\072621_J560NC_RIG2_MAP_TOPOXmaps.mat';
% files(n).topoxdata = '072621_J560NC_RIG2_MAP\072621_J560NC_RIG2_MAP_TOPOX\072621_J560NC_RIG2_MAP_TOPOX';
% files(n).topoy =  '072621_J560NC_RIG2_MAP\072621_J560NC_RIG2_MAP_TOPOY\072621_J560NC_RIG2_MAP_TOPOYmaps.mat';
% files(n).topoydata = '072621_J560NC_RIG2_MAP\072621_J560NC_RIG2_MAP_TOPOY\072621_J560NC_RIG2_MAP_TOPOY';
% files(n).rignum = 'rig2'; %%% or 'rig1'
% files(n).monitor = 'land'; %%% for topox and y
% files(n).label = 'camk2 gc6';
% files(n).notes = 'good imaging session';

% % analyzed 080221 EL

% n=n+1;
% files(n).subj = 'J557LT';
% files(n).expt = '080221';
% files(n).topox =  '080221_J557LT_RIG2_MAP\080221_J557LT_RIG2_MAP_TOPOX\080221_J557LT_RIG2_MAP_TOPOXmaps.mat';
% files(n).topoxdata = '080221_J557LT_RIG2_MAP\080221_J557LT_RIG2_MAP_TOPOX\080221_J557LT_RIG2_MAP_TOPOX';
% files(n).topoy =  '080221_J557LT_RIG2_MAP\080221_J557LT_RIG2_MAP_TOPOY\080221_J557LT_RIG2_MAP_TOPOYmaps.mat';
% files(n).topoydata = '080221_J557LT_RIG2_MAP\080221_J557LT_RIG2_MAP_TOPOY\080221_J557LT_RIG2_MAP_TOPOY';
% files(n).rignum = 'rig2'; %%% or 'rig1'
% files(n).monitor = 'land'; %%% for topox and y
% files(n).label = 'camk2 gc6';
% files(n).notes = 'good imaging session';
% 
% n=n+1;
% files(n).subj = 'J557NC';
% files(n).expt = '080221';
% files(n).topox =  '080221_J557NC_RIG2_MAP\080221_J557NC_RIG2_MAP_TOPOX\080221_J557NC_RIG2_MAP_TOPOXmaps.mat';
% files(n).topoxdata = '080221_J557NC_RIG2_MAP\080221_J557NC_RIG2_MAP_TOPOX\080221_J557NC_RIG2_MAP_TOPOX';
% files(n).topoy =  '080221_J557NC_RIG2_MAP\080221_J557NC_RIG2_MAP_TOPOY\080221_J557NC_RIG2_MAP_TOPOYmaps.mat';
% files(n).topoydata = '080221_J557NC_RIG2_MAP\080221_J557NC_RIG2_MAP_TOPOY\080221_J557NC_RIG2_MAP_TOPOY';
% files(n).rignum = 'rig2'; %%% or 'rig1'
% files(n).monitor = 'land'; %%% for topox and y
% files(n).label = 'camk2 gc6';
% files(n).notes = 'good imaging session';
% 
% n=n+1;
% files(n).subj = 'J557RT';
% files(n).expt = '080221';
% files(n).topox =  '080221_J557RT_RIG2_MAP\080221_J557RT_RIG2_MAP_TOPOX\080221_J557RT_RIG2_MAP_TOPOXmaps.mat';
% files(n).topoxdata = '080221_J557RT_RIG2_MAP\080221_J557RT_RIG2_MAP_TOPOX\080221_J557RT_RIG2_MAP_TOPOX';
% files(n).topoy =  '080221_J557RT_RIG2_MAP\080221_J557RT_RIG2_MAP_TOPOY\080221_J557RT_RIG2_MAP_TOPOYmaps.mat';
% files(n).topoydata = '080221_J557RT_RIG2_MAP\080221_J557RT_RIG2_MAP_TOPOY\080221_J557RT_RIG2_MAP_TOPOY';
% files(n).rignum = 'rig2'; %%% or 'rig1'
% files(n).monitor = 'land'; %%% for topox and y
% files(n).label = 'camk2 gc6';
% files(n).notes = 'good imaging session';
% 
% n=n+1;
% files(n).subj = 'J557TT';
% files(n).expt = '080221';
% files(n).topox =  '080221_J557TT_RIG2_MAP\080221_J557TT_RIG2_MAP_TOPOX\080221_J557TT_RIG2_MAP_TOPOXmaps.mat';
% files(n).topoxdata = '080221_J557TT_RIG2_MAP\080221_J557TT_RIG2_MAP_TOPOX\080221_J557TT_RIG2_MAP_TOPOX';
% files(n).topoy =  '080221_J557TT_RIG2_MAP\080221_J557TT_RIG2_MAP_TOPOY\080221_J557TT_RIG2_MAP_TOPOYmaps.mat';
% files(n).topoydata = '080221_J557TT_RIG2_MAP\080221_J557TT_RIG2_MAP_TOPOY\080221_J557TT_RIG2_MAP_TOPOY';
% files(n).rignum = 'rig2'; %%% or 'rig1'
% files(n).monitor = 'land'; %%% for topox and y
% files(n).label = 'camk2 gc6';
% files(n).notes = 'good imaging session';


% % analyzed 081321 MS
% n=n+1;
% files(n).subj = 'J542LT';
% files(n).expt = '080921';
% files(n).topox =  '080921_J542LT_RIG2_MAP\080921_J542LT_RIG2_MAP_TOPOX\080921_J542LT_RIG2_MAP_TOPOXmaps.mat';
% files(n).topoxdata = '080921_J542LT_RIG2_MAP\080921_J542LT_RIG2_MAP_TOPOX\080921_J542LT_RIG2_MAP_TOPOX';
% files(n).topoy =  '080921_J542LT_RIG2_MAP\080921_J542LT_RIG2_MAP_TOPOY\080921_J542LT_RIG2_MAP_TOPOYmaps.mat';
% files(n).topoydata = '080921_J542LT_RIG2_MAP\080921_J542LT_RIG2_MAP_TOPOY\080921_J542LT_RIG2_MAP_TOPOY';
% files(n).rignum = 'rig2'; %%% or 'rig1'
% files(n).monitor = 'land'; %%% for topox and y
% files(n).label = 'camk2 gc6';
% files(n).notes = 'good imaging session';
% 
% n=n+1;
% files(n).subj = 'J542NC';
% files(n).expt = '080921';
% files(n).topox =  '080921_J542NC_RIG2_MAP\080921_J542NC_RIG2_MAP_TOPOX\080921_J542NC_RIG2_MAP_TOPOXmaps.mat';
% files(n).topoxdata = '080921_J542NC_RIG2_MAP\080921_J542NC_RIG2_MAP_TOPOX\080921_J542NC_RIG2_MAP_TOPOX';
% files(n).topoy =  '080921_J542NC_RIG2_MAP\080921_J542NC_RIG2_MAP_TOPOY\080921_J542NC_RIG2_MAP_TOPOYmaps.mat';
% files(n).topoydata = '080921_J542NC_RIG2_MAP\080921_J542NC_RIG2_MAP_TOPOY\080921_J542NC_RIG2_MAP_TOPOY';
% files(n).rignum = 'rig2'; %%% or 'rig1'
% files(n).monitor = 'land'; %%% for topox and y
% files(n).label = 'camk2 gc6';
% files(n).notes = 'good imaging session';
% 
% n=n+1;
% files(n).subj = 'J561LT';
% files(n).expt = '080921';
% files(n).topox =  '080921_J561LT_RIG2_MAP\080921_J561LT_RIG2_MAP_TOPOX\080921_J561LT_RIG2_MAP_TOPOXmaps.mat';
% files(n).topoxdata = '080921_J561LT_RIG2_MAP\080921_J561LT_RIG2_MAP_TOPOX\080921_J561LT_RIG2_MAP_TOPOX';
% files(n).topoy =  '080921_J561LT_RIG2_MAP\080921_J561LT_RIG2_MAP_TOPOY\080921_J561LT_RIG2_MAP_TOPOYmaps.mat';
% files(n).topoydata = '080921_J561LT_RIG2_MAP\080921_J561LT_RIG2_MAP_TOPOY\080921_J561LT_RIG2_MAP_TOPOY';
% files(n).rignum = 'rig2'; %%% or 'rig1'
% files(n).monitor = 'land'; %%% for topox and y
% files(n).label = 'camk2 gc6';
% files(n).notes = 'good imaging session';
% 
% n=n+1;
% files(n).subj = 'J561RT';
% files(n).expt = '080921';
% files(n).topox =  '080921_J561RT_RIG2_MAP\080921_J561RT_RIG2_MAP_TOPOX\080921_J561RT_RIG2_MAP_TOPOXmaps.mat';
% files(n).topoxdata = '080921_J561RT_RIG2_MAP\080921_J561RT_RIG2_MAP_TOPOX\080921_J561RT_RIG2_MAP_TOPOX';
% files(n).topoy =  '080921_J561RT_RIG2_MAP\080921_J561RT_RIG2_MAP_TOPOY\080921_J561RT_RIG2_MAP_TOPOYmaps.mat';
% files(n).topoydata = '080921_J561RT_RIG2_MAP\080921_J561RT_RIG2_MAP_TOPOY\080921_J561RT_RIG2_MAP_TOPOY';
% files(n).rignum = 'rig2'; %%% or 'rig1'
% files(n).monitor = 'land'; %%% for topox and y
% files(n).label = 'camk2 gc6';
% files(n).notes = 'good imaging session';

% n=n+1;
% files(n).subj = 'J548TT';
% files(n).expt = '081721';
% files(n).topox =  '081721_J548TT_RIG2_MAP\081721_J548TT_RIG2_MAP_TOPOX\081721_J548TT_RIG2_MAP_TOPOXmaps.mat';
% files(n).topoxdata = '081721_J548TT_RIG2_MAP\081721_J548TT_RIG2_MAP_TOPOX\081721_J548TT_RIG2_MAP_TOPOX';
% files(n).topoy =  '081721_J548TT_RIG2_MAP\081721_J548TT_RIG2_MAP_TOPOY\081721_J548TT_RIG2_MAP_TOPOYmaps.mat';
% files(n).topoydata = '081721_J548TT_RIG2_MAP\081721_J548TT_RIG2_MAP_TOPOY\081721_J548TT_RIG2_MAP_TOPOY';
% files(n).rignum = 'rig2'; %%% or 'rig1'
% files(n).monitor = 'land'; %%% for topox and y
% files(n).label = 'camk2 gc6';
% files(n).notes = 'good imaging session';
% 
% n=n+1;
% files(n).subj = 'J557LN';
% files(n).expt = '081721';
% files(n).topox =  '081721_J557LN_RIG2_MAP\081721_J557LN_RIG2_MAP_TOPOX\081721_J557LN_RIG2_MAP_TOPOXmaps.mat';
% files(n).topoxdata = '081721_J557LN_RIG2_MAP\081721_J557LN_RIG2_MAP_TOPOX\081721_J557LN_RIG2_MAP_TOPOX';
% files(n).topoy =  '081721_J557LN_RIG2_MAP\081721_J557LN_RIG2_MAP_TOPOY\081721_J557LN_RIG2_MAP_TOPOYmaps.mat';
% files(n).topoydata = '081721_J557LN_RIG2_MAP\081721_J557LN_RIG2_MAP_TOPOY\081721_J557LN_RIG2_MAP_TOPOY';
% files(n).rignum = 'rig2'; %%% or 'rig1'
% files(n).monitor = 'land'; %%% for topox and y
% files(n).label = 'camk2 gc6';
% files(n).notes = 'good imaging session';

% n=n+1;
% files(n).subj = 'J557RN';
% files(n).expt = '081721';
% files(n).topox =  '081721_J557RN_RIG2_MAP\081721_J557RN_RIG2_MAP_TOPOX\081721_J557RN_RIG2_MAP_TOPOXmaps.mat';
% files(n).topoxdata = '081721_J557RN_RIG2_MAP\081721_J557RN_RIG2_MAP_TOPOX\081721_J557RN_RIG2_MAP_TOPOX';
% files(n).topoy =  '081721_J557RN_RIG2_MAP\081721_J557RN_RIG2_MAP_TOPOY\081721_J557RN_RIG2_MAP_TOPOYmaps.mat';
% files(n).topoydata = '081721_J557RN_RIG2_MAP\081721_J557RN_RIG2_MAP_TOPOY\081721_J557RN_RIG2_MAP_TOPOY';
% files(n).rignum = 'rig2'; %%% or 'rig1'
% files(n).monitor = 'land'; %%% for topox and y
% files(n).label = 'camk2 gc6';
% files(n).notes = 'good imaging session';
% 
% n=n+1;
% files(n).subj = 'J539NC';
% files(n).expt = '081721';
% files(n).topox =  '081721_J539NC_RIG2_MAP\081721_J539NC_RIG2_MAP_TOPOX\081721_J539NC_RIG2_MAP_TOPOXmaps.mat';
% files(n).topoxdata = '081721_J539NC_RIG2_MAP\081721_J539NC_RIG2_MAP_TOPOX\081721_J539NC_RIG2_MAP_TOPOX';
% files(n).topoy =  '081721_J539NC_RIG2_MAP\081721_J539NC_RIG2_MAP_TOPOY\081721_J539NC_RIG2_MAP_TOPOYmaps.mat';
% files(n).topoydata = '081721_J539NC_RIG2_MAP\081721_J539NC_RIG2_MAP_TOPOY\081721_J539NC_RIG2_MAP_TOPOY';
% files(n).rignum = 'rig2'; %%% or 'rig1'
% files(n).monitor = 'land'; %%% for topox and y
% files(n).label = 'camk2 gc6';
% files(n).notes = 'good imaging session';
% 
% n=n+1;
% files(n).subj = 'HGeM1LT';
% files(n).expt = '082521';
% files(n).topox =  '082521_HGeM1LT_RIG2_MAP\082521_HGeM1LT_RIG2_MAP_TOPOX\082521_HGeM1LT_RIG2_MAP_TOPOXmaps.mat';
% files(n).topoxdata = '082521_HGeM1LT_RIG2_MAP\082521_HGeM1LT_RIG2_MAP_TOPOX\082521_HGeM1LT_RIG2_MAP_TOPOX';
% files(n).topoy =  '082521_HGeM1LT_RIG2_MAP\082521_HGeM1LT_RIG2_MAP_TOPOY\082521_HGeM1LT_RIG2_MAP_TOPOYmaps.mat';
% files(n).topoydata = '082521_HGeM1LT_RIG2_MAP\082521_HGeM1LT_RIG2_MAP_TOPOY\082521_HGeM1LT_RIG2_MAP_TOPOY';
% files(n).rignum = 'rig2'; %%% or 'rig1'
% files(n).monitor = 'land'; %%% for topox and y
% files(n).label = 'camk2 gc6';
% files(n).notes = 'good imaging session';
% 
% 
% n=n+1;
% files(n).subj = 'HGeM1RT';
% files(n).expt = '082521';
% files(n).topox =  '082521_HGeM1RT_RIG2_MAP\082521_HGeM1RT_RIG2_MAP_TOPOX\082521_HGeM1RT_RIG2_MAP_TOPOXmaps.mat';
% files(n).topoxdata = '082521_HGeM1RT_RIG2_MAP\082521_HGeM1RT_RIG2_MAP_TOPOX\082521_HGeM1RT_RIG2_MAP_TOPOX';
% files(n).topoy =  '082521_HGeM1RT_RIG2_MAP\082521_HGeM1RT_RIG2_MAP_TOPOY\082521_HGeM1RT_RIG2_MAP_TOPOYmaps.mat';
% files(n).topoydata = '082521_HGeM1RT_RIG2_MAP\082521_HGeM1RT_RIG2_MAP_TOPOY\082521_HGeM1RT_RIG2_MAP_TOPOY';
% files(n).rignum = 'rig2'; %%% or 'rig1'
% files(n).monitor = 'land'; %%% for topox and y
% files(n).label = 'camk2 gc6';
% files(n).notes = 'good imaging session';

% n=n+1;
% files(n).subj = 'HGeM2LT';
% files(n).expt = '090121';
% files(n).topox =  '090121_HGeM2LT_RIG2_MAP\090121_HGeM2LT_RIG2_MAP_TOPOX\090121_HGeM2LT_RIG2_MAP_TOPOXmaps.mat';
% files(n).topoxdata = '090121_HGeM2LT_RIG2_MAP\090121_HGeM2LT_RIG2_MAP_TOPOX\090121_HGeM2LT_RIG2_MAP_TOPOX';
% files(n).topoy =  '090121_HGeM2LT_RIG2_MAP\090121_HGeM2LT_RIG2_MAP_TOPOY\090121_HGeM2LT_RIG2_MAP_TOPOYmaps.mat';
% files(n).topoydata = '090121_HGeM2LT_RIG2_MAP\090121_HGeM2LT_RIG2_MAP_TOPOY\090121_HGeM2LT_RIG2_MAP_TOPOY';
% files(n).rignum = 'rig2'; %%% or 'rig1'
% files(n).monitor = 'land'; %%% for topox and y
% files(n).label = 'camk2 gc6';
% files(n).notes = 'good imaging session';
% 
% n=n+1;
% files(n).subj = 'J559LT';
% files(n).expt = '090121';
% files(n).topox =  '090121_J559LT_RIG2_MAP\090121_J559LT_RIG2_MAP_TOPOX\090121_J559LT_RIG2_MAP_TOPOXmaps.mat';
% files(n).topoxdata = '090121_J559LT_RIG2_MAP\090121_J559LT_RIG2_MAP_TOPOX\090121_J559LT_RIG2_MAP_TOPOX';
% files(n).topoy =  '090121_J559LT_RIG2_MAP\090121_J559LT_RIG2_MAP_TOPOY\090121_J559LT_RIG2_MAP_TOPOYmaps.mat';
% files(n).topoydata = '090121_J559LT_RIG2_MAP\090121_J559LT_RIG2_MAP_TOPOY\090121_J559LT_RIG2_MAP_TOPOY';
% files(n).rignum = 'rig2'; %%% or 'rig1'
% files(n).monitor = 'land'; %%% for topox and y
% files(n).label = 'camk2 gc6';
% files(n).notes = 'good imaging session';

% % analyzed 091221 NC 
% n=n+1;
% files(n).subj = 'J566NC';
% files(n).expt = '090821';
% files(n).topox =  '090821_J566NC_RIG2_MAP\090821_J566NC_RIG2_MAP_TOPOX\090821_J566NC_RIG2_MAP_TOPOXmaps.mat';
% files(n).topoxdata = '090821_J566NC_RIG2_MAP\090821_J566NC_RIG2_MAP_TOPOX\090821_J566NC_RIG2_MAP_TOPOX';
% files(n).topoy =  '090821_J566NC_RIG2_MAP\090821_J566NC_RIG2_MAP_TOPOY\090821_J566NC_RIG2_MAP_TOPOYmaps.mat';
% files(n).topoydata = '090821_J566NC_RIG2_MAP\090821_J566NC_RIG2_MAP_TOPOY\090821_J566NC_RIG2_MAP_TOPOY';
% files(n).rignum = 'rig2'; %%% or 'rig1'
% files(n).monitor = 'land'; %%% for topox and y
% files(n).label = 'camk2 gc6';
% files(n).notes = 'good imaging session';

% n=n+1;
% files(n).subj = 'J566LT';
% files(n).expt = '090821';
% files(n).topox =  '090821_J566LT_RIG2_MAP\090821_J566LT_RIG2_MAP_TOPOX\090821_J566LT_RIG2_MAP_TOPOXmaps.mat';
% files(n).topoxdata = '090821_J566LT_RIG2_MAP\090821_J566LT_RIG2_MAP_TOPOX\090821_J566LT_RIG2_MAP_TOPOX';
% files(n).topoy =  '090821_J566LT_RIG2_MAP\090821_J566LT_RIG2_MAP_TOPOY\090821_J566LT_RIG2_MAP_TOPOYmaps.mat';
% files(n).topoydata = '090821_J566LT_RIG2_MAP\090821_J566LT_RIG2_MAP_TOPOY\090821_J566LT_RIG2_MAP_TOPOY';
% files(n).rignum = 'rig2'; %%% or 'rig1'
% files(n).monitor = 'land'; %%% for topox and y
% files(n).label = 'camk2 gc6';
% files(n).notes = 'good imaging session';

% n=n+1;
% files(n).subj = 'HGeM4LT';
% files(n).expt = '092321';
% files(n).topox =  '092321_HGeM4LT_RIG2_MAP\092321_HGeM4LT_RIG2_MAP_TOPOX\092321_HGeM4LT_RIG2_MAP_TOPOXmaps.mat';
% files(n).topoxdata = '092321_HGeM4LT_RIG2_MAP\092321_HGeM4LT_RIG2_MAP_TOPOX\092321_HGeM4LT_RIG2_MAP_TOPOX_00001';
% files(n).topoy =  '092321_HGeM4LT_RIG2_MAP\092321_HGeM4LT_RIG2_MAP_TOPOY\092321_HGeM4LT_RIG2_MAP_TOPOYmaps.mat';
% files(n).topoydata = '092321_HGeM4LT_RIG2_MAP\092321_HGeM4LT_RIG2_MAP_TOPOY\092321_HGeM4LT_RIG2_MAP_TOPOY_00001';
% files(n).rignum = 'rig2'; %%% or 'rig1'
% files(n).monitor = 'land'; %%% for topox and y
% files(n).label = 'camk2 gc6';
% files(n).notes = 'good imaging session';

% n=n+1;
% files(n).subj = 'HGeM4LT';
% files(n).expt = '092321';
% files(n).topox =  '092321_HGeM4LT_RIG2_MAP\092321_HGeM4LT_RIG2_MAP_TOPOX\092321_HGeM4LT_RIG2_MAP_TOPOXmaps.mat';
% files(n).topoxdata = '092321_HGeM4LT_RIG2_MAP\092321_HGeM4LT_RIG2_MAP_TOPOX\092321_HGeM4LT_RIG2_MAP_TOPOX_00001';
% files(n).topoy =  '092321_HGeM4LT_RIG2_MAP\092321_HGeM4LT_RIG2_MAP_TOPOY\092321_HGeM4LT_RIG2_MAP_TOPOYmaps.mat';
% files(n).topoydata = '092321_HGeM4LT_RIG2_MAP\092321_HGeM4LT_RIG2_MAP_TOPOY\092321_HGeM4LT_RIG2_MAP_TOPOY_00001';
% files(n).rignum = 'rig2'; %%% or 'rig1'
% files(n).monitor = 'land'; %%% for topox and y
% files(n).label = 'camk2 gc6';
% files(n).notes = 'good imaging session';

% n=n+1;
% files(n).subj = 'HGeM4NC';
% files(n).expt = '092421';
% files(n).topox =  '092421_HGeM4NC_RIG2_MAP\092421_HGeM4NC_RIG2_MAP_TOPOX\092421_HGeM4NC_RIG2_MAP_TOPOXmaps.mat';
% files(n).topoxdata = '092421_HGeM4NC_RIG2_MAP\092421_HGeM4NC_RIG2_MAP_TOPOX\092421_HGeM4NC_RIG2_MAP_TOPOX_00001';
% files(n).topoy =  '092421_HGeM4NC_RIG2_MAP\092421_HGeM4NC_RIG2_MAP_TOPOY\092421_HGeM4NC_RIG2_MAP_TOPOYmaps.mat';
% files(n).topoydata = '092421_HGeM4NC_RIG2_MAP\092421_HGeM4NC_RIG2_MAP_TOPOY\092421_HGeM4NC_RIG2_MAP_TOPOY_00001';
% files(n).rignum = 'rig2'; %%% or 'rig1'
% files(n).monitor = 'land'; %%% for topox and y
% files(n).label = 'camk2 gc6';
% files(n).notes = 'good imaging session';
% 
% n=n+1;
% files(n).subj = 'HGeM4TT';
% files(n).expt = '092421';
% files(n).topox =  '092421_HGeM4TT_RIG2_MAP\092421_HGeM4TT_RIG2_MAP_TOPOX\092421_HGeM4TT_RIG2_MAP_TOPOXmaps.mat';
% files(n).topoxdata = '092421_HGeM4TT_RIG2_MAP\092421_HGeM4TT_RIG2_MAP_TOPOX\092421_HGeM4TT_RIG2_MAP_TOPOX_00001';
% files(n).topoy =  '092421_HGeM4TT_RIG2_MAP\092421_HGeM4TT_RIG2_MAP_TOPOY\092421_HGeM4TT_RIG2_MAP_TOPOYmaps.mat';
% files(n).topoydata = '092421_HGeM4TT_RIG2_MAP\092421_HGeM4TT_RIG2_MAP_TOPOY\092421_HGeM4TT_RIG2_MAP_TOPOY_00001';
% files(n).rignum = 'rig2'; %%% or 'rig1'
% files(n).monitor = 'land'; %%% for topox and y
% files(n).label = 'camk2 gc6';
% files(n).notes = 'good imaging session';

% n=n+1;
% files(n).subj = 'J566LN';
% files(n).expt = '092621';
% files(n).topox =  '092621_J566LN_RIG2_MAP\092621_J566LN_RIG2_MAP_TOPOX\092621_J566LN_RIG2_MAP_TOPOXmaps.mat';
% files(n).topoxdata = '092621_J566LN_RIG2_MAP\092621_J566LN_RIG2_MAP_TOPOX\092621_J566LN_RIG2_MAP_TOPOX_0001';
% files(n).topoy =  '092621_J566LN_RIG2_MAP\092621_J566LN_RIG2_MAP_TOPOY\092621_J566LN_RIG2_MAP_TOPOYmaps.mat';
% files(n).topoydata = '092621_J566LN_RIG2_MAP\092621_J566LN_RIG2_MAP_TOPOY\092621_J566LN_RIG2_MAP_TOPOY_0001';
% files(n).rignum = 'rig2'; %%% or 'rig1'
% files(n).monitor = 'land'; %%% for topox and y
% files(n).label = 'camk2 gc6';
% files(n).notes = 'good imaging session';

% n=n+1;
% files(n).subj = 'J566RT';
% files(n).expt = '092621';
% files(n).topox =  '092621_J566RT_RIG2_MAP\092621_J566RT_RIG2_MAP_TOPOX\092621_J566RT_RIG2_MAP_TOPOXmaps.mat';
% files(n).topoxdata = '092621_J566RT_RIG2_MAP\092621_J566RT_RIG2_MAP_TOPOX\092621_J566RT_RIG2_MAP_TOPOX_0001';
% files(n).topoy =  '092621_J566RT_RIG2_MAP\092621_J566RT_RIG2_MAP_TOPOY\092621_J566RT_RIG2_MAP_TOPOYmaps.mat';
% files(n).topoydata = '092621_J566RT_RIG2_MAP\092621_J566RT_RIG2_MAP_TOPOY\092621_J566RT_RIG2_MAP_TOPOY_0001';
% files(n).rignum = 'rig2'; %%% or 'rig1'
% files(n).monitor = 'land'; %%% for topox and y
% files(n).label = 'camk2 gc6';
% files(n).notes = 'good imaging session';
% 
% %%
% n=n+1;
% files(n).subj = 'J559RT';
% files(n).expt = '092721';
% files(n).topox =  '092721_J559RT_RIG2_MAP\092721_J559RT_RIG2_MAP_TOPOX\092721_J559RT_RIG2_MAP_TOPOXmaps.mat';
% files(n).topoxdata = '092721_J559RT_RIG2_MAP\092721_J559RT_RIG2_MAP_TOPOX\092721_J559RT_RIG2_MAP_TOPOX_00001';
% files(n).topoy =  '092721_J559RT_RIG2_MAP\092721_J559RT_RIG2_MAP_TOPOY\092721_J559RT_RIG2_MAP_TOPOYmaps.mat';
% files(n).topoydata = '092721_J559RT_RIG2_MAP\092721_J559RT_RIG2_MAP_TOPOY\092721_J559RT_RIG2_MAP_TOPOY_00001';
% files(n).rignum = 'rig2'; %%% or 'rig1'
% files(n).monitor = 'land'; %%% for topox and y
% files(n).label = 'camk2 gc6';
% files(n).notes = 'good imaging session';
% 
% %%
% n=n+1;
% files(n).subj = 'J559TT';
% files(n).expt = '092721';
% files(n).topox =  '092721_J559TT_RIG2_MAP\092721_J559TT_RIG2_MAP_TOPOX\092721_J559TT_RIG2_MAP_TOPOXmaps.mat';
% files(n).topoxdata = '092721_J559TT_RIG2_MAP\092721_J559TT_RIG2_MAP_TOPOX\092721_J559TT_RIG2_MAP_TOPOX_00001';
% files(n).topoy =  '092721_J559TT_RIG2_MAP\092721_J559TT_RIG2_MAP_TOPOY\092721_J559TT_RIG2_MAP_TOPOYmaps.mat';
% files(n).topoydata = '092721_J559TT_RIG2_MAP\092721_J559TT_RIG2_MAP_TOPOY\092721_J559TT_RIG2_MAP_TOPOY_00001';
% files(n).rignum = 'rig2'; %%% or 'rig1'
% files(n).monitor = 'land'; %%% for topox and y
% files(n).label = 'camk2 gc6';
% files(n).notes = 'good imaging session';
%% 

% n=n+1;
% files(n).subj = 'J583LT';
% files(n).expt = '100421';
% files(n).topox =  '100421_J583LT_RIG2_MAP\100421_J583LT_RIG2_MAP_TOPOX\100421_J583LT_RIG2_MAP_TOPOXmaps.mat';
% files(n).topoxdata = '100421_J583LT_RIG2_MAP\100421_J583LT_RIG2_MAP_TOPOX\100421_J583LT_RIG2_MAP_TOPOX_0001';
% files(n).topoy =  '100421_J583LT_RIG2_MAP\100421_J583LT_RIG2_MAP_TOPOY\100421_J583LT_RIG2_MAP_TOPOYmaps.mat';
% files(n).topoydata = '100421_J583LT_RIG2_MAP\100421_J583LT_RIG2_MAP_TOPOY\100421_J583LT_RIG2_MAP_TOPOY_0001';
% files(n).rignum = 'rig2'; %%% or 'rig1'
% files(n).monitor = 'land'; %%% for topox and y
% files(n).label = 'camk2 gc6';
% files(n).notes = 'good imaging session';
% %% 
% 
% n=n+1;
% files(n).subj = 'J583NC';
% files(n).expt = '100421';
% files(n).topox =  '100421_J583NC_RIG2_MAP\100421_J583NC_RIG2_MAP_TOPOX\100421_J583NC_RIG2_MAP_TOPOXmaps.mat';
% files(n).topoxdata = '100421_J583NC_RIG2_MAP\100421_J583NC_RIG2_MAP_TOPOX\100421_J583NC_RIG2_MAP_TOPOX_0001';
% files(n).topoy =  '100421_J583NC_RIG2_MAP\100421_J583NC_RIG2_MAP_TOPOY\100421_J583NC_RIG2_MAP_TOPOYmaps.mat';
% files(n).topoydata = '100421_J583NC_RIG2_MAP\100421_J583NC_RIG2_MAP_TOPOY\100421_J583NC_RIG2_MAP_TOPOY_0001';
% files(n).rignum = 'rig2'; %%% or 'rig1'
% files(n).monitor = 'land'; %%% for topox and y
% files(n).label = 'camk2 gc6';
% files(n).notes = 'good imaging session';


% n=n+1;
% files(n).subj = 'J582LT';
% files(n).expt = '100421';
% files(n).topox =  '100421_J582LT_RIG2_MAP\100421_J582LT_RIG2_MAP_TOPOX\100421_J582LT_RIG2_MAP_TOPOXmaps.mat';
% files(n).topoxdata = '100421_J582LT_RIG2_MAP\100421_J582LT_RIG2_MAP_TOPOX\100421_J582LT_RIG2_MAP_TOPOX_00001';
% files(n).topoy =  '100421_J582LT_RIG2_MAP\100421_J582LT_RIG2_MAP_TOPOY\100421_J582LT_RIG2_MAP_TOPOYmaps.mat';
% files(n).topoydata = '100421_J582LT_RIG2_MAP\100421_J582LT_RIG2_MAP_TOPOY\100421_J582LT_RIG2_MAP_TOPOY_00001';
% files(n).rignum = 'rig2'; %%% or 'rig1'
% files(n).monitor = 'land'; %%% for topox and y
% files(n).label = 'camk2 gc6';
% files(n).notes = 'good imaging session';
% 
% n=n+1;
% files(n).subj = 'J582RT';
% files(n).expt = '100421';
% files(n).topox =  '100421_J582RT_RIG2_MAP\100421_J582RT_RIG2_MAP_TOPOX\100421_J582RT_RIG2_MAP_TOPOXmaps.mat';
% files(n).topoxdata = '100421_J582RT_RIG2_MAP\100421_J582RT_RIG2_MAP_TOPOX\100421_J582RT_RIG2_MAP_TOPOX_00001';
% files(n).topoy =  '100421_J582RT_RIG2_MAP\100421_J582RT_RIG2_MAP_TOPOY\100421_J582RT_RIG2_MAP_TOPOYmaps.mat';
% files(n).topoydata = '100421_J582RT_RIG2_MAP\100421_J582RT_RIG2_MAP_TOPOY\100421_J582RT_RIG2_MAP_TOPOY_00001';
% files(n).rignum = 'rig2'; %%% or 'rig1'
% files(n).monitor = 'land'; %%% for topox and y
% files(n).label = 'camk2 gc6';
% files(n).notes = 'good imaging session';

% %%
% n=n+1;
% files(n).subj = 'J569NC';
% files(n).expt = '101121';
% files(n).topox =  '101121_J569NC_RIG2_MAP\101121_J569NC_RIG2_MAP_TOPOX\101121_J569NC_RIG2_MAP_TOPOXmaps.mat';
% files(n).topoxdata = '101121_J569NC_RIG2_MAP\101121_J569NC_RIG2_MAP_TOPOX\101121_J569NC_RIG2_MAP_TOPOX_00001';
% files(n).topoy =  '101121_J569NC_RIG2_MAP\101121_J569NC_RIG2_MAP_TOPOY\101121_J569NC_RIG2_MAP_TOPOYmaps.mat';
% files(n).topoydata = '101121_J569NC_RIG2_MAP\101121_J569NC_RIG2_MAP_TOPOY\101121_J569NC_RIG2_MAP_TOPOY_00001';
% files(n).rignum = 'rig2'; %%% or 'rig1'
% files(n).monitor = 'land'; %%% for topox and y
% files(n).label = 'camk2 gc6';
% files(n).notes = 'good imaging session';
% 
% %%
% n=n+1;
% files(n).subj = 'J569LT';
% files(n).expt = '101121';
% files(n).topox =  '101121_J569LT_RIG2_MAP\101121_J569LT_RIG2_MAP_TOPOX\101121_J569LT_RIG2_MAP_TOPOXmaps.mat';
% files(n).topoxdata = '101121_J569LT_RIG2_MAP\101121_J569LT_RIG2_MAP_TOPOX\101121_J569LT_RIG2_MAP_TOPOX_00001';
% files(n).topoy =  '101121_J569LT_RIG2_MAP\101121_J569LT_RIG2_MAP_TOPOY\101121_J569LT_RIG2_MAP_TOPOYmaps.mat';
% files(n).topoydata = '101121_J569LT_RIG2_MAP\101121_J569LT_RIG2_MAP_TOPOY\101121_J569LT_RIG2_MAP_TOPOY_00001';
% files(n).rignum = 'rig2'; %%% or 'rig1'
% files(n).monitor = 'land'; %%% for topox and y
% files(n).label = 'camk2 gc6';
% files(n).notes = 'good imaging session';

% n=n+1;
% files(n).subj = 'J584LT';
% files(n).expt = '101121';
% files(n).topox =  '101121_J584LT_RIG2_MAP\101121_J584LT_RIG2_MAP_TOPOX\101121_J584LT_RIG2_MAP_TOPOXmaps.mat';
% files(n).topoxdata = '101121_J584LT_RIG2_MAP\101121_J584LT_RIG2_MAP_TOPOX\101121_J584LT_RIG2_MAP_TOPOX_00001';
% files(n).topoy =  '101121_J584LT_RIG2_MAP\101121_J584LT_RIG2_MAP_TOPOY\101121_J584LT_RIG2_MAP_TOPOYmaps.mat';
% files(n).topoydata = '101121_J584LT_RIG2_MAP\101121_J584LT_RIG2_MAP_TOPOY\101121_J584LT_RIG2_MAP_TOPOY_00001';
% files(n).rignum = 'rig2'; %%% or 'rig1'
% files(n).monitor = 'land'; %%% for topox and y
% files(n).label = 'camk2 gc6';
% files(n).notes = 'good imaging session';
% 
% n=n+1;
% files(n).subj = 'J584RT';
% files(n).expt = '101121';
% files(n).topox =  '101121_J584RT_RIG2_MAP\101121_J584RT_RIG2_MAP_TOPOX\101121_J584RT_RIG2_MAP_TOPOXmaps.mat';
% files(n).topoxdata = '101121_J584RT_RIG2_MAP\101121_J584RT_RIG2_MAP_TOPOX\101121_J584RT_RIG2_MAP_TOPOX_00001';
% files(n).topoy =  '101121_J584RT_RIG2_MAP\101121_J584RT_RIG2_MAP_TOPOY\101121_J584RT_RIG2_MAP_TOPOYmaps.mat';
% files(n).topoydata = '101121_J584RT_RIG2_MAP\101121_J584RT_RIG2_MAP_TOPOY\101121_J584RT_RIG2_MAP_TOPOY_00001';
% files(n).rignum = 'rig2'; %%% or 'rig1'
% files(n).monitor = 'land'; %%% for topox and y
% files(n).label = 'camk2 gc6';
% files(n).notes = 'good imaging session';

% n=n+1;
% files(n).subj = 'J571LT';
% files(n).expt = '101121';
% files(n).topox =  '101121_J571LT_RIG2_MAP\101121_J571LT_RIG2_MAP_TOPOX\101121_J571LT_RIG2_MAP_TOPOXmaps.mat';
% files(n).topoxdata = '101121_J571LT_RIG2_MAP\101121_J571LT_RIG2_MAP_TOPOX\101121_J571LT_RIG2_MAP_TOPOX_0001';
% files(n).topoy =  '101121_J571LT_RIG2_MAP\101121_J571LT_RIG2_MAP_TOPOY\101121_J571LT_RIG2_MAP_TOPOYmaps.mat';
% files(n).topoydata = '101121_J571LT_RIG2_MAP\101121_J571LT_RIG2_MAP_TOPOY\101121_J571LT_RIG2_MAP_TOPOY_0001';
% files(n).rignum = 'rig2'; %%% or 'rig1'
% files(n).monitor = 'land'; %%% for topox and y
% files(n).label = 'camk2 gc6';
% files(n).notes = 'good imaging session';
% 
% n=n+1;
% files(n).subj = 'J571NC';
% files(n).expt = '101121';
% files(n).topox =  '101121_J571NC_RIG2_MAP\101121_J571NC_RIG2_MAP_TOPOX\101121_J571NC_RIG2_MAP_TOPOXmaps.mat';
% files(n).topoxdata = '101121_J571NC_RIG2_MAP\101121_J571NC_RIG2_MAP_TOPOX\101121_J571NC_RIG2_MAP_TOPOX_0001';
% files(n).topoy =  '101121_J571NC_RIG2_MAP\101121_J571NC_RIG2_MAP_TOPOY\101121_J571NC_RIG2_MAP_TOPOYmaps.mat';
% files(n).topoydata = '101121_J571NC_RIG2_MAP\101121_J571NC_RIG2_MAP_TOPOY\101121_J571NC_RIG2_MAP_TOPOY_0001';
% files(n).rignum = 'rig2'; %%% or 'rig1'
% files(n).monitor = 'land'; %%% for topox and y
% files(n).label = 'camk2 gc6';
% files(n).notes = 'good imaging session';
%% 

% n=n+1;
% files(n).subj = 'G6CK1bLT';
% files(n).expt = '101221';
% files(n).topox =  '101221_G6CK1bLT_RIG2_MAP\101221_G6CK1bLT_RIG2_MAP_TOPOX\101221_G6CK1bLT_RIG2_MAP_TOPOXmaps.mat';
% files(n).topoxdata = '101221_G6CK1bLT_RIG2_MAP\101221_G6CK1bLT_RIG2_MAP_TOPOX\101221_G6CK1bLT_RIG2_MAP_TOPOX_0001';
% files(n).topoy =  '101221_G6CK1bLT_RIG2_MAP\101221_G6CK1bLT_RIG2_MAP_TOPOY\101221_G6CK1bLT_RIG2_MAP_TOPOYmaps.mat';
% files(n).topoydata = '101221_G6CK1bLT_RIG2_MAP\101221_G6CK1bLT_RIG2_MAP_TOPOY\101221_G6CK1bLT_RIG2_MAP_TOPOY_0001';
% files(n).rignum = 'rig2'; %%% or 'rig1'
% files(n).monitor = 'land'; %%% for topox and y
% files(n).label = 'camk2 gc6';
% files(n).notes = 'good imaging session';
% 
% n=n+1;
% files(n).subj = 'G6CK1bRT';
% files(n).expt = '101221';
% files(n).topox =  '101221_G6CK1bRT_RIG2_MAP\101221_G6CK1bRT_RIG2_MAP_TOPOX\101221_G6CK1bRT_RIG2_MAP_TOPOXmaps.mat';
% files(n).topoxdata = '101221_G6CK1bRT_RIG2_MAP\101221_G6CK1bRT_RIG2_MAP_TOPOX\101221_G6CK1bRT_RIG2_MAP_TOPOX_0001';
% files(n).topoy =  '101221_G6CK1bRT_RIG2_MAP\101221_G6CK1bRT_RIG2_MAP_TOPOY\101221_G6CK1bRT_RIG2_MAP_TOPOYmaps.mat';
% files(n).topoydata = '101221_G6CK1bRT_RIG2_MAP\101221_G6CK1bRT_RIG2_MAP_TOPOY\101221_G6CK1bRT_RIG2_MAP_TOPOY_0001';
% files(n).rignum = 'rig2'; %%% or 'rig1'
% files(n).monitor = 'land'; %%% for topox and y
% files(n).label = 'camk2 gc6';
% files(n).notes = 'good imaging session';

%% 
% 
% n=n+1;
% files(n).subj = 'J570LT';
% files(n).expt = '101821';
% files(n).topox =  '101821_J570LT_RIG2_MAP\101821_J570LT_RIG2_MAP_TOPOX\101821_J570LT_RIG2_MAP_TOPOXmaps.mat';
% files(n).topoxdata = '101821_J570LT_RIG2_MAP\101821_J570LT_RIG2_MAP_TOPOX\101821_J570LT_RIG2_MAP_TOPOX_00001';
% files(n).topoy =  '101821_J570LT_RIG2_MAP\101821_J570LT_RIG2_MAP_TOPOY\101821_J570LT_RIG2_MAP_TOPOYmaps.mat';
% files(n).topoydata = '101821_J570LT_RIG2_MAP\101821_J570LT_RIG2_MAP_TOPOY\101821_J570LT_RIG2_MAP_TOPOY_00001';
% files(n).rignum = 'rig2'; %%% or 'rig1'
% files(n).monitor = 'land'; %%% for topox and y
% files(n).label = 'camk2 gc6';
% files(n).notes = 'good imaging session';
% 
% n=n+1;
% files(n).subj = 'J570NC';
% files(n).expt = '101821';
% files(n).topox =  '101821_J570NC_RIG2_MAP\101821_J570NC_RIG2_MAP_TOPOX\101821_J570NC_RIG2_MAP_TOPOXmaps.mat';
% files(n).topoxdata = '101821_J570NC_RIG2_MAP\101821_J570NC_RIG2_MAP_TOPOX\101821_J570NC_RIG2_MAP_TOPOX_00001';
% files(n).topoy =  '101821_J570NC_RIG2_MAP\101821_J570NC_RIG2_MAP_TOPOY\101821_J570NC_RIG2_MAP_TOPOYmaps.mat';
% files(n).topoydata = '101821_J570NC_RIG2_MAP\101821_J570NC_RIG2_MAP_TOPOY\101821_J570NC_RIG2_MAP_TOPOY_00001';
% files(n).rignum = 'rig2'; %%% or 'rig1'
% files(n).monitor = 'land'; %%% for topox and y
% files(n).label = 'camk2 gc6';
% files(n).notes = 'good imaging session';
 
%
% n=n+1;
% files(n).subj = 'J569NC';
% files(n).expt = '101821';
% files(n).topox =  '101821_J569NC_RIG2_MAP\101821_J569NC_RIG2_MAP_TOPOX\101821_J569NC_RIG2_MAP_TOPOXmaps.mat';
% files(n).topoxdata = '101821_J569NC_RIG2_MAP\101821_J569NC_RIG2_MAP_TOPOX\101821_J569NC_RIG2_MAP_TOPOX_0001';
% files(n).topoy =  '101821_J569NC_RIG2_MAP\101821_J569NC_RIG2_MAP_TOPOY\101821_J569NC_RIG2_MAP_TOPOYmaps.mat';
% files(n).topoydata = '101821_J569NC_RIG2_MAP\101821_J569NC_RIG2_MAP_TOPOY\101821_J569NC_RIG2_MAP_TOPOY_0001';
% files(n).rignum = 'rig2'; %%% or 'rig1'
% files(n).monitor = 'land'; %%% for topox and y
% files(n).label = 'camk2 gc6';
% files(n).notes = 'good imaging session';
% 
% n=n+1;
% files(n).subj = 'J569LT';
% files(n).expt = '101821';
% files(n).topox =  '101821_J569LT_RIG2_MAP\101821_J569LT_RIG2_MAP_TOPOX\101821_J569LT_RIG2_MAP_TOPOXmaps.mat';
% files(n).topoxdata = '101821_J569LT_RIG2_MAP\101821_J569LT_RIG2_MAP_TOPOX\101821_J569LT_RIG2_MAP_TOPOX_0001';
% files(n).topoy =  '101821_J569LT_RIG2_MAP\101821_J569LT_RIG2_MAP_TOPOY\101821_J569LT_RIG2_MAP_TOPOYmaps.mat';
% files(n).topoydata = '101821_J569LT_RIG2_MAP\101821_J569LT_RIG2_MAP_TOPOY\101821_J569LT_RIG2_MAP_TOPOY_0001';
% files(n).rignum = 'rig2'; %%% or 'rig1'
% files(n).monitor = 'land'; %%% for topox and y
% files(n).label = 'camk2 gc6';
% files(n).notes = 'good imaging session';

% 
% n=n+1;
% files(n).subj = 'G6H21P8TT';
% files(n).expt = '102521';
% files(n).topox =  '102521_G6H21P8TT_RIG2_MAP\102521_G6H21P8TT_RIG2_MAP_TOPOX\102521_G6H21P8TT_RIG2_MAP_TOPOXmaps.mat';
% files(n).topoxdata = '102521_G6H21P8TT_RIG2_MAP\102521_G6H21P8TT_RIG2_MAP_TOPOX\102521_G6H21P8TT_RIG2_MAP_TOPOX_0001';
% files(n).topoy =  '102521_G6H21P8TT_RIG2_MAP\102521_G6H21P8TT_RIG2_MAP_TOPOY\102521_G6H21P8TT_RIG2_MAP_TOPOYmaps.mat';
% files(n).topoydata = '102521_G6H21P8TT_RIG2_MAP\102521_G6H21P8TT_RIG2_MAP_TOPOY\102521_G6H21P8TT_RIG2_MAP_TOPOY_0001';
% files(n).rignum = 'rig2'; %%% or 'rig1'
% files(n).monitor = 'land'; %%% for topox and y
% files(n).label = 'camk2 gc6';
% files(n).notes = 'good imaging session';

% n=n+1;
% files(n).subj = 'PVCHA9RT';
% files(n).expt = '102621';
% files(n).topox =  '102621_PVCHA9RT_RIG2_MAP\102621_PVCHA9RT_RIG2_MAP_TOPOX\102621_PVCHA9RT_RIG2_MAP_TOPOXmaps.mat';
% files(n).topoxdata = '102621_PVCHA9RT_RIG2_MAP\102621_PVCHA9RT_RIG2_MAP_TOPOX\102621_PVCHA9RT_RIG2_MAP_TOPOX_0001';
% files(n).topoy =  '102621_PVCHA9RT_RIG2_MAP\102621_PVCHA9RT_RIG2_MAP_TOPOY\102621_PVCHA9RT_RIG2_MAP_TOPOYmaps.mat';
% files(n).topoydata = '102621_PVCHA9RT_RIG2_MAP\102621_PVCHA9RT_RIG2_MAP_TOPOY\102621_PVCHA9RT_RIG2_MAP_TOPOY_0001';
% files(n).rignum = 'rig2'; %%% or 'rig1'
% files(n).monitor = 'land'; %%% for topox and y
% files(n).label = 'camk2 gc6';
% files(n).notes = 'good imaging session';
% 
% n=n+1;
% files(n).subj = 'PVCHA9LT';
% files(n).expt = '102621';
% files(n).topox =  '102621_PVCHA9LT_RIG2_MAP\102621_PVCHA9LT_RIG2_MAP_TOPOX\102621_PVCHA9LT_RIG2_MAP_TOPOXmaps.mat';
% files(n).topoxdata = '102621_PVCHA9LT_RIG2_MAP\102621_PVCHA9LT_RIG2_MAP_TOPOX\102621_PVCHA9LT_RIG2_MAP_TOPOX_0001';
% files(n).topoy =  '102621_PVCHA9LT_RIG2_MAP\102621_PVCHA9LT_RIG2_MAP_TOPOY\102621_PVCHA9LT_RIG2_MAP_TOPOYmaps.mat';
% files(n).topoydata = '102621_PVCHA9LT_RIG2_MAP\102621_PVCHA9LT_RIG2_MAP_TOPOY\102621_PVCHA9LT_RIG2_MAP_TOPOY_0001';
% files(n).rignum = 'rig2'; %%% or 'rig1'
% files(n).monitor = 'land'; %%% for topox and y
% files(n).label = 'camk2 gc6';
% files(n).notes = 'good imaging session';

% n=n+1;
% files(n).subj = 'PVCHA9LT';
% files(n).expt = '102621';
% files(n).topox =  '102621_PVCHA9LT_RIG2_MAP\102621_PVCHA9LT_RIG2_MAP_TOPOX\102621_PVCHA9LT_RIG2_MAP_TOPOXmaps.mat';
% files(n).topoxdata = '102621_PVCHA9LT_RIG2_MAP\102621_PVCHA9LT_RIG2_MAP_TOPOX\102621_PVCHA9LT_RIG2_MAP_TOPOX_00001';
% files(n).topoy =  '102621_PVCHA9LT_RIG2_MAP\102621_PVCHA9LT_RIG2_MAP_TOPOY\102621_PVCHA9LT_RIG2_MAP_TOPOYmaps.mat';
% files(n).topoydata = '102621_PVCHA9LT_RIG2_MAP\102621_PVCHA9LT_RIG2_MAP_TOPOY\102621_PVCHA9LT_RIG2_MAP_TOPOY_00001';
% files(n).rignum = 'rig2'; %%% or 'rig1'
% files(n).monitor = 'land'; %%% for topox and y
% files(n).label = 'camk2 gc6';
% files(n).notes = 'good imaging session';
% 

n=n+1;
files(n).subj = 'J581LT';
files(n).expt = '120621';
files(n).topox =  '120621_J581LT_RIG2_MAP\120621_J581LT_RIG2_MAP_TOPOX\120621_J581LT_RIG2_MAP_TOPOXmaps.mat';
files(n).topoxdata = '120621_J581LT_RIG2_MAP\120621_J581LT_RIG2_MAP_TOPOX\120621_J581LT_RIG2_MAP_TOPOX_0001';
files(n).topoy =  '120621_J581LT_RIG2_MAP\120621_J581LT_RIG2_MAP_TOPOY\120621_J581LT_RIG2_MAP_TOPOYmaps.mat';
files(n).topoydata = '120621_J581LT_RIG2_MAP\120621_J581LT_RIG2_MAP_TOPOY\120621_J581LT_RIG2_MAP_TOPOY_0001';
files(n).rignum = 'rig2'; %%% or 'rig1'
files(n).monitor = 'land'; %%% for topox and y
files(n).label = 'camk2 gc6';
files(n).notes = 'good imaging session';

n=n+1;
files(n).subj = 'J581RT';
files(n).expt = '120621';
files(n).topox =  '120621_J581RT_RIG2_MAP\120621_J581RT_RIG2_MAP_TOPOX\120621_J581RT_RIG2_MAP_TOPOXmaps.mat';
files(n).topoxdata = '120621_J581RT_RIG2_MAP\120621_J581RT_RIG2_MAP_TOPOX\120621_J581RT_RIG2_MAP_TOPOX_0001';
files(n).topoy =  '120621_J581RT_RIG2_MAP\120621_J581RT_RIG2_MAP_TOPOY\120621_J581RT_RIG2_MAP_TOPOYmaps.mat';
files(n).topoydata = '120621_J581RT_RIG2_MAP\120621_J581RT_RIG2_MAP_TOPOY\120621_J581RT_RIG2_MAP_TOPOY_0001';
files(n).rignum = 'rig2'; %%% or 'rig1'
files(n).monitor = 'land'; %%% for topox and y
files(n).label = 'camk2 gc6';
files(n).notes = 'good imaging session';


% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% run the batch analysis (keep this at the bottom of the script
batchDfofMovie


