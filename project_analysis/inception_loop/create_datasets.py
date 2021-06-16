"""
create_datasets.py
"""
import glob
import os
import pandas as pd
import argparse
import numpy as np
from functools import partial
import subprocess

# get user arguments
parser = argparse.ArgumentParser(description='Create Dataset for WC Data')
parser.add_argument('--rootdir',  '-r', help =  'RootDir',
                    default='/home/niell_lab/data/freely_moving_ephys/inception_loop/inputs/')

args = parser.parse_args()
rootdir = os.path.expanduser(args.rootdir)

# set up partial functions for directory managing
join = partial(os.path.join,rootdir)

def check_path(basepath, path):
    """
    checks if path exists, if not then creates directory
    """
    if path in basepath:
        return basepath
    elif not os.path.exists(os.path.join(basepath, path)):
        os.makedirs(os.path.join(basepath, path))
        print('Added Directory:'+ os.path.join(basepath, path))
        return os.path.join(basepath, path)
    else:
        return os.path.join(basepath, path)

def extract_frames_from_csv(csv_path):
    AllExps = pd.read_csv(csv_path)

    GoodExps = AllExps[(AllExps['Experiment outcome']=='good')].copy().reset_index()
    GoodExps = pd.concat((GoodExps[(GoodExps['Computer']=='Kraken')][['Experiment date','Animal name','Computer','Drive']],
                        GoodExps[(GoodExps['Computer']=='Niell-V2-W7')][['Experiment date','Animal name','Computer','Drive']]))
    GoodExps['Experiment date'] = pd.to_datetime(GoodExps['Experiment date'],infer_datetime_format=True,format='%m%d%Y').dt.strftime('%2m%2d%2y')
    GoodExps['Computer']=GoodExps['Computer'].str.capitalize()

    for n in range(len(GoodExps)):
        Comp=GoodExps['Computer'][n]
        Drive=GoodExps['Drive'][n]
        Date=GoodExps['Experiment date'][n]
        Ani=GoodExps['Animal name'][n]
        WorldPath = os.path.join(os.path.expanduser('~/'),Comp,Drive,'freely_moving_ephys/ephys_recordings',Date,Ani,'fm1','*WORLDcalib.avi')

        FM1Cam = glob.glob(WorldPath)
        if len(FM1Cam) > 0:
            SavePath = os.path.join(check_path(rootdir,os.path.basename(FM1Cam[0])[:-9]),'frame_%06d.png')
            subprocess.call(['ffmpeg', '-i', FM1Cam[0], '-vf','fps=60', '-vf','scale=128:128', SavePath])

def create_train_val_csv(TrainSet,ValSet):
    ExpDir = []
    DNum = []
    for exp in TrainSet:
        DataPaths = sorted(glob.glob(join(exp,'*.png')))
        print('{}: '.format(exp),len(DataPaths))
        for n in range(len(DataPaths)):
            ExpDir.append(DataPaths[n].split('/')[-2])
            DNum.append(DataPaths[n].split('/')[-1])

    df_train = pd.DataFrame({'BasePath':ExpDir,'FileName':DNum})
    df_train.to_csv(join('WC_Train_Data.csv'))

    print('Total Training Size: ', len(df_train))


    ExpDir = []
    DNum = []
    for exp in ValSet:
        DataPaths = sorted(glob.glob(join(exp,'*.png')))
        print('{}: '.format(exp),len(DataPaths))
        for n in range(len(DataPaths)):
            ExpDir.append(DataPaths[n].split('/')[-2])
            DNum.append(DataPaths[n].split('/')[-1])

    df_val = pd.DataFrame({'BasePath':ExpDir,'FileName':DNum})
    df_val.to_csv(join('WC_Val_Data.csv'))

    print('Total Validation Size: ', len(df_val))
    return df_train, df_val

def create_train_val_csv_3d(TrainSet,ValSet,save_dir,N_fm=16):
    ExpDir = []
    DNum = []
    for exp in TrainSet:
        DataPaths = sorted(glob.glob(join(exp,'*.png')))
        print('{}: '.format(exp),len(DataPaths))
        for n in range(len(DataPaths)):
            if n < N_fm: 
                DNum_temp = [DataPaths[0].split('/')[-1] for t in range(N_fm-n)]
                DNum_temp = sorted(DNum_temp + [DataPaths[n-t].split('/')[-1] for t in range(N_fm - len(DNum_temp))])
                DNum.append(DNum_temp)
            else:
                DNum.append([DataPaths[n+t-N_fm+1].split('/')[-1] for t in range(N_fm)])
            ExpDir.append(DataPaths[n].split('/')[-2])
        df_val = pd.DataFrame(DNum)
        colNames =  ['N_{:02d}'.format(n) for n in range(N_fm)]
        df_train.columns = colNames
        df_train.insert(0, 'BasePath', ExpDir)
        df_train.to_csv(os.path.join(save_dir,'WC3d_Train_Data.csv'))

    print('Total Training Size: ', len(df_train))

    ExpDir = []
    DNum = []
    for exp in ValSet:
        DataPaths = sorted(glob.glob(join(exp,'*.png')))
        print('{}: '.format(exp),len(DataPaths))
        for n in range(len(DataPaths)):
            if n < N_fm: 
                DNum_temp = [DataPaths[0].split('/')[-1] for t in range(N_fm-n)]
                DNum_temp = sorted(DNum_temp + [DataPaths[n-t].split('/')[-1] for t in range(N_fm - len(DNum_temp))])
                DNum.append(DNum_temp)
            else:
                DNum.append([DataPaths[n+t-N_fm+1].split('/')[-1] for t in range(N_fm)])
            ExpDir.append(DataPaths[n].split('/')[-2])

        df_val = pd.DataFrame(DNum)
        colNames =  ['N_{:02d}'.format(n) for n in range(N_fm)]
        df_val.columns = colNames
        df_val.insert(0, 'BasePath', ExpDir)
        df_val.to_csv(os.path.join(save_dir,'WC3d_Val_Data.csv'))

    print('Total Validation Size: ', len(df_val))
    return df_train, df_val

if __name__ == '__main__':
    
    csv_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),'/metadata/exp_pool.csv')

    extract_frames_from_csv(csv_path)
    TrainSet = sorted([os.path.basename(x) for x in glob.glob(join('*WORLD'))])
    valnum = np.random.randint(len(TrainSet))
    ValSet = [TrainSet[valnum]]
    TrainSet.pop(valnum)
    
    df_train, df_val = create_train_val_csv_3d(TrainSet,ValSet,rootdir)