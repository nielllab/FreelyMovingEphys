"""
analyze_new_vids.py

Analyze new videos using DeepLabCut and/or Anipose.

Last modified August 27, 2020
"""

# package imports
import deeplabcut
import numpy as np
from aniposelib.boards import CharucoBoard, Checkerboard
from aniposelib.cameras import Camera, CameraGroup
from aniposelib.utils import load_pose2d_fnames
from aniposelib
import argparse
from glob import glob
import shutil

# module imports
from util.read_data import find

def analyze2D(vid_list, data_path, save_path, config_path):
    for vid in vid_list:
        vid_source_path = os.path.split(vid)[0]
        vid_save_path = current_path.replace(data_path, save_path)
        print('analyzing ' + vid)
        deeplabcut.analyze_videos(config_path, vid, destfolder=vid_save_path)

parser = argparse.ArgumentParser(description='analyze new videos using DeepLabCut and Anipose using an already-trained network')
parser.add_argument('-type', '--cam_type', help='camera_type must be eye, top3D, top2D, or side')
parser.add_argument('-data', '--data_directory', help='parent directory for .avi videos; should be directory with config.toml in it if cam_type=top3D')
parser.add_argument('-save', '--save_directory', help='path into which outputs will be saved')
parser.add_argument('-cnfgDLC', '--deeplabcut_config_path', help='path to the DLC network config file')
args = parser.parse_args()

if args.cam_type == 'eye':
    eye_list = find('*EYE*.avi', args.data_directory)
    analyze2D(eye_list, args.data_directory, args.save_directory, args.deeplabcut_config_path)
    print('done analyzing ' + str(len(eye_list)) ' eye videos')

elif args.cam_type == 'top2D':
    top2D_list = find('*TOP*.avi', args.data_directory)
    analyze2D(eye_list, args.data_directory, args.save_directory, args.deeplabcut_config_path)
    print('done analyzing ' + str(len(top2D_list)) ' top videos')

elif args.cam_type == 'side':
    side_list = find('*SIDE*.avi', args.data_directory)
    analyze2D(side_list, args.data_directory, args.save_directory, args.deeplabcut_config_path)
    print('done analyzing ' + str(len(side_list)) ' side videos')

elif args.cam_type == 'top3D':
    # first, analyze the videos
    print('reorganizing files for anipose project...')
    top3d_allAVIfiles = find('*TOP*.avi', args.data_directory)
    top3d_allCSVfiles = find('*TOP*.csv', args.data_directory)
    top3d_allfiles_list = top3d_CSVfiles + top3d_allAVIfiles
    for file in top3d_allfiles_list:
        date = file.split('/')[-3]; mouse = file.split('/')[-2]; trial = file.split('/')[-1]
        file_newhead = current_path.replace(args.data_directory, args.save_directory)
        file_newtail = os.path.join(date, mouse, trial, 'videos-raw')
        file_newpath = os.path.join(file_newpath, file_newtail)
        shutil.copyfile(file, file_newpath)

    print('files have been moved to ' + args.save_directory)
    print('user should put the Anipose config.toml inside the new directory')
    input('press Enter when this is done: ')

    print('analyzing new videos')
    aniposelib.analyze(file_newpath)
    print('labeling new videos')
    aniposelib.label(file_newpath)
    print('done analyzing ' + str(len(top3d_allAVIfiles)) ' top videos')


    # for vid in vid_list:
    #     vid_source_path = os.path.split(vid)[0]
    #     vid_save_path = current_path.replace(args.data_directory, os.path.join(args.save_directory, 'anipose2d'))
    #     print('analyzing ' + vid)
    #     aniposelib.analyze(args.data_directory)

    # cam_names = ['TOP1', 'TOP2', 'TOP3']
    # # anipose calibration
    # if args.calib_vids_directory:
    #     vid1 = find('*TOP1*.avi', args.calib_vids_directory)
    #     vid2 = find('*TOP2*.avi', args.calib_vids_directory)
    #     vid3 = find('*TOP3*.avi', args.calib_vids_directory)
    #     vidnames = [vid1, vid2, vid3]
    #     n_cams = len(vidnames)
    #     board = CharucoBoard(7, 10,
    #                  square_length=25,
    #                  marker_length=18.75,
    #                  marker_bits=4, dict_size=50)
    #     # treat the cameras as a group
    #     cgroup = CameraGroup.from_names(cam_names)
    #     # calibrate videos with board
    #     print('calibrating from videos... this will take 15 minutes')
    #     cgroup.calibrate_videos(vidnames, board)
    # elif args.calib_load_directory:
    #     print('loading previously saved calibration')
    #     cgroup = CameraGroup.load(calib_load_directory)
    #
    # # anipose triangulation
    # top1_3D_list = find('*TOP1*.h5', os.path.join(args.save_directory, 'deeplabcut'))
    # top2_3D_list = find('*TOP2*.h5', os.path.join(args.save_directory, 'deeplabcut'))
    # top3_3D_list = find('*TOP3*.h5', os.path.join(args.save_directory, 'deeplabcut'))
    #
    # for trialname1 in top1_d3_list:
    #     trialname = trialname1.split('_')[:-1]
    #     print('working on ' + trial)
    #     top1vidpath = [i for i in top1_3D_list if trialname in i]
    #     top2vidpath = [i for i in top2_3D_list if trialname in i]
    #     top3vidpath = [i for i in top3_3D_list if trialname in i]
    #
    #     fname_dict = {'TOP1': top1vidpath[0], 'TOP2': top2vidpath[0], 'TOP3': top3vidpath[0]}
    #
    #     d = load_pose2d_fnames(fname_dict, cam_names=cgroup.get_names())
    #     score_threshold = 0.5
    #
    #     n_cams, n_points, n_joints, _ = d['points'].shape
    #     points = d['points']
    #     scores = d['scores']
    #     bodyparts = d['bodyparts']
    #     points[scores < score_threshold] = np.nan
    #
    #     points_flat = points.reshape(n_cams, -1, 2)
    #     scores_flat = scores.reshape(n_cams, -1)
    #
    #     p3ds_flat = cgroup.triangulate(points_flat, progress=True)
    #     reprojerr_flat = cgroup.reprojection_error(p3ds_flat, points_flat, mean=True)
    #
    #     p3ds = p3ds_flat.reshape(n_points, n_joints, 3)
    #     reprojerr = reprojerr_flat.reshape(n_points, n_joints)
