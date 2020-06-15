#####################################################################################
"""
corral_files.py of FreelyMovingEphys/file_management/

Renames files so that there is a preceding 1 before single digits so that glob
function can tell apart _1 from _11

last modified: June 15, 2020 by Dylan Martins (dmartins@uoregon.edu)
"""
#####################################################################################

from glob import glob
import pandas as pd
import os.path
import cv2

def rename_files(vid_list, missing_zero_pos, save_path):
    for file in vid_list:
        split_path = os.path.split(file)
        file_name = split_path[1]
        path_mising_pos = missing_zero_pos
        if file_name[path_mising_pos].isalpha():
            output_name1 = file_name[:(path_mising_pos-1)] + '0' + file_name[(path_mising_pos-1):]
            output_name = save_path + '/' + output_name1
        elif file_name[path_mising_pos].isdigit():
            output_name = save_path + file_name
        else:
            output_name1 = file_name[:(path_mising_pos - 1)] + '0' + file_name[(path_mising_pos - 1):]
            output_name = save_path + '/' + output_name1

        print(output_name)

        if 'avi' not in output_name:
            if 'TS' not in output_name:
                open_file = pd.read_hdf(file)
                open_file.to_hdf(output_name, key='data', mode='w')
            elif 'TS' in output_name:
                open_file = pd.read_csv(file)
                open_file.to_csv(output_name)
        elif 'avi' in output_name:
            cap = cv2.VideoCapture(file)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_rate = int(cap.get(cv2.CAP_PROP_XI_FRAMERATE))
            if width > 0 and height > 0:
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                out = cv2.VideoWriter(output_name, fourcc, frame_rate, (width, height))

                while(cap.isOpened()):
                    ret, frame = cap.read()
                    out.write(frame)
                    cv2.imshow('frame', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                cap.release()
                out.release()
                cv2.destroyAllWindows()

vid_path = '/Users/dylanmartins/data/Niell/PreyCapture/Cohort3/J463c(blue)/*/ApproachVids/'
save_path = '/Users/dylanmartins/data/Niell/PreyCapture/Cohort3/J463c(blue)/110719/CorralledApproachVids/'

righteye_vid_list = set(glob(vid_path + '*eye1r*.avi')) - set(glob(vid_path + '*DeInter*.avi'))
lefteye_vid_list = set(glob(vid_path + '*eye2l*.avi')) - set(glob(vid_path + '*DeInter*.avi'))
topdown_vid_list = set(glob(vid_path + '*top*.avi')) - set(glob(vid_path + '*DeInter*.avi'))
worldcam_vid_list = set(glob(vid_path + '*world*.avi')) - set(glob(vid_path + '*DeInter*.avi'))

# topdown=27, right/lefteye=29 right/lefttime=31 # topdowntime=29

rename_files(righteye_vid_list, 29, save_path)
rename_files(lefteye_vid_list, 29, save_path)
rename_files(topdown_vid_list, 27, save_path)