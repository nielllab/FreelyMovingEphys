#make_worldcam_stim_vid.py

import os
import numpy as np
import cv2 as cv
import io_dict_to_hdf5 as ioh5

model_data_file = r'\\goeppert\nlab-nas\freely_moving_ephys\ephys_recordings\070921\J553RT\fm1\070921_J553RT_ModelData_dt025_rawWorldCam_2ds.h5'
data = ioh5.load(model_data_file)
wc_data = data['model_vis_sm_shift']

st=20000 #start frame
tot_frames = 1*60*40 # num frames needed for 10 min movie at 40 fps
good_idxs = np.ones(len(data['model_active']),dtype=bool) # frames when mouse is active
good_idxs[data['model_active']<.5] = False
good_idxs.astype('int')

out_name = r'C:\Users\nlab\Desktop\worldStimTest.avi'
fps=40
frame_width=wc_data.shape[2]
frame_height=wc_data.shape[1]
out = cv.VideoWriter(out_name, cv.VideoWriter_fourcc('M','J','P','G'), fps,(frame_width,frame_height))

#convert data into uint8
wc_min = np.min(np.min(np.min(wc_data)))
wc_data_norm = (wc_data-wc_min)
wc_max = np.max(np.max(np.max(wc_data_norm)))
wc_data_norm /= wc_max
wc_data_norm *= 255
wc_data_norm = wc_data_norm.astype('uint8')

for i in np.arange(st,st+tot_frames):
    frame = wc_data_norm[i,:,:]
    out.write(frame)
    print('wrote frame %d' % i)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
out.release()