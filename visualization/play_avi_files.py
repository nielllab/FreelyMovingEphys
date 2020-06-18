# quick script to play videos saved out of load_from_DLC.py by functions in check_tracking.py

import numpy as np
import cv2

# sets the next frame to be read
def getFrame(frame_nr):
    global cap
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_nr)

cap = cv2.VideoCapture('/Users/dylanmartins/data/Niell/PreyCapture/Cohort3Outputs/J463c(blue)_110719/analysis_test_03/mouse_J463c_trial_1_110719_07.avi')
# cap = cv2.VideoCapture('/Users/dylanmartins/data/Niell/PreyCapture/Cohort3Outputs/J463c(blue)_110719/analysis_test_02/_mouse_J463c_trial_1_110719_11.avi')
trial_name = 'mouse_J463c_trial_1_110719_07'
cv2.namedWindow(trial_name)
cv2.createTrackbar("frame_position", trial_name, 0, int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), getFrame)

while 1:

    ret, frame = cap.read()
    if ret:
        cv2.imshow(trial_name, frame)
        cv2.setTrackbarPos("frame_position", trial_name, int(cap.get(cv2.CAP_PROP_POS_FRAMES)))
    else:
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()