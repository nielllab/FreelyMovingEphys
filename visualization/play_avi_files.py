# quick script to play videos saved out of load_from_DLC.py by functions in check_all_tracking.py

import numpy as np
import cv2

cap = cv2.VideoCapture('/home/dylan/data/Niell/PreyCapture/Cohort3Outputs/J463c(blue)_110719/analysis_test_U13d/mouse_J463c_trial_1_110719_03/mouse_J463c_trial_1_110719_03.mp4')
trial_name = 'mouse_J463c_trial_1_110719_03'

while 1:

    ret, frame = cap.read()
    cv2.imshow(trial_name, frame)
    cv2.waitKey(60)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()