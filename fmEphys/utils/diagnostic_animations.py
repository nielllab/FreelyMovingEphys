"""
Diagnostic animations
"""

def diagnostic_video(self):
    vidread = cv2.VideoCapture(self.video_path)
    width = int(vidread.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vidread.get(cv2.CAP_PROP_FRAME_HEIGHT))

    savepath = os.path.join(self.recording_path, (self.recording_name+'_plot.avi'))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out_vid = cv2.VideoWriter(savepath, fourcc, 60.0, (width, height))
    plot_color0 = (225, 255, 0)

    if self.config['internals']['video_frames_to_save'] > int(vidread.get(cv2.CAP_PROP_FRAME_COUNT)):
        num_save_frames = int(vidread.get(cv2.CAP_PROP_FRAME_COUNT))
    else:
        num_save_frames = self.config['internals']['video_frames_to_save']

    for frame_num in tqdm(range(0,num_save_frames)):
        ret, frame = vidread.read()
        if not ret:
            break
        try:
            for k in range(0, len(self.xrpts['point_loc']), 3):
                try:
                    td_pts_x = self.xrpts.isel(frame=frame_num, point_loc=k).values
                    td_pts_y = self.xrpts.isel(frame=frame_num, point_loc=k+1).values
                    center_xy = (int(td_pts_x), int(td_pts_y))
                    frame = cv2.circle(frame, center_xy, 6, plot_color0, -1)
                except (ValueError, OverflowError) as e:
                    pass
        except KeyError:
            pass
        out_vid.write(frame)
    out_vid.release()

    if self.config['internals']['diagnostic_preprocessing_videos'] and self.make_speed_yaw_video:
        vid_save_path = os.path.join(self.recording_path,(self.recording_name+'_'+self.camname+'_speed_yaw.avi'))
        start = 1000
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out_vid = cv2.VideoWriter(vid_save_path, fourcc, 60.0, (432, 288))
        maxprev = 25
        for f in tqdm(range(start,start+3600)):
            fig = plt.figure()
            plt.imshow(self.xrframes[f,:,:].astype(np.uint8), cmap='gray')
            plt.ylim([135,0]); plt.xlim([0,180])
            plt.axis('off')
            plt.plot(lear_x[f]*0.25, lear_y[f]*0.25, 'b*')
            plt.plot(rear_x[f]*0.25, rear_y[f]*0.25, 'b*')
            plt.plot([neck_x[f]*0.25, (neck_x[f]*0.25)+15*np.cos(head_yaw[f])],
                        [neck_y[f]*0.25,(neck_y[f]*0.25)+15*np.sin(head_yaw[f])],
                        '-', linewidth=2, color='cyan') # head yaw
            plt.plot([back_x[f]*0.25, (back_x[f]*0.25)-15*np.cos(body_yaw[f])],
                        [back_y[f]*0.25, (back_y[f]*0.25)-15*np.sin(body_yaw[f])],
                        '-', linewidth=2, color='pink') # body yaw
            for p in range(maxprev):
                prevf = f - p
                plt.plot(neck_x[prevf]*0.25,
                            neck_y[prevf]*0.25, 'o', color='tab:purple',
                            alpha=(maxprev-p)/maxprev) # neck position history
            # arrow for vector of motion
            if forward_run[f]:
                movvec_color = 'tab:green'
            elif backward_run[f]:
                movvec_color = 'tab:orange'
            elif fine_motion[f]:
                movvec_color = 'tab:olive'
            elif immobility[f]:
                movvec_color = 'tab:red'
            plt.arrow(neck_x[f]*0.25, neck_y[f]*0.25, x_disp[f]*3, y_disp[f]*3, color=movvec_color, width=1)
            # save the frame out
            fig.canvas.draw()
            frame_as_array = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
            frame_as_array = frame_as_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close()
            img = cv2.cvtColor(frame_as_array, cv2.COLOR_RGB2BGR)
            out_vid.write(img.astype('uint8'))
        out_vid.release()
    