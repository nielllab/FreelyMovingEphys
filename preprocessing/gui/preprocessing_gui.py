"""
gui_launch.py

launch a GUI which writes a .json to file and then starts the preprocessing pipeline onusing that config file

Jan. 16, 2021
"""
# package imports
import pandas as pd
import numpy as np
from tkinter import filedialog
from tkinter import *
from tkinter import ttk
import json, os
from tkinter import scrolledtext
import tkinter as tk
# module imports
from preprocessing.gui.auto_config import write_config
from preprocessing.gui.auto_preprocessing import run_auto_preprocessing

def launch_gui():
    # get the path of the default json config file in this repository, relative to util/config.py
    try:
        default_json_path = '/'.join(os.path.abspath(__file__).split('\\')[:-3]) + '/example_configs/preprocessing_config.json'
        # read in the json
        with open(default_json_path, 'r') as fp:
            default_config = json.load(fp)
    except FileNotFoundError:
        default_json_path = '/'.join(os.path.abspath(__file__).split('/')[:-3]) + '/example_configs/preprocessing_config.json'
        with open(default_json_path, 'r') as fp:
            default_config = json.load(fp)
    
    # set up GUI window
    window = Tk()
    window.title('FreelyMovingEphys - Preprocessing')

    tab_control = ttk.Notebook(window)

    welcome = ttk.Frame(tab_control)
    data_sel = ttk.Frame(tab_control)
    deinter = ttk.Frame(tab_control)
    calib = ttk.Frame(tab_control)
    dlc = ttk.Frame(tab_control)
    params = ttk.Frame(tab_control)
    addtl_params = ttk.Frame(tab_control)
    config_tab = ttk.Frame(tab_control)
    run = ttk.Frame(tab_control)

    tab_control.add(welcome, text='Welcome')
    tab_control.add(data_sel, text='Select Animal Path')
    tab_control.add(deinter, text='Deinterlace')
    tab_control.add(calib, text='Calibrate')
    tab_control.add(dlc, text='Pose Estimation')
    tab_control.add(params, text='Parameters')
    tab_control.add(addtl_params, text='Extra Parameters')
    tab_control.add(config_tab, text='Write Config')
    tab_control.add(run, text='Run!')

    tab_control.pack(expand=1, fill='both')

    ### welcome tab
    welcome_title = Label(welcome, text='FreelyMovingEphys Preprocessing Pipeline').grid(column=0, row=0)
    welcome_text = Label(welcome, text='Provide information in the tabs of this window to build a .json config file and execute it. If a config file already exists, you can load a .json with the browse button below, and skip to the Run! tab to execute the preprocessing pipeline using those config options. Do not use the browse button below unless the .json already exists. Presently, if you need to modify an existing .json, it is best to either edit it manually by opening the file, or simply overwrite it with the needed parameters.', wraplength=500).grid(column=0, row=1)

    json_path_label = Label(welcome, text='Choose an existing .json config file:')
    json_path_label.grid(column=0, row=2)
    json_path = None
    def clicked_json_button():
        global json_path
        json_path = filedialog.askopenfilename(title='Choose an existing .json config file')
        print('preloaded .json selected as ' + json_path)
        tab_control.tab(data_sel, state='disabled')
        tab_control.tab(deinter, state='disabled')
        tab_control.tab(calib, state='disabled')
        tab_control.tab(dlc, state='disabled')
        tab_control.tab(params, state='disabled')
        tab_control.tab(addtl_params, state='disabled')
        tab_control.tab(config_tab, state='disabled')
    json_path_button = Button(welcome, text="browse", command=clicked_json_button)
    json_path_button.grid(column=1, row=2)

    ### data selection tab
    data_path_label = Label(data_sel, text='Select an animal directory')
    data_path_label.grid(column=0, row=0)
    def clicked_data_button():
        global data_path
        data_path = filedialog.askdirectory(title='Select an animal directory:')
        print('animal directory set to ' + data_path)
    data_path_button = Button(data_sel, text="browse", command=clicked_data_button)
    data_path_button.grid(column=1, row=0)

    ### deinterlace tab
    deinter_label = Label(deinter, text="World and eye cameras should have interlacing removed so that they are put into the later steps of the pipeline at 60fps and not 30fps. During this video preprocessing step, eye and world videos can also be flipped verticlly. Videos should be right-side-up for pose estimation with DeepLabCut, so make sure to check the box to rotate the video if it's needed.", wraplength=500)
    deinter_label.grid(column=0, row=0)

    deinter_eye = None
    deinter_world = None

    def add_deinter_flip_options():
        if run_deinter.get() is True:
            global deinter_eye
            global deinter_world
            # options to flip world and eye cameras
            deinter_eye_label = Label(deinter, text="Flip eye camera vertically?")
            deinter_eye_label.grid(column=0, row=2)
            deinter_eye = BooleanVar()
            deinter_eye1 = Checkbutton(deinter, variable=deinter_eye)
            deinter_eye1.grid(column=1, row=2)

            deinter_world_label = Label(deinter, text="Flip world camera vertically?")
            deinter_world_label.grid(column=0, row=3)
            deinter_world = BooleanVar()
            deinter_world1 = Checkbutton(deinter, variable=deinter_world)
            deinter_world1.grid(column=1, row=3)

    # run deinterlacing?
    run_deinter_label = Label(deinter, text="Deinterlace eye and world cameras?")
    run_deinter_label.grid(column=0, row=1)
    run_deinter = BooleanVar()
    run_deinter1 = Checkbutton(deinter, variable=run_deinter, command=add_deinter_flip_options)
    run_deinter1.grid(column=1, row=1)

    ### calibration tab
    calib_label = Label(calib, text="Cameras, especially the world camera, collect images with distortions which can be removed using paramters calculated from videos of checkboards. You don't have to get the calibration paramters every time--these can be reused between recordings.", wraplength=500)
    calib_label.grid(column=0, row=0)

    top_checker = None; world_checker = None; npz_save_dir = None; top_npz_name = None; world_npz_name = None

    def add_checkerboard_path_options():
        if checker.get() is True:

            top_checker_path_label = Label(calib, text='Path to top camera checkerboard video:')
            top_checker_path_label.grid(column=0, row=2)
            def clicked_top_checker_button():
                global top_checker
                top_checker = filedialog.askopenfilename(title='Select the top camera checkerboard video')
                print('top camera checkerboard video set to ' + top_checker)
            top_checker_button = Button(calib, text="browse", command=clicked_top_checker_button)
            top_checker_button.grid(column=1, row=2)

            world_checker_path_label = Label(calib, text='Path to world camera checkerboard video:')
            world_checker_path_label.grid(column=0, row=3)
            def clicked_world_checker_button():
                global world_checker
                world_checker = filedialog.askopenfilename(title='Select the world camera checkerboard video')
                print('world camera checkerboard video set to ' + world_checker)
            world_checker_button = Button(calib, text="browse", command=clicked_world_checker_button)
            world_checker_button.grid(column=1, row=3)

            npz_save_dir_label = Label(calib, text='Parameter .npz save path:')
            npz_save_dir_label.grid(column=0, row=4)
            def clicked_npz_save_dir_button():
                global npz_save_dir
                npz_save_dir = filedialog.askdirectory(title='Select a directory into which the paramters can be saved:')
                print('calibration parameter directory set to ' + npz_save_dir)
            npz_save_dir_button = Button(calib, text="browse", command=clicked_npz_save_dir_button)
            npz_save_dir_button.grid(column=1, row=4)

            top_npz_name_label = Label(calib, text="File name to use for top calibration npz")
            top_npz_name_label.grid(column=0, row=5)
            top_npz_name = Entry(calib, width=35)
            top_npz_name.insert(END, 'top1_checkerboard_calib.npz')
            top_npz_name.grid(column=1, row=5)

            world_npz_name_label = Label(calib, text="File name to use for world calibration npz")
            world_npz_name_label.grid(column=0, row=6)
            world_npz_name = Entry(calib, width=35)
            world_npz_name.insert(END, 'world_checkerboard_calib.npz')
            world_npz_name.grid(column=1, row=6)

            default_calib_label = Label(calib, text='Print default options in terminal:')
            default_calib_label.grid(column=0, row=7)
            def clicked_default_calib_button():
                default_calib_dict = default_config['calibration']
                print('default top checkerboard .avi: ' + default_calib_dict['top_checker_vid'])
                print('default world checkerboard .avi: ' + default_calib_dict['world_checker_vid'])
                print('default top undistortion paramter save .npz: ' + default_calib_dict['top_checker_npz'])
                print('default world undistortion paramter save .npz: ' + default_calib_dict['world_checker_npz'])
            default_calib_button = Button(calib, text="print defaults", command=clicked_default_calib_button)
            default_calib_button.grid(column=1, row=7)

            current_calib_label = Label(calib, text='Print current options in terminal:')
            current_calib_label.grid(column=0, row=8)
            def clicked_current_calib_button():
                print('current top checkerboard .avi: ' + str(top_checker))
                print('current world checkerboard .avi: ' + str(world_checker))
                if npz_save_dir is not None and top_npz_name is not None:
                    undistort_top_npz_print = os.path.join(npz_save_dir,top_npz_name)
                else:
                    undistort_top_npz_print = str(None)
                if npz_save_dir is not None and world_npz_name is not None:
                    undistort_world_npz_print = os.path.join(npz_save_dir,world_npz_name)
                else:
                    undistort_world_npz_print = str(None)
                print('current top undistortion paramter save .npz: ' + undistort_top_npz_print)
                print('current world undistortion paramter save .npz: ' + undistort_world_npz_print)
            current_calib_button = Button(calib, text="print current", command=clicked_current_calib_button)
            current_calib_button.grid(column=1, row=8)

    checker_label = Label(calib, text="Get parameters from checkerboard videos?")
    checker_label.grid(column=0, row=1)
    checker = BooleanVar()
    checker1 = Checkbutton(calib, variable=checker, command=add_checkerboard_path_options)
    checker1.grid(column=1, row=1)

    top_npz_read_dir = None; world_npz_read_dir = None

    def add_undistortion_filepath_options():
        if undistort.get() is True:
            global top_npz_read_dir
            global world_npz_read_dir
            top_npz_read_dir_label = Label(calib, text='Path to top camera calibration .npz')
            top_npz_read_dir_label.grid(column=0, row=11)
            def clicked_top_npz_read_dir_button():
                global top_npz_read_dir
                top_npz_read_dir = filedialog.askopenfilename(title='Select a directory from which to read the top calibration .npz:')
                print('TOP calibration parameter read directory set to ' + top_npz_read_dir)
            top_npz_read_dir_button = Button(calib, text="browse", command=clicked_top_npz_read_dir_button)
            top_npz_read_dir_button.grid(column=1, row=11)

            world_npz_read_dir_label = Label(calib, text='Path to world camera calibration .npz')
            world_npz_read_dir_label.grid(column=0, row=12)
            def clicked_world_npz_read_dir_button():
                global world_npz_read_dir
                world_npz_read_dir = filedialog.askopenfilename(title='Select a directory from which to read the world calibration .npz:')
                print('world calibration parameter read directory set to ' + world_npz_read_dir)
            world_npz_read_dir_button = Button(calib, text="browse", command=clicked_world_npz_read_dir_button)
            world_npz_read_dir_button.grid(column=1, row=12)

            default_undistort_label = Label(calib, text='Print default options in terminal:')
            default_undistort_label.grid(column=0, row=13)
            def clicked_default_undistort_button():
                default_undistort_dict = default_config['calibration']
                print('default top undistortion paramter save .npz: ' + default_undistort_dict['top_checker_npz'])
                print('default world undistortion paramter save .npz: ' + default_undistort_dict['world_checker_npz'])
            default_undistort_button = Button(calib, text="print defaults", command=clicked_default_undistort_button)
            default_undistort_button.grid(column=1, row=13)

            current_undistort_label = Label(calib, text='Print current options in terminal:')
            current_undistort_label.grid(column=0, row=14)
            def clicked_current_undistort_button():
                print('current top undistortion paramter save .npz: ' + top_npz_read_dir)
                print('current world undistortion paramter save .npz: ' + world_npz_read_dir)
            current_undistort_button = Button(calib, text="print current", command=clicked_current_undistort_button)
            current_undistort_button.grid(column=1, row=14)

    undistort_label = Label(calib, text="Undistort top and world videos using existing calibration parameters?")
    undistort_label.grid(column=0, row=9)
    undistort = BooleanVar()
    undistort1 = Checkbutton(calib, variable=undistort, command=add_undistortion_filepath_options)
    undistort1.grid(column=1, row=9)
    undistort_info_label = Label(calib, text="If the .npz files already exist, select the above checkbox and browse for the existing calibration .npz files. If you are creating the .npz files from checkerboard videos during this session, the save paths that you entered can be used as the read paths. In this case, check the box below to use paramters saved out during this session and leave the above option to read them in unchecked.", wraplength=500)
    undistort_info_label.grid(column=0, row=10)
    undistortCS_label = Label(calib, text="Or: use parameters saved out during this session?")
    undistortCS_label.grid(column=0, row=15)
    undistort_CS = BooleanVar()
    undistortCS1 = Checkbutton(calib, variable=undistort_CS)
    undistortCS1.grid(column=1, row=15)

    ### pose estimation tab
    dlc_label = Label(dlc, text="DeepLabCut generates pose estiamtions for top, side, and/or eye cameras. You need to select each of the cameras you want to include in the pose estimation, and then browse for a DeepLabCut .yaml project config file. DeepLabCut can be set up to track IR LEDs in the extra paramters tab, seperate from the mouse tracking DLC which happens here. If no cameras are entered, the pose estimation step will be skipped.", wraplength=500)
    dlc_label.grid(column=0, row=0)

    camera_options = ['None','TOP','Top','TOP1','TOP2','TOP3','REYE','LEYE','WORLD','World','SIDE','Side']

    dlc_c1_label = Label(dlc, text="First Camera:")
    dlc_c1_label.grid(column=0,row=1)
    dlc.c1 = StringVar()
    dlc.c1.set(camera_options[0])
    dlc_c1 = OptionMenu(dlc, dlc.c1, *camera_options)
    dlc_c1.grid(column=1, row=1)
    dlc_c1_label = Label(dlc, text="DLC .yaml path:")
    dlc_c1_label.grid(column=2,row=1)
    def clicked_c1_button():
        global c1_yaml_path
        c1_yaml_path = filedialog.askopenfilename(title='Select the yaml path for cam of type '+str(dlc.c1.get())+':')
        print('Camera 1 .yaml path set to: ' + c1_yaml_path)
    dlc_c1_yaml_button = Button(dlc, text="browse", command=clicked_c1_button)
    dlc_c1_yaml_button.grid(column=3, row=1)

    dlc_c2_label = Label(dlc, text="Second Camera:")
    dlc_c2_label.grid(column=0,row=2)
    dlc.c2 = StringVar()
    dlc.c2.set(camera_options[0])
    dlc_c2 = OptionMenu(dlc, dlc.c2, *camera_options)
    dlc_c2.grid(column=1, row=2)
    dlc_c2_label = Label(dlc, text="DLC .yaml path:")
    dlc_c2_label.grid(column=2,row=2)
    def clicked_c2_button():
        global c2_yaml_path
        c2_yaml_path = filedialog.askopenfilename(title='Select the yaml path for cam of type '+str(dlc.c2.get())+':')
        print('Camera 2 .yaml path set to: ' + c2_yaml_path)
    dlc_c2_yaml_button = Button(dlc, text="browse", command=clicked_c2_button)
    dlc_c2_yaml_button.grid(column=3, row=2)

    dlc_c3_label = Label(dlc, text="Third Camera:")
    dlc_c3_label.grid(column=0,row=3)
    dlc.c3 = StringVar()
    dlc.c3.set(camera_options[0])
    dlc_c3 = OptionMenu(dlc, dlc.c3, *camera_options)
    dlc_c3.grid(column=1, row=3)
    dlc_c3_label = Label(dlc, text="DLC .yaml path:")
    dlc_c3_label.grid(column=2,row=3)
    def clicked_c3_button():
        global c3_yaml_path
        c3_yaml_path = filedialog.askopenfilename(title='Select the yaml path for cam of type '+str(dlc.c3.get())+':')
        print('Camera 3 .yaml path set to: ' + c3_yaml_path)
    dlc_c3_yaml_button = Button(dlc, text="browse", command=clicked_c3_button)
    dlc_c3_yaml_button.grid(column=3, row=3)

    dlc_c4_label = Label(dlc, text="Fourth Camera:")
    dlc_c4_label.grid(column=0,row=4)
    dlc.c4 = StringVar()
    dlc.c4.set(camera_options[0])
    dlc_c4 = OptionMenu(dlc, dlc.c4, *camera_options)
    dlc_c4.grid(column=1, row=4)
    dlc_c4_label = Label(dlc, text="DLC .yaml path:")
    dlc_c4_label.grid(column=2,row=4)
    def clicked_c4_button():
        global c4_yaml_path
        c4_yaml_path = filedialog.askopenfilename(title='Select the yaml path for cam of type '+str(dlc.c4.get())+':')
        print('Camera 4 .yaml path set to: ' + c4_yaml_path)
    dlc_c4_yaml_button = Button(dlc, text="browse", command=clicked_c4_button)
    dlc_c4_yaml_button.grid(column=3, row=4)

    dlc_c5_label = Label(dlc, text="Fifth Camera:")
    dlc_c5_label.grid(column=0,row=5)
    dlc.c5 = StringVar()
    dlc.c5.set(camera_options[0])
    dlc_c5 = OptionMenu(dlc, dlc.c5, *camera_options)
    dlc_c5.grid(column=1, row=5)
    dlc_c5_label = Label(dlc, text="DLC .yaml path:")
    dlc_c5_label.grid(column=2,row=5)
    def clicked_c5_button():
        global c5_yaml_path
        c5_yaml_path = filedialog.askopenfilename(title='Select the yaml path for cam of type '+str(dlc.c5.get())+':')
        print('Camera 5 .yaml path set to: ' + c5_yaml_path)
    dlc_c5_yaml_button = Button(dlc, text="browse", command=clicked_c5_button)
    dlc_c5_yaml_button.grid(column=3, row=5)

    dlc_c6_label = Label(dlc, text="Sixth Camera:")
    dlc_c6_label.grid(column=0,row=6)
    dlc.c6 = StringVar()
    dlc.c6.set(camera_options[0])
    dlc_c6 = OptionMenu(dlc, dlc.c6, *camera_options)
    dlc_c6.grid(column=1, row=6)
    dlc_c6_label = Label(dlc, text="DLC .yaml path:")
    dlc_c6_label.grid(column=2,row=6)
    def clicked_c6_button():
        global c6_yaml_path
        c6_yaml_path = filedialog.askopenfilename(title='Select the yaml path for cam of type '+str(dlc.c6.get())+':')
        print('Camera 6 .yaml path set to: ' + c6_yaml_path)
    dlc_c6_yaml_button = Button(dlc, text="browse", command=clicked_c6_button)
    dlc_c6_yaml_button.grid(column=3, row=6)

    dlc_option_label = Label(dlc, text='Pose esimation options:')
    dlc_option_label.grid(column=0, row=7)

    crop_vids_label = Label(dlc, text="Crop videos prior to pose estimation?")
    crop_vids_label.grid(column=0, row=8)
    crop_vids = BooleanVar()
    crop_vids1 = Checkbutton(dlc, variable=crop_vids)
    crop_vids1.grid(column=1, row=8)

    multiTOP_label = Label(dlc, text="Read top camera as multianimal project?")
    multiTOP_label.grid(column=0, row=9)
    multiTOP = BooleanVar()
    multiTOP1 = Checkbutton(dlc, variable=multiTOP)
    multiTOP1.grid(column=1, row=9)

    ### parameters tab
    params_label = Label(params, text='Options for getting paramters out of pose estimation datasets are organized by subject. Ignore anything that does not apply to your dataset, and the pipeline will ignore it too.', wraplength=500)
    params_label.grid(column=0, row=1)

    run_params_labels = Label(params, text="Get parameters?")
    run_params_labels.grid(column=1, row=1)
    run_params = BooleanVar()
    run_params1 = Checkbutton(params, variable=run_params)
    run_params1.grid(column=2, row=1)

    lik_thresh_label = Label(params, text="Likelihood threhold:")
    lik_thresh_label.grid(column=0, row=2)
    lik_thresh = Entry(params, width=10)
    lik_thresh.insert(END, 0.99)
    lik_thresh.grid(column=1, row=2)

    eye_label = Label(params, text='Eye options:', wraplength=500)
    eye_label.grid(column=0, row=3)

    tear_label = Label(params, text="Is the tear duct labeled in this eye network?")
    tear_label.grid(column=0, row=4)
    tear = BooleanVar()
    tear1 = Checkbutton(params, variable=tear)
    tear1.grid(column=1, row=4)

    pxl_thresh_label = Label(params, text="Maximum acceptable number of pixels for radius of the pupil:")
    pxl_thresh_label.grid(column=0, row=5)
    pxl_thresh = Entry(params, width=10)
    pxl_thresh.insert(END, default_config['pxl_thresh'])
    pxl_thresh.grid(column=1, row=5)

    ell_thresh_label = Label(params, text="Maximum ratio of ellipse shortaxis to longaxis during ellipse fit of pupil:")
    ell_thresh_label.grid(column=0, row=6)
    ell_thresh = Entry(params, width=10)
    ell_thresh.insert(END, default_config['ell_thresh'])
    ell_thresh.grid(column=1, row=6)

    eye_dist_thresh_cm_label = Label(params, text="Maximum acceptable distance from mean position of a point that any frame's point can be (in cm):")
    eye_dist_thresh_cm_label.grid(column=0, row=7)
    eye_dist_thresh_cm = Entry(params, width=10)
    eye_dist_thresh_cm.insert(END, default_config['eye_dist_thresh_cm'])
    eye_dist_thresh_cm.grid(column=1, row=7)

    eyecam_pxl_per_cm_label = Label(params, text="Scale factor for camera from pixels to cm on eye:")
    eyecam_pxl_per_cm_label.grid(column=0, row=8)
    eyecam_pxl_per_cm = Entry(params, width=10)
    eyecam_pxl_per_cm.insert(END, default_config['eyecam_pxl_per_cm'])
    eyecam_pxl_per_cm.grid(column=1, row=8)

    range_radius_label = Label(params, text="Acceptable range in radius while calculating omega:")
    range_radius_label.grid(column=0, row=9)
    range_radius = Entry(params, width=10)
    range_radius.insert(END, default_config['range_radius'])
    range_radius.grid(column=1, row=9)

    num_ellipse_pts_needed_label = Label(params, text="Number of ellipse points (excluding tear duct) that are needed before an ellipse can confidenlty be plotted over a frame:")
    num_ellipse_pts_needed_label.grid(column=0, row=10)
    num_ellipse_pts_needed = Entry(params, width=10)
    num_ellipse_pts_needed.insert(END, default_config['num_ellipse_pts_needed'])
    num_ellipse_pts_needed.grid(column=1, row=10)

    save_label = Label(params, text='Topdown options:', wraplength=500)
    save_label.grid(column=0, row=11)

    cricket_label = Label(params, text="Recording includes cricket?")
    cricket_label.grid(column=0, row=12)
    cricket = BooleanVar()
    cricket1 = Checkbutton(params, variable=cricket)
    cricket1.grid(column=1, row=12)

    hf_label = Label(params, text='Headfixed options:', wraplength=500)
    hf_label.grid(column=0, row=13)

    optical_mouse_defaults = default_config['optical_mouse_screen_center']

    hf_ball_label = Label(params, text="To what coordinates does the optical mouse get reset for headfixed recordings (i.e. where is the center of the screen)?")
    hf_ball_label.grid(column=0, row=14)
    hf_ball_xlabel = Label(params, text="x:")
    hf_ball_xlabel.grid(column=0, row=15)
    hf_ball_x = Entry(params, width=10)
    hf_ball_x.insert(END, optical_mouse_defaults['x'])
    hf_ball_x.grid(column=1, row=15)
    hf_ball_ylabel = Label(params, text="y:")
    hf_ball_ylabel.grid(column=0, row=16)
    hf_ball_y = Entry(params, width=10)
    hf_ball_y.insert(END, optical_mouse_defaults['y'])
    hf_ball_y.grid(column=1, row=16)

    optical_mouse_pix2cm_label = Label(params, text="Scale factor for optical mouse pixels to cm:")
    optical_mouse_pix2cm_label.grid(column=0, row=17)
    optical_mouse_pix2cm = Entry(params, width=10)
    optical_mouse_pix2cm.insert(END, default_config['optical_mouse_pix2cm'])
    optical_mouse_pix2cm.grid(column=1, row=17)

    optical_mouse_sample_rate_ms_label = Label(params, text="Optical mouse sample rate in ms:")
    optical_mouse_sample_rate_ms_label.grid(column=0, row=18)
    optical_mouse_sample_rate_ms = Entry(params, width=10)
    optical_mouse_sample_rate_ms.insert(END, default_config['optical_mouse_sample_rate_ms'])
    optical_mouse_sample_rate_ms.grid(column=1, row=18)

    misc_label = Label(params, text='Misc. options:', wraplength=500)
    misc_label.grid(column=0, row=19)

    ephys_sample_rate_label = Label(params, text="Sample rate for electrophysiology:")
    ephys_sample_rate_label.grid(column=0, row=20)
    ephys_sample_rate = Entry(params, width=10)
    ephys_sample_rate.insert(END, default_config['ephys_sample_rate'])
    ephys_sample_rate.grid(column=1, row=20)

    imu_sample_rate_label = Label(params, text="Sample rate for IMU:")
    imu_sample_rate_label.grid(column=0, row=21)
    imu_sample_rate = Entry(params, width=10)
    imu_sample_rate.insert(END, default_config['imu_sample_rate'])
    imu_sample_rate.grid(column=1, row=21)

    imu_downsample_label = Label(params, text="Factor to downsample IMU data by:")
    imu_downsample_label.grid(column=0, row=22)
    imu_downsample = Entry(params, width=10)
    imu_downsample.insert(END, default_config['imu_downsample'])
    imu_downsample.grid(column=1, row=22)

    exe_label = Label(params, text='Execution options:', wraplength=500)
    exe_label.grid(column=2, row=3)

    run_pupil_rotation_label = Label(params, text="Get eye rotation (slow!)?")
    run_pupil_rotation_label.grid(column=2, row=4)
    run_pupil_rotation = BooleanVar()
    run_pupil_rotation1 = Checkbutton(params, variable=run_pupil_rotation)
    run_pupil_rotation1.grid(column=3, row=4)

    run_top_angles_label = Label(params, text="Get topdown head and body angle?")
    run_top_angles_label.grid(column=2, row=5)
    run_top_angles = BooleanVar()
    run_top_angles1 = Checkbutton(params, variable=run_top_angles)
    run_top_angles1.grid(column=3, row=5)

    save_label = Label(params, text='Save options:', wraplength=500)
    save_label.grid(column=2, row=6)

    dwnsmpl_label = Label(params, text="Factor to downsample videos by before compressing them into the .nc file:")
    dwnsmpl_label.grid(column=2, row=7)
    dwnsmpl = Entry(params, width=10)
    dwnsmpl.insert(END, default_config['dwnsmpl'])
    dwnsmpl.grid(column=3, row=7)

    save_figs_label = Label(params, text="Save diagnostic figures?")
    save_figs_label.grid(column=2, row=8)
    save_figs = BooleanVar()
    save_figs1 = Checkbutton(params, variable=save_figs)
    save_figs1.grid(column=3, row=8)

    save_nc_vids_label = Label(params, text="Compress full videos into .nc files?")
    save_nc_vids_label.grid(column=2, row=9)
    save_nc_vids = BooleanVar()
    save_nc_vids1 = Checkbutton(params, variable=save_nc_vids)
    save_nc_vids1.grid(column=3, row=9)

    num_save_frames = None
    def clicked_avi_vid_button():
        if save_avi_vids.get() is True:
            num_save_frames_label = Label(params, text="Number of frames to save into diagnostic .avi videos:")
            num_save_frames_label.grid(column=2, row=11)
            num_save_frames = Entry(params, width=10)
            num_save_frames.insert(END, default_config['num_save_frames'])
            num_save_frames.grid(column=3, row=11)

    save_avi_vids_label = Label(params, text="Save short diagnostic .avi videos?")
    save_avi_vids_label.grid(column=2, row=10)
    save_avi_vids = BooleanVar()
    save_avi_vids1 = Checkbutton(params, variable=save_avi_vids, command=clicked_avi_vid_button)
    save_avi_vids1.grid(column=3, row=10)

    # camera types to analyze
    save_avi_vids_label = Label(params, text="Camera views to include while getting parameters?")
    save_avi_vids_label.grid(column=2, row=12)

    params_c1_label = Label(params, text="First Camera:")
    params_c1_label.grid(column=2,row=13)
    params.c1 = StringVar()
    params.c1.set(camera_options[0])
    params_c1 = OptionMenu(params, params.c1, *camera_options)
    params_c1.grid(column=3, row=13)

    params_c2_label = Label(params, text="Second Camera:")
    params_c2_label.grid(column=2,row=14)
    params.c2 = StringVar()
    params.c2.set(camera_options[0])
    params_c2 = OptionMenu(params, params.c2, *camera_options)
    params_c2.grid(column=3, row=14)

    params_c3_label = Label(params, text="Third Camera:")
    params_c3_label.grid(column=2,row=15)
    params.c3 = StringVar()
    params.c3.set(camera_options[0])
    params_c3 = OptionMenu(params, params.c3, *camera_options)
    params_c3.grid(column=3, row=15)

    params_c4_label = Label(params, text="Fourth Camera:")
    params_c4_label.grid(column=2,row=16)
    params.c4 = StringVar()
    params.c4.set(camera_options[0])
    params_c4 = OptionMenu(params, params.c4, *camera_options)
    params_c4.grid(column=3, row=16)

    params_c5_label = Label(params, text="Fifth Camera:")
    params_c5_label.grid(column=2,row=17)
    params.c5 = StringVar()
    params.c5.set(camera_options[0])
    params_c5 = OptionMenu(params, params.c5, *camera_options)
    params_c5.grid(column=3, row=17)

    params_c6_label = Label(params, text="Sixth Camera:")
    params_c6_label.grid(column=2,row=18)
    params.c6 = StringVar()
    params.c6.set(camera_options[0])
    params_c6 = OptionMenu(params, params.c6, *camera_options)
    params_c6.grid(column=3, row=18)

    ### extra parameters

    led_label = Label(addtl_params, text='Track the movement of IR LEDs in a dark room both on the surface of the pupil and in space from the perspective of the worldcam.', wraplength=500)
    led_label.grid(column=0, row=0)

    ledW = None
    ledE = None
    LED_dir_name = None
    strict_lik_thresh = None

    def clicked_run_led_tracking():
        # LED configs
        ledW_label = Label(addtl_params, text="World camera DLC .yaml path for IR LED tracking:")
        ledW_label.grid(column=0,row=2)
        def clicked_ledW_button():
            global ledW
            ledW = filedialog.askopenfilename(title='Select the IR LED tracking .yaml path for worldcam:')
            print('World IR LED .yaml path set to: ' + ledW)
        ledW_button = Button(addtl_params, text="browse", command=clicked_ledW_button)
        ledW_button.grid(column=1, row=2)

        ledE_label = Label(addtl_params, text="eye camera DLC .yaml path for IR LED tracking:")
        ledE_label.grid(column=0,row=3)
        def clicked_ledE_button():
            global ledE
            ledE = filedialog.askopenfilename(title='Select the IR LED tracking .yaml path for eyecam:')
            print('eye IR LED .yaml path set to: ' + ledE)
        ledE_button = Button(addtl_params, text="browse", command=clicked_ledE_button)
        ledE_button.grid(column=1, row=3)

        # name of IR LED directory
        LED_dir_name_label = Label(addtl_params, text="Name of the directory that contains the IR LED calibration videos (e.g. hf3_IRspot):")
        LED_dir_name_label.grid(column=0, row=4)
        LED_dir_name = Entry(addtl_params, width=20)
        LED_dir_name.insert(END, default_config['LED_dir_name'])
        LED_dir_name.grid(column=1, row=4)

        strict_lik_thresh_label = Label(addtl_params, text="Strict likelihood threshold:")
        strict_lik_thresh_label.grid(column=0, row=5)
        strict_lik_thresh = Entry(addtl_params, width=8)
        strict_lik_thresh.insert(END, default_config['lik_thresh_strict'])
        strict_lik_thresh.grid(column=1, row=5)

    run_addtl_params_label = Label(addtl_params, text="Track IR LED position in world and eye cameras?")
    run_addtl_params_label.grid(column=0, row=1)
    run_addtl_params = BooleanVar()
    run_addtl_params1 = Checkbutton(addtl_params, variable=run_addtl_params, command=clicked_run_led_tracking)
    run_addtl_params1.grid(column=1, row=1)

    ### config tab
    config_label = Label(config_tab, text="First, write the parameters you have entered into a .json file. This will be written into the animal directory that you entered on the second page of this program. When you write to file, the config file that's written will be printed into the terminal. It's a good idea to read through this and make sure everything was entered and saved correctly.", wraplength=500)
    config_label.grid(column=0, row=0)

    json_path = None
    def write_to_file():
        c1_yaml_path, c2_yaml_path, c3_yaml_path, c4_yaml_path, c5_yaml_path, c6_yaml_path = [None if (c not in globals) or (c not in locals) else c for c in [c1_yaml_path, c2_yaml_path, c3_yaml_path, c4_yaml_path, c5_yaml_path, c6_yaml_path]]

        user_entries = {
            'data_path': data_path,
            'deinterlace': run_deinter,
            'flip_eyecam': deinter_eye,
            'flip_worldcam': deinter_world,
            'get_checker_calib': checker,
            'world_checker_avi_path': world_checker,
            'top_checker_avi_path': top_checker,
            'npz_save_path': npz_save_dir,
            'top_npz_save_name': top_npz_name,
            'world_npz_save_name': world_npz_name,
            'undistort_vids': undistort,
            'top_npz_read_dir': top_npz_read_dir,
            'world_npz_read_dir': world_npz_read_dir,
            'undistort_vids_use_paths_from_current_session': undistort_CS,
            'cam_inputs':{
                dlc.c1.get(): c1_yaml_path,
                dlc.c2.get(): c2_yaml_path,
                dlc.c3.get(): c3_yaml_path,
                dlc.c4.get(): c4_yaml_path,
                dlc.c5.get(): c5_yaml_path,
                dlc.c6.get(): c6_yaml_path
            },
            'param_cams':[params.c1.get(),params.c2.get(),params.c3.get(),params.c4.get(),params.c5.get(),params.c6.get()],
            'has_ephys': False,
            'lik_thresh':lik_thresh,
            'tear':tear,
            'pxl_thresh':pxl_thresh,
            'ell_thresh':ell_thresh,
            'eye_dist_thresh_cm':eye_dist_thresh_cm,
            'eyecam_pxl_per_cm':eyecam_pxl_per_cm,
            'range_radius':range_radius,
            'world_interp_method':'linear',
            'num_ellipse_pts_needed':num_ellipse_pts_needed,
            'cricket':cricket,
            'hf_ball_x':hf_ball_x,
            'hf_ball_y':hf_ball_y,
            'optical_mouse_pix2cm':optical_mouse_pix2cm,
            'optical_mouse_sample_rate_ms':optical_mouse_sample_rate_ms,
            'use_BonsaiTS':True,
            'ephys_sample_rate':ephys_sample_rate,
            'imu_sample_rate':imu_sample_rate,
            'imu_downsample':imu_downsample,
            'run_pupil_rotation':run_pupil_rotation,
            'run_top_angles':run_top_angles,
            'dwnsmpl':dwnsmpl,
            'save_figs':save_figs,
            'save_nc_vids':save_nc_vids,
            'num_save_frames':num_save_frames,
            'save_avi_vids':save_avi_vids,
            'ledW':ledW,
            'ledE':ledE,
            'LED_dir_name':LED_dir_name,
            'run_addtl_params':run_addtl_params,
            'run_with_form_time':True,
            'run_params': run_params,
            'crop_vids': crop_vids,
            'multiTOP': multiTOP,
            'strict_lik_thresh': strict_lik_thresh
        }
        user_entries_opened = {key:val.get() for (key,val) in user_entries.items() if isinstance(val, BooleanVar) is True or isinstance(val, Entry) is True}
        user_entries_opened1 = {key:val for (key,val) in user_entries.items() if key not in user_entries_opened}
        user_entries_opened.update(user_entries_opened1)
        json_path = write_config(user_entries_opened)

    write_label = Label(config_tab, text="Write options to file:", wraplength=500)
    write_label.grid(column=0, row=1)
    write_button = Button(config_tab, text="write", command=write_to_file)
    write_button.grid(column=1, row=1)

    ### run tab
    run_label = Label(run, text="Now the preprocessing can be executed using the config file you have provided or built. Progress will be reported in the terminal in which this window was opened. Do not close this window or the terminal while it runs.", wraplength=500)
    run_label.grid(column=0, row=0)

    def run_pipeline():
        if json_path is not None:
            print('starting preprocessing')
            run_auto_preprocessing(json_path)
        else:
            print('could not find a json path -- has a config file been provided or written already?')

    run_label = Label(run, text="Run preprocessing:", wraplength=500)
    run_label.grid(column=0, row=1)
    run_button = Button(run, text="run!", command=run_pipeline)
    run_button.grid(column=1, row=1)

    window.mainloop()

    

    