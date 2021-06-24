"""
launch_gui.py

launch GUI to prepare ephys analysis

Jan. 17, 2021
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
from project_analysis.ephys.analyze_ephys import find_files, run_ephys_analysis

def launch_ephys_gui():

    # set up GUI window
    window = Tk()
    window.title('FreelyMovingEphys - Ephys Analysis')

    tab_control = ttk.Notebook(window)

    entry = ttk.Frame(tab_control)
    run = ttk.Frame(tab_control)

    tab_control.add(entry, text='Entry')
    tab_control.add(run, text='Run!')

    tab_control.pack(expand=1, fill='both')

    ### entry tab

    # user selects a recording directrory
    data_path_label = Label(entry, text='Select a recording directory:')
    data_path_label.grid(column=0, row=0)
    def clicked_data_button():
        global data_path
        data_path = filedialog.askdirectory(title='Select a recording directory:')
        print('recording directory set to ' + data_path)
    data_path_button = Button(entry, text="browse", command=clicked_data_button)
    data_path_button.grid(column=1, row=0)

    # recording name entry field
    recording_name_label = Label(entry, text="Name of the recording to analyze (e.g. 113020_G6H27P8LT_control_Rig2_fm1):")
    recording_name_label.grid(column=0, row=1)
    recording_name = Entry(entry, width=40)
    recording_name.grid(column=1, row=1)

    # unit number to highlight
    this_unit_label = Label(entry, text="Unit to highlight:")
    this_unit_label.grid(column=0, row=2)
    this_unit = Entry(entry, width=10)
    this_unit.insert(END, 0)
    this_unit.grid(column=1, row=2)

    # checkbox for freemoving or headfixed
    fm_label = Label(entry, text="Freely moving recording?")
    fm_label.grid(column=0, row=3)
    fm = BooleanVar()
    fm1 = Checkbutton(entry, variable=fm)
    fm1.grid(column=1, row=3)

    # menu of stims
    hf_stim_options = ['None','gratings','sparse_noise','white_noise','revchecker']
    stim_type_label = Label(entry, text="Specific headfixed stimulus?")
    stim_type_label.grid(column=0, row=4)
    stim_type = StringVar()
    stim_type.set(hf_stim_options[0])
    stim_type1 = OptionMenu(entry, stim_type, *hf_stim_options)
    stim_type1.grid(column=1, row=4)

    # checkbox for saving out videos
    mp4_label = Label(entry, text="Save .mp4 video?")
    mp4_label.grid(column=0, row=5)
    mp4 = BooleanVar()
    mp4_1 = Checkbutton(entry, variable=mp4)
    mp4_1.grid(column=1, row=5)

    # menu of probe ch nums
    ch_count_options = ['default16', 'NN_H16', 'default64', 'NN_H64-LP', 'DB_P64-3', 'DB_P64-8']
    ch_count_label = Label(entry, text="probe mapping?")
    ch_count_label.grid(column=0, row=6)
    ch_count = StringVar()
    ch_count.set(ch_count_options[0])
    ch_count1 = OptionMenu(entry, ch_count, *ch_count_options)
    ch_count1.grid(column=1, row=6)

    ### run tab
    run_label = Label(run, text="Do not close this window or the terminal while the analysis runs.", wraplength=500)
    run_label.grid(column=0, row=0)

    def run_pipeline():
        file_dict = find_files(data_path, recording_name.get(), fm.get(), int(this_unit.get()), stim_type.get(), mp4.get(), ch_count.get())
        run_ephys_analysis(file_dict)

    run_label = Label(run, text="Run ephys analysis:", wraplength=500)
    run_label.grid(column=0, row=1)
    run_button = Button(run, text="run!", command=run_pipeline)
    run_button.grid(column=1, row=1)

    window.mainloop()