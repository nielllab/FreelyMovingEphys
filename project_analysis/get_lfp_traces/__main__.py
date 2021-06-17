"""
__main__.py

launch GUI to run minimal analysis needed to get lfp traces using reversing
checkerboard stimulus
"""
from project_analysis.get_lfp_traces.initial_revchecker_analysis import quick_revchecker_analysis
import argparse, json, sys, os, subprocess, shutil
import tkinter as tk
from tkinter import filedialog
from tkinter import *
from tkinter import ttk

# select a directory to use
root = tk.Tk()
root.withdraw()
this_subject = filedialog.askdirectory(title='Select the hf revchecker dir of subject:')
# second window to select, from options, what probe was used
window = Tk()
window.title('select probe used')
tab_control = ttk.Notebook(window)
main = ttk.Frame(tab_control)
tab_control.add(main, text='main')
tab_control.pack(expand=1, fill='both')
ch_count_options = ['default16', 'NN_H16', 'default64', 'NN_H64-LP', 'DB_P64-3', 'DB_P64-8']
ch_count = StringVar()
ch_count.set(ch_count_options[0])
ch_count1 = OptionMenu(main, ch_count, *ch_count_options)
ch_count1.grid(column=0, row=0)
def run_main():
    quick_revchecker_analysis(this_subject, ch_count.get())
    root.destroy()
run_button = Button(main, text="select", command=run_main)
run_button.grid(column=1, row=0)
window.mainloop()
