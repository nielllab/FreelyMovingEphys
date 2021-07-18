"""
__main__.py
"""
import yaml, os
import tkinter as tk
from tkinter import filedialog

from utils.ephys import format_spikes

root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename()

mat_path = os.path.join(os.getcwd(), file_path)

format_spikes(mat_path, 30000)