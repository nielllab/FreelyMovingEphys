"""
__main__.py
"""
from prelim_whitenoise.prelim_whitenoise import main as run_prelim_whitenoise
import tkinter as tk
from tkinter import filedialog
from tkinter import *
from tkinter import ttk

root = tk.Tk()
root.withdraw()
whitenosie_directory = filedialog.askdirectory(title='Select the headfixed whitenoise directory')
window = Tk()
window.title('Select probe mapping')
window.geometry("500x300+250+100")
tab_control = ttk.Notebook(window)
main = ttk.Frame(tab_control)
tab_control.add(main, text='')
tab_control.pack(expand=1, fill='both')
# probe options
probes = ['default16', 'NN_H16', 'default64', 'NN_H64-LP', 'DB_P64-3', 'DB_P64-8', 'DB_P128-6']
probe = StringVar()
probe.set(probes[0])
probe_menu = OptionMenu(main, probe, *probes)
probe_menu.grid(column=0, row=0)
# when 'select' button is clicked, run preliminary whitenoise analysis
def run_main():
    run_prelim_whitenoise(whitenosie_directory, probe.get())
    root.destroy()
run_button = Button(main, text="select", command=run_main)
run_button.grid(column=1, row=0)
window.mainloop()