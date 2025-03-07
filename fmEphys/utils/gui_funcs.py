
import tkinter as tk
from tkinter import filedialog

def select_file(title, filetypes):
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title=title,
        filetypes=filetypes
    )
    return file_path

def select_directory(title):
    root = tk.Tk()
    root.withdraw()
    directory_path = filedialog.askdirectory(
        title=title,
    )
    return directory_path

def get_string_input(title):
    root = tk.Tk()
    root.minsize(width=300, height= 20)
    root.title(title)
    string_var = tk.StringVar()
    entry = tk.Entry(root, textvariable=string_var, )
    entry.pack()
    result = ""
    def on_button_click():
        nonlocal result
        result = string_var.get()
        root.destroy()
    button = tk.Button(root, text="Enter", command=on_button_click)
    button.pack()
    root.mainloop()
    return result

