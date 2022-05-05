import xarray as xr
from scipy.io import savemat
import os
import PySimpleGUI as sg

def nc2mat():
    f = sg.popup_get_file('Choose .nc file.')
    data = xr.open_dataset(f)
    data_dict = dict(zip(list(data.REYE_ellipse_params['ellipse_params'].values), [data.REYE_ellipse_params.sel(ellipse_params=p).values for p in list(data.REYE_ellipse_params['ellipse_params'].values)]))
    save_name = os.path.join(os.path.split(f)[0], os.path.splitext(os.path.split(f)[1])[0])+'.mat'
    print('saving {}'.format(save_name))
    savemat(save_name, data_dict)