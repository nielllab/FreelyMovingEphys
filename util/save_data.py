"""
save_data.py

Data saving utilities

Last modified September 09, 2020
"""

# package imports
import os

# save xarray Dataset as a .nc file
def savetrial(data, path, name, camtype):
    if data is not None:
        # camtype should be 'TOP1', 'LEye', etc.
        dir = os.path.join(path, name)
        if not os.path.exists(dir):
            os.makedirs(dir)
        data.to_netcdf(os.path.join(dir, str(name + '_' + camtype + '.nc')))
    elif data is None:
        pass

# save xarray Dataset as a .nc file
def savecomplete(data, savedir, camext):
    # ext should describe the data, something like 'top', 'eye', etc.
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    data.to_netcdf(os.path.join(savedir, str(camext + '.nc')))
