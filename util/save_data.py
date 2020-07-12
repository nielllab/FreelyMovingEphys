"""
FreelyMovingEphys data saving utilities
save_data.py

Last modified July 12, 2020
"""

# package imports
import os

# save xarray Dataset as a .nc file
def savetrial(data, path, name, camtype):
    # camtype should be 'TOP1', 'LEye', etc.
    dir = os.path.join(path, name)
    if not os.path.exists(dir):
        os.makedirs(dir)
    data.to_netcdf(os.path.join(dir, str(name + '_' + camtype + '.nc')))

# save xarray Dataset as a .nc file
def savecomplete(data, dir, ext):
    # ext should describe the data, something like 'top', 'eye', etc.
    if not os.path.exists(dir):
        os.makedirs(dir)
    data.to_netcdf(os.path.join(dir, str(ext + '.nc')))