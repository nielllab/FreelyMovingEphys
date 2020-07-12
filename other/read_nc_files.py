# open .nc files once they're saved out from the main script, load_from_DLC.py

import xarray as xr

def read_from_nc(path):
    out = xr.open_dataset(path)
    return out

main_path = '/Users/dylanmartins/data/Niell/PreyCapture/Cohort3Outputs/J463c(blue)_110719/analysis_test_02/'

out = read_from_nc(main_path)
print(out)