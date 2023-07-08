import os
import shutil
import argparse
import pandas as pd
import numpy as np
import PySimpleGUI as sg
import fmEphys as fme


def apply_bias_fix(bin_path, savepath):
    dtypes = np.dtype([
        ("acc_x",np.uint16),
        ("acc_y",np.uint16),
        ("acc_z",np.uint16),
        ("none1",np.uint16),
        ("gyro_x",np.uint16),
        ("gyro_y",np.uint16),
        ("gyro_z",np.uint16),
        ("none2",np.uint16)
    ])

    # read in binary file
    binary_in = pd.DataFrame(np.fromfile(bin_path, dtypes, -1, ''))

    # convert to voltage (from int16 in binary file)
    data = 10 * (binary_in.astype(float) / (2**16) - 0.5)

    # shift values
    bias_shifted = data.copy() * 2

    # convert back to int16
    data_out = ((bias_shifted * (2**16) + 0.5) / 10).astype(np.uint16)

    # write the binary file out
    data_out.to_numpy().tofile(savepath)

    print('Raw data moved to {}.\n Corrected data saved to {}'.format(bin_path, savepath))

def setup_paths(bin_path):

    rec_dir = fme.up_dir(bin_path,2)
    uncorbias_dir = os.path.join(rec_dir, 'uncorrected_bias_IMU')
    if not os.path.exists(uncorbias_dir):
        os.makedirs(uncorbias_dir)
    moved_bin_path = os.path.join(uncorbias_dir, os.path.split(bin_path)[1])

    # move the raw binary file
    shutil.move(bin_path, uncorbias_dir)

    return moved_bin_path


def main(bin_path):

    if bin_path is None:

        bin_path = sg.popup_get_file('Pick IMU binary file',
                    title='Pick IMU binary file',
                    file_types=(('BIN,','*.bin'),),
                    no_window=True)
        
    moved_bin_path = setup_paths(bin_path)

    apply_bias_fix(moved_bin_path, bin_path)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Fix IMU bias')
    parser.add_argument('-p', '--path', type=str, default=None)
    args = parser.parse_args()

    main(args.path)
