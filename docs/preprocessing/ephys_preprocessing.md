# Ephys Preprocessing -- User Guide

## Merge and preprocess ephys data
1. Open Matlab
2. Change the working directory to the main experiment folder
3. Run the script `preprocessEphysData.m`, filling out variables as needed
    * If 64ch, make sure code in the top cell is uncommented, and the code in the bottom cell is commented out. For 16ch, do vice versa. `Ctrl + R` for comment and `Ctrl + T` for uncomment
4. First select the `ephys.bin` files for each trial (in each subfolder) in chronological order (e.g. hf1_wn, hf2_...)
5. Hit cancel when you're finished selecting files
6. Enter the name the merged file to save out (usually in the form: date_animal_hf_fm)
7. This will spit out two figures per trial that are automatically saved
8. When finished, type ‘close all’ to close the figures

## Split Ephys Data
1. Open a command terminal, make sure you're in the directory for the FreelyMovingEphys repository, and type: `conda activate DLC-GPU2`
2. Type: `python -m split_ephys_recordings`
3. When the dialogue box comes up, select the .mat file of the concatenated data that you just sorted, e.g. `date_animal_hf_fm.mat`
4. This will spit out a json file into each of the subfolders (check that they are there!)

## Receptive Field Mapping
1. Type: `python -m project_analysis.map_receptive_fields` (in `DLC-GPU2` conda environment)
2. When the dialogue box pops up, select the white noise folder in your experiment. This will do basic receptive field analysis and save the results into a PDF in the hf1_wn folder



