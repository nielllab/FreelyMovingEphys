#!/bin/bash

while getopts t: flag
do
    case "${flag}" in
        t) date_ani=${OPTARG};;
    esac
done

full_talapas_path="/gpfs/projects/niell/nlab/freely_moving_ephys/ephys_recordings/$date_ani/"
full_lab_path="/volume1/nlab-nas/Phil/freely_moving_ephys/$date_ani/"

mkdir -p $full_talapas_path
rsync -avhSP nlabadmin@184.171.85.98:$full_lab_path $full_talapas_path --no-compress

touch "$full_talapas_path/Test2.txt"

rsync -avhSP $full_talapas_path nlabadmin@184.171.85.98:$full_lab_path --no-compress

# rm -rf $full_talapas_path

echo "$full_talapas_path"
echo "$full_lab_path"