#!/bin/bash
while getopts u:c: flag
do
    case "${flag}" in
        u) user=${OPTARG};;
        c) DLCProject=${OPTARG};;
    esac
done

full_talapas_path="/gpfs/projects/niell/nlab/DLCProjects/$DLCProject/"

python -u /gpfs/projects/niell/nlab/FreelyMovingEphys/Talapas/DLC/TrainNetwork.py --config_path="$full_talapas_path/config.yaml" --data_path="$full_talapas_path/videos"

