#!/bin/bash
#SBATCH --job-name=DLC      ### Job Name
#SBATCH --partition=longgpu       ### Quality of Service (like a queue in PBS)
#SBATCH --time=1-12:00:00       ### Wall clock time limit in Days-HH:MM:SS
#SBATCH --nodes=1             ### Node count required for the job
#SBATCH --ntasks-per-node=1   ### Nuber of tasks to be launched per Node
#SBATCH --gres=gpu:1          ### General REServation of gpu:number of gpus
#SBATCH --mem=60G            ### memory limit per node, in GB
#SBATCH --cpus-per-task=28    ### number of cores for each task
####SBATCH --array=0-15       ### Array index
#SBATCH --account=niell       
#SBATCH --verbose
#SBATCH -o /gpfs/projects/niell/nlab/OutFiles/slurm-%A_%a.out

##turn on e-mail notification
#SBATCH --mail-type=ALL
#SBATCH --mail-user=nlab@uoregon.com

module load tensorflow 
conda deactivate
conda activate /gpfs/projects/niell/nlab/DLC_GPU2
set -x

while getopts u:c: flag
do
    case "${flag}" in
        u) user=${OPTARG};;
        c) config=${OPTARG};;
    esac
done

full_talapas_path="/gpfs/projects/niell/nlab/DLCProjects/$user/$config/"
full_lab_path="/volume1/nlab-nas/DLCProjects/$user/$config/"

mkdir -p $full_talapas_path
rsync -ahSP goeppert:$full_lab_path $full_talapas_path --no-compress

python -u /gpfs/projects/niell/nlab/FreelyMovingEphys/Talapas/DLC/TrainNetwork.py --config_path="$full_talapas_path/config.yaml" --data_path="$full_talapas_path/videos"

rsync -ahSP $full_talapas_path goeppert:$full_lab_path --no-compress

# rm -rf $full_talapas_path
