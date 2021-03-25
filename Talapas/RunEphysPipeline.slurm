#!/bin/bash
#SBATCH --job-name=Ephys      ### Job Name
#SBATCH --partition=gpu       ### Quality of Service (like a queue in PBS)
#SBATCH --time=24:00:00       ### Wall clock time limit in Days-HH:MM:SS
#SBATCH --nodes=1             ### Node count required for the job
#SBATCH --ntasks-per-node=1   ### Nuber of tasks to be launched per Node
#SBATCH --gres=gpu:1          ### General REServation of gpu:number of gpus
#SBATCH --mem=120G            ### memory limit per node, in GB
#SBATCH --cpus-per-task=28    ### number of cores for each task
####SBATCH --array=0-15       ### Array index
#SBATCH --account=niell       
#SBATCH --verbose
#SBATCH -o /gpfs/projects/niell/nlab/OutFiles/slurm-%A_%a.out

##turn on e-mail notification
#SBATCH --mail-type=ALL
#SBATCH --mail-user=elliottabe@gmail.com

module load tensorflow #anaconda3/2019.07 cuda/10.1
conda deactivate
conda activate /gpfs/projects/niell/nlab/DLC_GPU2
set -x

while getopts t: flag
do
    case "${flag}" in
        t) date_ani=${OPTARG};;
    esac
done

full_talapas_path="/gpfs/projects/niell/nlab/freely_moving_ephys/ephys_recordings/$date_ani/"
full_lab_path="/volume1/nlab-nas/Phil/freely_moving_ephys/$date_ani/"

mkdir -p $full_talapas_path
rsync -ahSP nlabadmin@184.171.85.98:$full_lab_path $full_talapas_path --no-compress

python -u /gpfs/projects/niell/nlab/FreelyMovingEphys/manual_preprocessing_talapas.py --config_path=/gpfs/projects/niell/nlab/FreelyMovingEphys/Talapas/preprocessing_config.json --data_path=$full_talapas_path

# rm -rf $full_talapas_path
rsync -ahSP $full_talapas_path nlabadmin@184.171.85.98:$full_lab_path --no-compress

