#!/usr/bin/env bash
#SBATCH -A ###
#SBATCH -t 2:00:00
#SBATCH -o out_real.txt
#SBATCH -e err_real.txt
#SBATCH -n 1


ml python/3.12.3
source ~/my_python/bin/activate
python real.py seed=$SLURM_ARRAY_TASK_ID $*
