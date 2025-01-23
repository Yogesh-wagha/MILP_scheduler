#!/bin/bash

#SBATCH --job-name=time_Scheduling      ## Name of the job
#SBATCH --output=gwemopt_milp.out    ## Output file
#SBATCH --time=36:00:00                 ## Job Duration
#SBATCH --ntasks=1                      ## Number of tasks (analyses) to run
#SBATCH --cpus-per-task=16             ## Number of threads the code will use
#SBATCH --mem-per-cpu=8G               ## Memory per CPU required by the job.
#SBATCH --account=bcrv-delta-cpu
#SBATCH --partition=cpu
eval "$(conda shell.bash hook)"   # Ensure conda shell functionality

conda activate MILP_3                  ## Activate the conda environment

srun python /u/ywagh/GWEMOPT_MILP_csv.py 