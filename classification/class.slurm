#!/bin/bash

#SBATCH --mail-user=jin.young.jeon@vanderbilt.edu
#SBATCH --mail-type=ALL
#SBATCH --job-name=class
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=2:00:00
#SBATCH --mem=10G
#SBATCH --cpus-per-task=12
#SBATCH --array=2-15
#SBATCH --output=/home/jeonj1/class_output/class_cdcatmr%a_jobid%A.out

echo "SLURM_JOBID: " $SLURM_JOBID
echo "SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "SLURM_ARRAY_JOB_ID: " $SLURM_ARRAY_JOB_ID

source activate py3
# run below to compare rfunc vs. wrfunc
python -c "from class_byroi import *; class_byroi($SLURM_ARRAY_TASK_ID)"
