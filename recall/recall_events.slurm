#!/bin/bash

#SBATCH --mail-user=jin.young.jeon@vanderbilt.edu
#SBATCH --mail-type=ALL
#SBATCH --job-name=create_glm_event
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=4:00:00
#SBATCH --mem=10G
#SBATCH --cpus-per-task=12
#SBATCH --array=2-5
#SBATCH --output=/home/jeonj1/proj/cdcat_spm/slurm_output/recall_events_%a_%A.out


module load MATLAB/2019a   # load the default version of Matlab

echo "JOB ID Summary:"
echo "SLURM_JOBID: " $SLURM_JOBID
echo "SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "SLURM_ARRAY_JOB_ID: " $SLURM_ARRAY_JOB_ID


# create events data
echo "Let's run createEvents.m" 
matlab -nodisplay -nosplash -r "recall_createEvents(${SLURM_ARRAY_TASK_ID})"

echo "Let's run makeRunMats.m"
matlab -nodisplay -nosplash -r "recall_makeRunMats(${SLURM_ARRAY_TASK_ID})"

echo "JOB ID Summary:"
echo "SLURM_JOBID: " $SLURM_JOBID
echo "SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "SLURM_ARRAY_JOB_ID: " $SLURM_ARRAY_JOB_ID

echo "ALL DONE!"
