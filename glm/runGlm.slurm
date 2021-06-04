#!/bin/bash

#SBATCH --mail-user=jin.young.jeon@vanderbilt.edu
#SBATCH --mail-type=ALL
#SBATCH --job-name=create_glm_event
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=4:00:00
#SBATCH --mem=10G
#SBATCH --cpus-per-task=12
#SBATCH --array=18-28
#SBATCH --output=/home/jeonj1/proj/cdcat_spm/slurm_output/create_GLMevents%a_%A.out


module load MATLAB/2019a   # load the default version of Matlab

echo "JOB ID Summary:"
echo "SLURM_JOBID: " $SLURM_JOBID
echo "SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "SLURM_ARRAY_JOB_ID: " $SLURM_ARRAY_JOB_ID


# create events data
echo "Let's run createEvents.m" 
matlab -nodisplay -nosplash -r "createEvents(${SLURM_ARRAY_TASK_ID})"

echo "Let's run makeRunMats.m"
matlab -nodisplay -nosplash -r "makeRunMats(${SLURM_ARRAY_TASK_ID})"

# echo "Let's run est_model_job.m"
# matlab -nodisplay -nosplash -r "addpath(genpath('/home/jeonj1/proj/cdcat_spm')), addpath(genpath('/home/jeonj1/matlab/spm12')), est_model_job(${SLURM_ARRAY_TASK_ID})"

# echo "Let's rename the beta study images"
# matlab -nodisplay -nosplash -r "rename_beta(${SLURM_ARRAY_TASK_ID}),quit()"

echo "JOB ID Summary:"
echo "SLURM_JOBID: " $SLURM_JOBID
echo "SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "SLURM_ARRAY_JOB_ID: " $SLURM_ARRAY_JOB_ID

echo "ALL DONE!"
