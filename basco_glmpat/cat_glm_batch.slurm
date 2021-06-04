#!/bin/bash

#SBATCH --mail-user=jin.young.jeon@vanderbilt.edu
#SBATCH --mail-type=ALL
#SBATCH --job-name=cat_betaseriesGLM
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=4:00:00
#SBATCH --mem=10G
#SBATCH --cpus-per-task=12
#SBATCH --array=3
#SBATCH --output=/home/jeonj1/proj/cdcat_spm/slurm_output/cat_betaSeries%a_glm_%A.out


module load MATLAB/2019a   # load the default version of Matlab

echo "JOB ID Summary:"
echo "SLURM_JOBID: " $SLURM_JOBID
echo "SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "SLURM_ARRAY_JOB_ID: " $SLURM_ARRAY_JOB_ID


# run create_onsets.m
matlab -nodisplay -nosplash -r "addpath(genpath('/home/jeonj1/proj/cdcat_spm')), addpath(genpath('/home/jeonj1/matlab/spm12')), create_onsets((${SLURM_ARRAY_TASK_ID})"

echo "Let's run cat_glm.m"
matlab -nodisplay -nosplash -r "addpath(genpath('/home/jeonj1/proj/cdcat_spm')), addpath(genpath('/home/jeonj1/matlab/spm12')), basco_batch('cat_glm_batch', ${SLURM_ARRAY_TASK_ID}), quit()"


echo "JOB ID Summary:"
echo "SLURM_JOBID: " $SLURM_JOBID
echo "SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "SLURM_ARRAY_JOB_ID: " $SLURM_ARRAY_JOB_ID

echo "ALL DONE!"
