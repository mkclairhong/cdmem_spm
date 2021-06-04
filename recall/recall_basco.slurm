#!/bin/bash

#SBATCH --mail-user=jin.young.jeon@vanderbilt.edu
#SBATCH --mail-type=ALL
#SBATCH --job-name=betaseriesGLM
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=4:00:00
#SBATCH --mem=10G
#SBATCH --cpus-per-task=12
#SBATCH --array=24-29
#SBATCH --output=/home/jeonj1/proj/cdcat_spm/slurm_output/recall_basco_%a_%A.out

echo "JOB ID Summary:"
echo "SLURM_JOBID: " $SLURM_JOBID
echo "SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "SLURM_ARRAY_JOB_ID: " $SLURM_ARRAY_JOB_ID


source activate py3
python -c "from create_recall_onsets import *; create_recall_onsets($SLURM_ARRAY_TASK_ID)"


module load MATLAB/2019a   # load the default version of Matlab
echo "Let's run item_glm.m"
matlab -nodisplay -nosplash -r "addpath(genpath('/home/jeonj1/proj/cdcat_spm')), addpath(genpath('/home/jeonj1/matlab/spm12')), basco_batch('recall_glm_batch', ${SLURM_ARRAY_TASK_ID})"


echo "ALL DONE!"
