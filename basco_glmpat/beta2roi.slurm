#!/bin/bash

#SBATCH --mail-user=jin.young.jeon@vanderbilt.edu
#SBATCH --mail-type=ALL
#SBATCH --job-name=PatItem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=6:00:00
#SBATCH --mem=10G
#SBATCH --cpus-per-task=12
#SBATCH --array=2-3,9
#SBATCH --output=/home/jeonj1/proj/cdcat_spm/slurm_output/PatItem%a_Pat_%A.out


module load MATLAB/2019a   # load the default version of Matlab

echo "JOB ID Summary:"
echo "SLURM_JOBID: " $SLURM_JOBID
echo "SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "SLURM_ARRAY_JOB_ID: " $SLURM_ARRAY_JOB_ID

matlab -nodisplay -nosplash -jvm -r "beta2roi(${SLURM_ARRAY_TASK_ID}); exit"


echo "JOB ID Summary:"
echo "SLURM_JOBID: " $SLURM_JOBID
echo "SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "SLURM_ARRAY_JOB_ID: " $SLURM_ARRAY_JOB_ID

echo "ALL DONE!"
