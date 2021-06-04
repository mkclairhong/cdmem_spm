#!/bin/bash

#SBATCH --mail-user=jin.young.jeon@vanderbilt.edu
#SBATCH --mail-type=ALL
#SBATCH --job-name=RecallPat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=6:00:00
#SBATCH --mem=10G
#SBATCH --cpus-per-task=12
#SBATCH --array=24-29
#SBATCH --output=/home/jeonj1/proj/cdcat_spm/slurm_output/recall_pat_%a_%A.out


module load MATLAB/2019a   # load the default version of Matlab

echo "JOB ID Summary:"
echo "SLURM_JOBID: " $SLURM_JOBID
echo "SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "SLURM_ARRAY_JOB_ID: " $SLURM_ARRAY_JOB_ID

matlab -nodisplay -nosplash -jvm -r "recall_beta2roi(${SLURM_ARRAY_TASK_ID}); exit"

source activate py3
python -c "from filter_pads import *; filter_pads($SLURM_ARRAY_TASK_ID)"

echo "ALL DONE!"
