#!/bin/bash

#SBATCH --mail-user=jin.young.jeon@vanderbilt.edu
#SBATCH --mail-type=ALL
#SBATCH --job-name=MIPlot
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=15:00:00
#SBATCH --mem=10G
#SBATCH --cpus-per-task=12
#SBATCH --array=9,24
#SBATCH --output=/home/jeonj1/proj/cdcat_spm/qa/mutual_info/mi_slurm_output/MIPlot_cdcatmr%a_jobid%A.out

echo "SLURM_JOBID: " $SLURM_JOBID
echo "SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "SLURM_ARRAY_JOB_ID: " $SLURM_ARRAY_JOB_ID

source activate py3
# run below to compare rfunc vs. wrfunc
python -c "from mi_plot_accre import *; mi_plot('$SLURM_ARRAY_TASK_ID', verbose=True)"

# run below to compare register to first vs. register to mean
# python -c "from mi_plot_accre import *; mi_plot('$SLURM_ARRAY_TASK_ID', type='coreg', verbose=True)"
