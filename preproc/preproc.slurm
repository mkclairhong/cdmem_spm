#!/bin/bash

#SBATCH --mail-user=jin.young.jeon@vanderbilt.edu
#SBATCH --mail-type=ALL
#SBATCH --job-name=cdcatmr_RTMpreproc_array
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=15:00:00
#SBATCH --mem=10G
#SBATCH --cpus-per-task=12
#SBATCH --array=9,24
#SBATCH --output=/home/jeonj1/proj/cdcat_spm/slurm_output/cdcatmr%a_RTMpreproc_%A.out


module load MATLAB/2019a   # load the default version of Matlab

echo "SLURM_JOBID: " $SLURM_JOBID
echo "SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "SLURM_ARRAY_JOB_ID: " $SLURM_ARRAY_JOB_ID


# for anatomical scans
matlab -nodisplay -nosplash -r "addpath(genpath('/home/jeonj1/proj/cdcat_spm')), split_imgs_anat(${SLURM_ARRAY_TASK_ID})"

# for functional scans
matlab -nodisplay -nosplash -r "addpath(genpath('/home/jeonj1/matlab/spm12')), split_imgs_func(${SLURM_ARRAY_TASK_ID})"

# to register to first scan image, append 0 or 'rtf'
# to register to mean image, append 1 or 'rtm'
matlab -nodisplay -nosplash -r "addpath(genpath('/home/jeonj1/proj/cdcat_spm')), addpath(genpath('/home/jeonj1/matlab/spm12')), preproc_cdcatmr(${SLURM_ARRAY_TASK_ID}, 1),quit()"

echo "ALL DONE!"
