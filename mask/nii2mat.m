% quick script to convert all nii files in a given directory to mat file
% local 
% niiDir = '/Users/Jin/Documents/MATLAB/research/fmri_scratch/resliced';
% accre
niiDir = '/scratch/polynlab/fmri/cdcatmr/all_subjects/masks/resliced/';
chdir(niiDir);
ls = dir(niiDir);

for j=1:length(ls)
    if ~ls(j).isdir && all(ls(j).name(end-3:end)=='.nii')
        v = niftiread(ls(j).name);
        save([ls(j).name(1:end-4), '.mat'], 'v');
    end
end