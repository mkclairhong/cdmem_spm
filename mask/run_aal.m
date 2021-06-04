% run script to make binary aal mask
%%%%%%%%%%%%%%%
% single version: hard code below to tailor the mask you want

addpath(genpath('/home/jeonj1/matlab/spm12'));

aal_path = '/home/jeonj1/matlab/spm12/toolbox/AAL3/AAL3.nii';

% out_path = '/scratch/polynlab/fmri/cdcatmr/all_subjects/hippocampus_lr.nii';

% roi_num = [41 42];

% write_binary_aal_mask(aal_path,roi_num,out_path);


%%%%%%%%%%%%%%%
% loop version: this will read in the atlas text file and create all LR masks
out_path_base = '/scratch/polynlab/fmri/cdcatmr/all_subjects/masks/raw_masks/';

roi_file = fopen('/scratch/polynlab/fmri/cdcatmr/all_subjects/masks/AAL3atlas_roi.txt');
roi = textscan(roi_file, '%d %s %d');
fclose(roi_file);

index = [1:length(roi{1})];
for i=1:length(index)
    odds = index(mod(index,2)==1);
end 

for i=1:length(odds)
    roi_num = [odds(i) odds(i) + 1];
    roi_name = char(roi{2}(odds(i)));
    roi_name = roi_name(1:end-2);
    out_path = strcat(out_path_base, roi_name, "_lr.nii");
    write_binary_aal_mask(aal_path, roi_num, char(out_path));
end